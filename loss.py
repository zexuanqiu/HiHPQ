import torch
import torch.nn as nn
import torch.nn.functional as F
from random import sample

from lorentz import LorentzCalculation
              

class HyperSimCLRLoss(nn.Module):
    def __init__(self, temp, writer=None, assymetric_mode=False):
        super(HyperSimCLRLoss, self).__init__()
        # self.temperature = temperature
        self.temp = temp 
        self.lorentz_calculator = LorentzCalculation()
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
        self.writer = writer
        self.global_step = 0
        self.assymetric_mode = assymetric_mode
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def assymetric_forward(self, z_i, z_j_hat, neg_curvs: nn.parameter):
        # z_i: [b, M, D]
        # z_j_hat:[b,M,D]
        N = z_i.shape[0]
        dist = [self.lorentz_calculator.lorentz_dist(z_i[:,i,:], z_j_hat[:,i,:], neg_curvs[i]) for i in range(z_i.shape[1])] #bb
        dist = torch.stack(dist, dim=-1) # BBM
        dist = torch.sum(dist, dim=-1) # BB
        sim = -dist # BB
        sim = sim / self.temp

        labels = torch.arange(N).to(sim.device)
        logits = sim
        loss = self.criterion(logits, labels)
        loss /= N

        if self.writer is not None:
            self.global_step+=1
            avg_dist_pos = torch.diag(dist).mean()
            avg_dist_neg = (torch.sum(dist) - torch.diag(dist).sum()) / (dist.shape[0] * (dist.shape[0] -1))
            self.writer.add_scalar("pos_avg_lorentzian_dist", avg_dist_pos.item(), self.global_step)
            self.writer.add_scalar("neg_avg_lorentzian_dist", avg_dist_neg.item(), self.global_step)

        return loss 
    

    def forward(self, z_i, z_j, neg_curvs:nn.parameter, im2cluster=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2( 1) augmented examples within a minibatch as negative examples.
        """
        assert not torch.any(torch.isnan(z_i))
        assert not torch.any(torch.isnan(z_j))

        if self.assymetric_mode:
            return self.assymetric_forward(z_i=z_i, z_j_hat=z_j, neg_curvs=neg_curvs)

        self.global_step +=1

        batch_size = z_i.shape[0]
        N = 2 * batch_size

        # z_i: [b, M, D]
        # z: [2b, M, D]
        # z_i, z_j are in the hyperboloid space
        z = torch.cat((z_i, z_j), dim=0)

        
        dist = [self.lorentz_calculator.lorentz_dist(z[:,i,:], z[:,i,:], neg_curvs[i]) for i in range(z.shape[1])] #
        dist = torch.stack(dist, dim=-1) # BBM
        dist = torch.sum(dist, dim=-1) # BB
        sim = -dist # BB
        sim = sim / self.temp

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)
        if self.writer is not None:
            sim_pos = torch.mean(positive_samples)
            prob_pos = torch.exp(positive_samples)
            prob_neg = negative_samples.exp().sum(dim=-1)
            norm_prob_pos = torch.mean(prob_pos / (prob_pos + prob_neg))
            self.writer.add_scalar("sim_for_cl", sim_pos, self.global_step)
            self.writer.add_scalar("norm_prob_for_cl", norm_prob_pos, self.global_step)
        labels = torch.zeros(N).long().to(positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        aug_inst_loss = self.criterion(logits, labels)
        aug_inst_loss /= N

        def neigbor_mask(im2cluster):
            im2cluster = im2cluster.reshape(im2cluster.shape[0], 1)
            mask_mat = im2cluster == im2cluster.T # bsz * bsz
            mask_mat = mask_mat.float() # 0 or 1
            mask_mat.fill_diagonal_(0.) 
            
            indices = torch.argmax(mask_mat, dim=1)
            result_mask_mat = torch.zeros_like(mask_mat).cuda()
            result_mask_mat.scatter_(1, indices.unsqueeze(1), 1) 
            result_mask_mat = result_mask_mat * mask_mat 
            return result_mask_mat   

        if im2cluster is not None:
            neighbor_inst_loss_list = []
            for ith_im2cluster in im2cluster:         
                im_neighbor_mask = neigbor_mask(ith_im2cluster)
                not_all_zero_row_idx = im_neighbor_mask.sum(dim=1) != 0
                sim_i = sim[:batch_size, :batch_size] 
                sim_i_useful = sim_i[not_all_zero_row_idx]
                im_neighbor_mask_useful = im_neighbor_mask[not_all_zero_row_idx]
                # print(im_neighbor_mask_useful)
                labels = torch.argmax(im_neighbor_mask_useful, dim=1)
                # print("not_all_zero_rwo_idx's len is:{}".format(len(sim_i_useful)))
                # print("neighbor's num is: {}".format(im_neighbor_mask_useful.sum()))
                neighbor_inst_loss = self.criterion(sim_i_useful, labels) 
                neighbor_inst_loss /= len(im_neighbor_mask)
                neighbor_inst_loss_list.append(neighbor_inst_loss)
            
            return aug_inst_loss, sum(neighbor_inst_loss_list) / len(neighbor_inst_loss_list)


        return aug_inst_loss, torch.Tensor([0.]).cuda()



class ProtoLoss(nn.Module):

    def __init__(self, temp):
        super(ProtoLoss, self).__init__()
        self.temp = temp
        self.lorentz_calculator = LorentzCalculation()
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')

    def forward(self, view1_hat_feats, neg_curvs, tangent_to_hyper_func, clus_mode,cluster_result=None, index=None):
        """
        Input:
            view1_hat_feats: quantized embeddings of a batch of query images, [b, M, D]
            cluster_result: cluster assignments, centroids, and density. Shape of centroids (in product of tangent space) is: [num_cluster, M * D]
            index: indices for training samples
        Output:
            proto_losses: list of prototypical losses that corresponds to each num_cluster
        """
        proto_loss_list = []
        prev_pos_prototypes = None
        for step, (im2cluster, prototypes) in enumerate(zip(cluster_result['im2cluster'], 
                                                                     cluster_result['centroids'])):
            # get positive prototypes
            pos_proto_id = im2cluster[index]
            pos_prototypes = prototypes[pos_proto_id]

            # sample negative prototypes, 
            all_proto_id = [i for i in range(im2cluster.max()+1)]       
            neg_proto_id = list(set(all_proto_id)-set(pos_proto_id.tolist()))
            # neg_proto_id = sample(neg_proto_id, self.r) #sample r negative prototypes. Do not sample
            neg_prototypes = prototypes[neg_proto_id]    

            # if clus_mode == "hier_residual" and step>0:
            #     proto_selected = pos_prototypes
            # else:
            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0) # [bsz+len(neg), M*D], in tangent space

            hyper_proto_selected = tangent_to_hyper_func(proto_selected) # [bsz+len(neg), M, D] in hyper space

            # compute lorentzian dist
            dist = [self.lorentz_calculator.lorentz_dist(view1_hat_feats[:,i,:], hyper_proto_selected[:,i,:], neg_curvs[i]) for i in range(view1_hat_feats.shape[1])] # [bsz, bsz+len(neg)]
            dist = torch.stack(dist, dim=-1) # [bsz  bsz+len(neg), M]
            dist = dist.sum(dim = -1) # [bsz, bsz+len(neg)]
            logits_proto = -dist # [bsz, bsz+len(neg)]

            # targets for prototype assignment
            labels_proto = torch.linspace(0, view1_hat_feats.size(0)-1, steps=view1_hat_feats.size(0)).long().cuda()

            # scaling temperatures for the selected prototypes
            # if clus_mode == "hier_residual" and step>0:
            #     temp_proto = density[pos_proto_id]
            # else:
            logits_proto /= self.temp 
                
            # logits_proto /= temp_proto


            cur_proto_loss = self.criterion(logits_proto, labels_proto) / view1_hat_feats.shape[0]
            # cur_proto_loss = -torch.diagonal(logits_proto).mean()

            proto_loss_list.append(cur_proto_loss)

        return proto_loss_list
        
        

        



