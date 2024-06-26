import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lorentz import LorentzCalculation
import numpy as np 
import logging

class FullHyperPQhead(nn.Module):
    def __init__(self, feat_dim, M, K, softmax_temp, quant_method,
                init_neg_curvs=1.0, clip_r=1.0, use_alpha=False, writer=None):
        # MlogK bits
        # M = M; number of codebooks
        # K = K; number of codewords
        # D: D; dimensions of the codewords
        super(FullHyperPQhead, self).__init__()
        self.feat_dim, self.M, self.K, self.D = feat_dim, M, K, feat_dim // M
        self.softmax_temp, self.quant_method = softmax_temp, quant_method
        self.lorentz_calculator = LorentzCalculation()
        self.C = nn.Parameter(torch.empty((self.K, self.M * self.D)), requires_grad=True) # in Ambient Space Originally
        self.clip_r = nn.Parameter(torch.tensor(clip_r), requires_grad=False)
        nn.init.xavier_uniform_(self.C.data)
        self.writer = writer 
        self.global_step=0

        if use_alpha:
            self.alpha_scaler = nn.Parameter(torch.log(torch.sqrt(torch.Tensor([1./self.D]*M))), requires_grad=False)
        else:
            self.alpha_scaler = None
        # self.use_soft_sort = use_soft_sort
        # if self.use_soft_sort:
        #     raise NotImplementedError("Not implemented")
        #     self.soft_sort = SoftSort(tau=tau)


        if isinstance(init_neg_curvs, float):
            tmp_neg_curvs = torch.Tensor([init_neg_curvs] *M)
            self.neg_curvs = nn.Parameter(tmp_neg_curvs, requires_grad= True)
        elif isinstance(init_neg_curvs, list):
            tmp_neg_curvs = torch.Tensor(init_neg_curvs)
            self.neg_curvs = nn.Parameter(tmp_neg_curvs, requires_grad= True)

    def quant(self, i, xi, ci, i_neg_curvs):
        # x[i] is in the lorentz sapce now 
        # c[i] is in the tangent space now

        # logits: [bsz, K] unnormalized log weights in lorentz space 
        # logits = self.lorentz_calculator.lorentz_simlarity(xi, ci, i_neg_curvs)
        # calcualte the squared lorentzian distance
        sqrt_dist = self.lorentz_calculator.sqrt_lorentz_dist(xi, ci, i_neg_curvs)
        logits = -sqrt_dist

        if self.quant_method == "softmax":
            # soft_prob: [bsz, K]
            soft_prob = F.softmax(logits * self.softmax_temp, dim=1)
            xi_hat = self.lorentz_calculator.mid_point(soft_prob, ci, i_neg_curvs)
            if self.writer is not None:
                max_val,_ = torch.max(logits, dim=1)
                self.writer.add_scalar('max_val_of_logits_%d'%i, torch.mean(max_val), self.global_step)
                max_val_prob,_ = torch.max(soft_prob, dim=1)
                self.writer.add_scalar('max_val_of_prob_%d'%i, torch.mean(max_val_prob), self.global_step)
        else:
            raise NotImplementedError("Wrong Quantization Method.")
        
        return xi_hat, logits


    def encode_hyper_feats(self, x):
        x = torch.split(x, self.D, dim=1) # [[B, D]] * M
        tan_x = [self.lorentz_calculator.proj_tan0(xi) for xi in x]
        hyper_x = [self.lorentz_calculator.expmap0(v=tan_x[i], clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None) for i in range(len(tan_x))]
        hyper_x = torch.stack(hyper_x, dim=-1)
        hyper_x = torch.transpose(hyper_x, 1, 2) # [b, M, D]
        return hyper_x

    def encode_tangent_feats(self, x):
        x = torch.split(x, self.D, dim=1) # [[B, D]] * M
        tan_x = [self.lorentz_calculator.proj_tan0(xi) for xi in x]
        tan_x = torch.stack(tan_x, dim=-1)
        tan_x = torch.transpose(tan_x, 1, 2) # [b, M, D]
        tan_x = tan_x.reshape(tan_x.shape[0], - 1) # [b, M*D]
        return tan_x
    
    def tangent_to_hyper(self, x):
        # x: [b, M*D] in tangent space
        tan_x = torch.split(x, self.D, dim=1) # [[B, D]] * M
        hyper_x = [self.lorentz_calculator.expmap0(v=tan_x[i], clip_r=self.clip_r, c=self.neg_curvs[i],alpha_scaler=None) for i in range(len(tan_x))]
        hyper_x = torch.stack(hyper_x, dim=-1)
        hyper_x = torch.transpose(hyper_x, 1, 2) # [b, M, D]
        return hyper_x

    
    @torch.no_grad()
    def _codebook_normalization(self):
        # normalize the codewords
        codewords = self.C.data.clone()
        codewords = codewords.view(self.K, self.M, self.D)
        codewords = F.normalize(codewords, dim=-1)
        codewords = codewords.view(self.K , self.M * self.D)
        self.C.copy_(codewords)

    def forward(self, x):
        # tuple; ele of the tuple has the shape like (bsz, D), M element
        x = torch.split(x, self.D, dim=1)
        # tuple; ele of the tuple has the shape like (K, D), M elements.
        c = torch.split(self.C, self.D, dim = 1)
        self.global_step+=1

        x_hat = []
        codes = []
        soft_codes = []
        quant_err = []
        hyper_x = [] # test
        # v_norm_all = []

        # if self.use_soft_sort:
        #     softsort_cwd = []
        for i in range(self.M):
            # first transform xi and ci into tangent space
            if self.writer is not None:
                self.writer.add_scalar("curvature_%d"%i,self.neg_curvs[i], self.global_step)
            xi = x[i]
            ci = c[i]
            xi_tan0 = self.lorentz_calculator.proj_tan0(xi)
            xi_hyper = self.lorentz_calculator.expmap0(v=xi_tan0, clip_r=self.clip_r, c=self.neg_curvs[i],alpha_scaler=None)
            ci_tan0 = self.lorentz_calculator.proj_tan0(ci)
            ci_hyper = self.lorentz_calculator.expmap0(v=ci_tan0, clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None)

            # 保存每个ci_norm
            # v_norm_i = torch.norm(xi_tan0, dim=-1, keepdim=True)
            # #v_norm_i = torch.minimum(torch.ones_like(v_norm_i), self.clip_r/v_norm_i)*v_norm_i # clipped embeddings
            # v_norm_all.append(v_norm_i)
            xi_hat_hyper, logits_i, = self.quant(i, xi_hyper, ci_hyper, self.neg_curvs[i])
            codes_i = logits_i.argmax(dim=1,keepdim=True)
            soft_codes_i = F.softmax(logits_i * self.softmax_temp, dim=1)

            # TODO. Calculate the quantization error.
            quanti_err = self.lorentz_calculator.lorentz_dist(xi_hyper, xi_hat_hyper, self.neg_curvs[i]) # BD,BD->BB
            quanti_err = torch.diag(quanti_err) # B
            
            x_hat.append(xi_hat_hyper)
            codes.append(codes_i)
            soft_codes.append(soft_codes_i)
            quant_err.append(quanti_err)
            hyper_x.append(xi_hyper)
            # if softsort_cwd_i != None:
            #     softsort_cwd.append(softsort_cwd_i)
            
        hyper_x = torch.stack(hyper_x, dim=-1)
        hyper_x = torch.transpose(hyper_x, 1, 2)
        x_hat = torch.stack(x_hat, dim=-1) # [b, D, M]
        x_hat = torch.transpose(x_hat, 1, 2) # [b, M, D]
        codes = torch.cat(codes, dim=1) # [b, M]
        soft_codes = torch.stack(soft_codes, dim=-1) # [b, D, M]
        soft_codes = torch.transpose(soft_codes, 1, 2) #[b, M, D]
        quant_err = torch.mean(torch.stack(quant_err,0)).detach()
        # v_norm_all = torch.cat(v_norm_all, dim=1) # [b,M]

        return hyper_x, x_hat, codes, soft_codes, quant_err
    
    def hyper_codebooks(self):
        C = torch.split(self.C, self.D, dim=1)
        hyper_C = []
        for i in range(len(C)):
            ci = C[i]
            ci = self.lorentz_calculator.proj_tan0(ci)
            hyper_ci = self.lorentz_calculator.expmap0(v=ci, clip_r=self.clip_r, c=self.neg_curvs[i], alpha_scaler=None)
            hyper_C.append(hyper_ci)
        hyper_C = torch.stack(hyper_C, dim=1)
        hyper_C = hyper_C.view(hyper_C.shape[0], -1) # (K, M*D)
        return hyper_C.detach()


    
    def save_codebooks(self, path):
        # first transform the codewords to hyperbolic space, then save 
        C = torch.split(self.C, self.D, dim = 1)
        hyper_C = []
        for i in range(C):
            ci = C[i]
            ci = self.lorentz_calculator.proj_tan0(ci)
            hyper_ci = self.lorentz_calculator.expmap0(v=ci, clip_r=self.clip_r, c=self.neg_curvs[i],alpha_scaler=None)
            hyper_C.append(hyper_ci)
        with open(path, 'wb') as f:
            np.save(f, hyper_C.detach().cpu().numpy())



class HyperPQ(nn.Module):
    def __init__(self, feat_dim, M, K, softmax_temp, quant_method,
                trainable_layer_num=0, init_neg_curvs=1.0, clip_r=1.0,
                add_supp_layer=False,full_hyperpq=False, use_alpha=False, use_soft_sort=False, writer=None):
        super(HyperPQ, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        self.use_soft_sort = use_soft_sort
        
        assert trainable_layer_num <= 2 
        for i, param in enumerate(self.vgg.parameters()):
            if (i + trainable_layer_num * 2) < 30:
                param.requires_grad = False 
        
        self.projction_layer = nn.Linear(4096, feat_dim)
        self.supp_layer = nn.ModuleList([nn.Sequential(nn.Linear(feat_dim//M, feat_dim//M),
                                         nn.ReLU(), nn.Linear(feat_dim//M, feat_dim//M))for _ in range(M)]) if add_supp_layer else None 


        if full_hyperpq:
            self.hyper_pq_head = FullHyperPQhead(feat_dim=feat_dim, M=M, K=K,
                                        softmax_temp=softmax_temp, quant_method=quant_method,
                                        init_neg_curvs=init_neg_curvs,
                                        clip_r=clip_r,
                                        use_alpha=use_alpha,
                                        writer=writer)
        else:
            raise NotImplementedError()

        self.codebook_normalization()
    
    def forward(self, x, model_training = True):
        x = self.vgg.features(x)
        x = x.view(x.shape[0], -1)
        x = self.vgg.classifier(x)
        x = self.projction_layer(x)
        if self.supp_layer is not None:
            x = torch.split(x, self.hyper_pq_head.D, dim=-1)
            x = [self.supp_layer[i](x[i]) for i in range(len(x))] # [[bsz, D]] * M
            x = torch.cat(x, dim=-1)

        if model_training:
            x_hyper, x_hat, _, soft_codes, quant_err =  self.hyper_pq_head(x)
            return x_hyper, x_hat, soft_codes, quant_err
        else:
            x_hyper, _, code, _, _ = self.hyper_pq_head(x)
            # x_hyper = self.hyper_pq_head.encode_hyper_feats(x)
            return x_hyper, code

    def hyper_codebooks(self):
        return self.hyper_pq_head.hyper_codebooks()
    
    def encode_backbone_feats(self, x):
        raise NotImplementedError()
    
    def encode_tangent_feats(self, x):
        x = self.vgg.features(x)
        x = x.view(x.shape[0], -1)
        x = self.vgg.classifier(x)
        x = self.projction_layer(x)
        if self.supp_layer is not None:
            x = torch.split(x, self.hyper_pq_head.D, dim=-1)
            x = [self.supp_layer[i](x[i]) for i in range(len(x))] # [[bsz, D]] * M
            x = torch.cat(x, dim=-1)
        x_tangent = self.hyper_pq_head.encode_tangent_feats(x)
        return x_tangent

    def encode_hyper_feats(self, x):
        x = self.vgg.features(x)
        x = x.view(x.shape[0], -1)
        x = self.vgg.classifier(x)
        x = self.projction_layer(x)
        if self.supp_layer is not None:
            x = torch.split(x, self.hyper_pq_head.D, dim=-1)
            x = [self.supp_layer[i](x[i]) for i in range(len(x))] # [[bsz, D]] * M
            x = torch.cat(x, dim=-1)  
        x_hyper = self.hyper_pq_head.encode_hyper_feats(x)
        return x_hyper
    

    def codebook_normalization(self):
        self.hyper_pq_head._codebook_normalization()

    def save_codebooks(self, path):
        self.hyper_pq_head.save_codebooks(path)
    




