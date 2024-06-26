import torch 
import numpy as np 
from lorentz import LorentzCalculation
from tqdm import tqdm 
import logging
from network import HyperPQ

def save_tensor(tensor, to_path):
    with open(to_path, 'wb') as f:
        np.save(f, tensor.cpu().numpy())


def read_tensor(from_path, device='cpu'):
    with open(from_path, 'rb') as f:
        data = torch.from_numpy(np.load(f)).to(device)
    return data


def read_and_parse_file(file_path):
    data_tbl = np.loadtxt(file_path, dtype=np.str)
    data, targets = data_tbl[:, 0], data_tbl[:, 1:].astype(np.int8)
    return data, targets


@torch.no_grad()
def get_db_codes_and_targets(database_loader, model, device='cpu'):
    model.eval()
    code_list, target_list = [], []
    for data, targets in tqdm(database_loader):
        data, targets = data.to(device), targets.to(device)
        target_list.append(targets)
        _, codes = model(data, model_training=False)
        code_list.append(codes)
    logging.info("Getting codes for dataset is done.")
    db_codes = torch.cat(code_list)
    db_targets = torch.cat(target_list)
    model.train()
    return db_codes, db_targets

class Evaluator:
    def __init__(self, feat_dim, M, K, codebooks=None, db_codes=None, db_targets=None, is_asym_dist=True, 
                 codebook_file=None, db_code_file=None, db_target_file=None, device='cpu'):
        self.feat_dim, self.M, self.K, self.D, self.device = feat_dim, M, K, feat_dim//M, device
        self.is_asym_dist = is_asym_dist
        self.set_codebooks(codebooks, codebook_file)
        self.set_db_codes(db_codes, db_code_file)
        self.set_db_targets(db_targets, db_target_file)
        self.lorentz_calculator = LorentzCalculation()

    def set_codebooks(self, codebooks=None, codebook_file=None):
        if codebook_file:  # Higher priority
            self.C = read_tensor(codebook_file, device=self.device) # [K, M*D]
            self.C = torch.split(self.C, self.D, dim=1) # [[K,D]]*M
        else:
            self.C = codebooks
            if self.C is not None:
                self.C = torch.split(self.C, self.D, dim=1) # [[K,D]]*M
        # Compute the lookup tables after updating the codebooks
        if (not self.is_asym_dist) and (self.C is not None):
            with torch.no_grad():
                # C:[MxKxD], intra_dist_tbls:[MxKxK]
                raise NotImplementedError("Not Supporting currently.")
                # self.intra_dist_tbls = torch.einsum('mkd,mjd->mkj', self.C, self.C)

    def set_db_codes(self, db_codes=None, db_code_file=None):
        # db_codes:[db_sizexM]
        if db_code_file:  # Higher priority
            self.db_codes = read_tensor(db_code_file, device=self.device)
        else:
            self.db_codes = db_codes

    def set_db_targets(self, db_targets=None, db_target_file=None):
        # db_targets:[db_size](single target version) OR [db_sizextgt_size](multi-target version)
        if db_target_file: # Higher priority
            self.db_targets = read_tensor(db_target_file, device=self.device)
        else:
            self.db_targets = db_targets

    def _symmetric_distance(self, query_codes):
        # query_codes:[bxM]
        dists = self.intra_dist_tbls[0][query_codes[:,0]][:, self.db_codes[:,0]]
        for i in range(1, self.M):
            # intra_dist_tbls[i]:[KxK].index(query_codes[:,i]:[b])=>[bxK]
            # intra_dist_tbls[i][query_codes[:,i]]:[bxK].column_index(db_codes[:,i]:[db_size])=>[bxdb_size]
            sub_dists = self.intra_dist_tbls[i][query_codes[:,i]][:, self.db_codes[:,i]]
            dists += sub_dists
        return dists
    
    def _batch_asymmetrib_dist_tbl(self, query_feats, neg_curvs):
        # query_feats: [b, M, D], in hyperbolic space
        # self.C: [K, M*D], in hyperbolic space 
        dist_tbl = []
        query_feats = torch.transpose(query_feats, 0, 1)
        for i in range(query_feats.shape[0]):
            ith_query_feats = query_feats[i] # [b, D]
            ith_C = self.C[i] # [k, D]
            # print("ith_C.shape", ith_C.shape)
            # print("ith_query_feats.shape", ith_query_feats.shape)
            ith_dist_tbl = self.lorentz_calculator.lorentz_dist(ith_query_feats, ith_C, neg_curvs[i]) #[b,k]
            dist_tbl.append(ith_dist_tbl)
        dist_tbl = torch.stack(dist_tbl, dim=0)
        return dist_tbl # [m,b,k]



    def _asymmetric_distance(self, query_feats, neg_curvs):
        qry_asym_dist_tbl = self._batch_asymmetrib_dist_tbl(query_feats, neg_curvs)
        # qry_asym_dist_tbl[i]:[bxK].column_index(db_codes[:,i]:[db_size])=>[bxdb_size]
        dists = qry_asym_dist_tbl[0][:, self.db_codes[:,0]]
        for i in range(1, self.M):
            sub_dists = qry_asym_dist_tbl[i][:, self.db_codes[:,i]]
            dists += sub_dists
        return dists

    @torch.no_grad()
    def distance(self, query_inputs, neg_curvs):
        if self.is_asym_dist:
            return self._asymmetric_distance(query_inputs, neg_curvs)
        else:
            return self._symmetric_distance(query_inputs)
    



    @torch.no_grad()
    def MAP(self, test_loader, model: HyperPQ, topK=None, test_batch_num=np.inf):
        model.eval()
        AP_list = []
        for i, (query_data, query_targets) in enumerate(tqdm(test_loader, desc="Test batch")):
            query_data, query_targets = query_data.to(self.device), query_targets.to(self.device)
            if self.is_asym_dist:
                # feats = model(query_data, only_feats=True, norm_feats=False)
                feats = model.encode_hyper_feats(query_data)
                dists = self.distance(feats, model.hyper_pq_head.neg_curvs)
            else:
                raise NotImplementedError("Not implementing symmeteric distance now")
                _, _, codes = model(query_data, hard_quant=True)
                dists = self.distance(codes)
            top_indices = torch.argsort(dists, descending=False) 
            if topK:
                top_indices = top_indices[:, :topK]
            else: # topK is None
                topK = top_indices.shape[-1]
            
            # db_targets:[db_size] OR [db_sizexlabel_size].index(top_indices:[bxtopK])=>[bxtopK] OR [bxtopKxlabel_size]
            top_targets = self.db_targets[top_indices]

            # query_targets:[bxlabel_size] or [b]
            # single target version
            if len(query_targets.shape) == 1 and len(self.db_targets.shape) == 1:
                # top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=-1) == top_targets).float()
            # multi-target version
            elif len(query_targets.shape) == 2 and len(self.db_targets.shape) == 2:
                # query_targets:[bxlabel_size].matmul(top_targets:[bxtopKxlabel_size])=>top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=1) * top_targets).sum(dim=-1).bool().float()
            else:
                raise RuntimeError("Invalid target shape: dimension of query target is %d, and dimension of database target is %d" % 
                                    (len(query_targets.shape), len(self.db_targets.shape)))

            # hit_counts:[b]
            hit_counts = top_hit_list.sum(dim=-1)
            hit_counts[hit_counts <= 10e-6] = 1.0 # avoid zero division
            # hit_cumsum_list:[bxtopK]
            hit_cumsum_list = top_hit_list.cumsum(dim=-1)
            # position_list:[topK]
            position_list = torch.arange(1, topK+1, dtype=torch.float, device=self.device)
            # precision_list:[bxtopK]
            precision_list = hit_cumsum_list / position_list
            # recall_list:[bxtopK]
            recall_list = hit_cumsum_list / hit_counts.unsqueeze(dim=-1)
            # AP:[b]
            AP = (precision_list * top_hit_list).sum(dim=-1) / hit_counts
            AP_list.append(AP)

            if i + 1 >= test_batch_num:
                break
    
        mAP = torch.cat(AP_list).mean().item()
        model.train()
        return mAP
    


    def _asymmetric_distance_ith_codebook(self, query_feats, neg_curvs, ith_codebook):
        qry_asym_dist_tbl = self._batch_asymmetrib_dist_tbl(query_feats, neg_curvs)
        dist = qry_asym_dist_tbl[ith_codebook][:, self.db_codes[:,ith_codebook]]
        return dist


    @torch.no_grad()
    def distance_ith_codebook(self, query_inputs, neg_curvs, ith_codebook):
        if self.is_asym_dist:
            return self._asymmetric_distance_ith_codebook(query_inputs, neg_curvs, ith_codebook)
        else:
            return self._symmetric_distance(query_inputs)


    @torch.no_grad()
    def MAP_of_each_codebok(self, test_loader, model: HyperPQ, topK=None, test_batch_num=np.inf):
        all_map = []
        for ith_codebook in range(self.M):
            map_ith = self.MAP_of_ith_codebok(test_loader, model, topK, test_batch_num, ith_codebook)
            all_map.append(map_ith)
        model.train()
        return all_map

    @torch.no_grad()
    def MAP_of_ith_codebok(self, test_loader, model: HyperPQ, topK=None, test_batch_num=np.inf,
                            ith_codebook=0):
        model.eval()
        AP_list = []
        for i, (query_data, query_targets) in enumerate(tqdm(test_loader, desc="Test batch")):
            query_data, query_targets = query_data.to(self.device), query_targets.to(self.device)
            if self.is_asym_dist:
                # feats = model(query_data, only_feats=True, norm_feats=False)
                feats = model.encode_hyper_feats(query_data)
                dists = self.distance_ith_codebook(feats, model.hyper_pq_head.neg_curvs, ith_codebook)
            else:
                raise NotImplementedError("Not implementing symmeteric distance now")

            top_indices = torch.argsort(dists, descending=False) # TODO, Important here.  Distance-> asecnd; if simialarity (dot)-> descend.
            if topK:
                top_indices = top_indices[:, :topK]
            else: # topK is None
                topK = top_indices.shape[-1]
            
            # db_targets:[db_size] OR [db_sizexlabel_size].index(top_indices:[bxtopK])=>[bxtopK] OR [bxtopKxlabel_size]
            top_targets = self.db_targets[top_indices]

            # query_targets:[bxlabel_size] or [b]
            # single target version
            if len(query_targets.shape) == 1 and len(self.db_targets.shape) == 1:
                # top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=-1) == top_targets).float()
            # multi-target version
            elif len(query_targets.shape) == 2 and len(self.db_targets.shape) == 2:
                # query_targets:[bxlabel_size].matmul(top_targets:[bxtopKxlabel_size])=>top_hit_list:[bxtopK]
                top_hit_list = (query_targets.unsqueeze(dim=1) * top_targets).sum(dim=-1).bool().float()
            else:
                raise RuntimeError("Invalid target shape: dimension of query target is %d, and dimension of database target is %d" % 
                                    (len(query_targets.shape), len(self.db_targets.shape)))

            # hit_counts:[b]
            hit_counts = top_hit_list.sum(dim=-1)
            hit_counts[hit_counts <= 10e-6] = 1.0 # avoid zero division
            # hit_cumsum_list:[bxtopK]
            hit_cumsum_list = top_hit_list.cumsum(dim=-1)
            # position_list:[topK]
            position_list = torch.arange(1, topK+1, dtype=torch.float, device=self.device)
            # precision_list:[bxtopK]
            precision_list = hit_cumsum_list / position_list
            # recall_list:[bxtopK]
            recall_list = hit_cumsum_list / hit_counts.unsqueeze(dim=-1)
            # AP:[b]
            AP = (precision_list * top_hit_list).sum(dim=-1) / hit_counts
            AP_list.append(AP)

            if i + 1 >= test_batch_num:
                break
    
        mAP = torch.cat(AP_list).mean().item()
        return mAP