import torch
import torch.nn as nn 
import torch.nn.functional as F
import math_util

class LorentzCalculation:
    def __init__(self):
        self.name = "Hyperboloid Network"
    
    def proj_tan0(self, u):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:,0:1] = narrowed 
        return u - vals 

    def expmap0(self, v, clip_r, c, alpha_scaler=None):
        if alpha_scaler is not None:
            v = v * torch.exp(alpha_scaler)
        k = 1./c
        # v is in the tangent space 
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_clipped = torch.minimum(torch.ones_like(v_norm), clip_r/v_norm)*v # clipped embeddings v_norm 
        x = math_util.expmap0(v_clipped, k=k)
        return x 

    def lorentz_dot(self, x, y):
        # BD, KD -> BK
        # minkowski product, one2all
        result_space = torch.matmul(x[:,1:], y[:,1:].T) # BD,KD->BK
        result_time = torch.matmul(x[:,0].view(-1,1), y[:,0].view(1,-1)) # B1,1K->BK
        result = result_space - result_time 
        return result 
    
    def lorentz_dot_o2o(self, x, y):
        # BD, BD -> B1
        m = x * y
        result = m[:, 1:].sum(dim = 1) - m[:,0]
        return result.reshape(-1, 1)
    
    def lorentz_dist(self, x, y, c):
        # BD,KD -> BK
        k = 1./c
        prod = self.lorentz_dot(x,y)
        assert not torch.any(-prod/k < 0.99999)
        # dist = torch.sqrt(k) * math_util.arcosh(-prod/k)
        return torch.sqrt(k) * math_util.arcosh(-prod /k)
    
    def sqrt_lorentz_dist(self, x, y, c):
        # BD, KD -> BK
        k = 1./c
        prod = self.lorentz_dot(x,y)
        return -2 * k - 2 * prod 
    
    def mid_point(self, prob, x, c):
        # BK, KD -> BD
        k = 1./c
        avg = torch.matmul(prob, x) #BD
        denom = -self.lorentz_dot_o2o(avg, avg) # B1
        denom = denom.abs().clamp_min(1e-8).sqrt()
        centroid = torch.sqrt(k) * avg / denom
        return centroid

    def mid_point_for_softsort(self, P_hat, x, c):
        # BKK, KD -> BKD
        k = 1./c 
        avg = torch.einsum("bhk,kd->bhd", P_hat, x) # BKD
        dot_avg = avg * avg # BKD
        denom = (dot_avg[:,:,1:].sum(dim=-1) - dot_avg[:,:,0]) #BK
        denom = denom.abs().clamp_min_(1e-8).sqrt()
        centroid = torch.sqrt(k) * avg / (denom.unsqueeze(-1))
        return centroid #BKD
    
    def lorentz_dist_for_softsort(self, z_i, z_j_cwd, c):
        # z_i: [B,D]
        # z_j_cwd: [B,K,D]
        # return [B,K] 
        def lorentz_dot_for_softsort(x, y):
            # BD, BKD -> BK
            x_space = x[:,1:].unsqueeze(dim=-1) #[B,1, D-1]
            y_space = y[:,:,1:] # [B,K, D-1]
            result_space = torch.bmm(y_space, x_space).squeeze() # [B,K]
            x_time = x[:,0].unsqueeze(-1) # [B,1]
            y_time = y[:,:,0] # [B,K]
            result_time = x_time * y_time # broadcast
            result = result_space - result_time 
            return result 
        
        k = 1./c
        prod = lorentz_dot_for_softsort(z_i, z_j_cwd)
        assert not torch.any(-prod/k < 0.99999)
        # dist = torch.sqrt(k) * math_util.arcosh(-prod/k)
        return torch.sqrt(k) * math_util.arcosh(-prod /k)




    

        

    
    