from random import random
import torch
from torch import nn
import torch.nn.functional as F
import math
import utils.compute as utils
import numpy as np
from pointnet2_repo.ops_utils import furthest_point_sample as fps
import MinkowskiEngine as ME
import logging
from model.model_apen import APENBlock,APENDecoder

class Frames_Base(nn.Module):
    def __init__(self,with_bn,is_rotation_only):
        super(Frames_Base, self).__init__()
        
        self.is_detach_frame = True
        self.is_rotation_only = is_rotation_only
        
        
        
        self.with_bn = with_bn
        self.is_2d_frame = False

        if self.is_rotation_only:
            if self.is_2d_frame:
                ops = torch.tensor([[1,1,1],
                               # [1,-1,1],
                                ]).unsqueeze(1)
                
            else:
                ops = torch.tensor([[1,1,1],
                                    [1,-1,-1],
                                    [-1,-1,1],
                                    [-1,1,-1]]).unsqueeze(1)    
        else:
            if self.is_2d_frame:
                ops = torch.tensor([[1,1,1],
                                    [1,-1,1],
                                ]).unsqueeze(1)
            else:
                ops = torch.tensor([[1,1,1],
                                        [1,1,-1],
                                        [1,-1,1],
                                        [1,-1,-1],
                                        [-1,1,1],
                                        [-1,1,-1],
                                        [-1,-1,1],
                                        [-1,-1,-1]]).unsqueeze(1)
        self.register_buffer("ops", ops)
    
    def conc(self, input: ME.SparseTensor, input_glob: ME.SparseTensor):
        
        assert isinstance(input, ME.SparseTensor)
        assert isinstance(input_glob, ME.SparseTensor)

        broadcast_feat = input.F.new(len(input), input_glob.size()[1])
        #batch_indices, batch_rows = input.coordinate_manager.origin_map(input.coordinate_map_key)
        if self.is_rotation_only:
            jump = 1 if self.is_2d_frame else 4
        else:
            jump = 2 if self.is_2d_frame else 8
        for b in input_glob.coordinates[:,0].unique():
            ff = input_glob.coordinates_and_features_at(b)
            # for loop on groups
            for e in range(0,ff[0][:,0].shape[0],jump):
                # for loop on frames
                for ll,j in enumerate(ff[0][e:e+jump,-1]):
            
            #for e,i in [(x[0]//4,x[1]) for x in enumerate(ff[0][:,0]) if x[0] % 4 == 0]:#ff[0][:,0][list(range(0,ff[0].shape[0],4))]:
                
                #for j in range(4):
                    i = ff[0][:,0][e]
                    tiled = ff[1][e + ll].unsqueeze(0).tile(((input.coordinates[:,0] == b) & (input.coordinates[:,1] == i) & (input.coordinates[:,-1] == j)).sum(),1)
                    broadcast_feat[(input.coordinates[:,0] == b) & (input.coordinates[:,1] == i) & (input.coordinates[:,-1] == j)] = tiled

            #broadcast_feat[row_ind] = input_glob.features_at(b)

        broadcast_cat = torch.cat((input.F, broadcast_feat), dim=1)
        return ME.SparseTensor(
            broadcast_cat,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )
    
    def get_frame(self,pnts,weights=None,grp_equiv_featues=None):
        if weights is None:
                center = pnts.mean(1,True)
                pnts_centered = pnts - center
                R = torch.bmm(pnts_centered.transpose(1,2),pnts_centered)
                lambdas,V_ = torch.symeig(R.detach(),True)            
                F =  V_.to(R)#.unsqueeze(1).repeat(1,pnts.shape[1],1,1)
                if self.is_detach_frame:
                    return F.detach(), center.detach()
                else:
                    return F,center
        else:

            #weights = weights 
            if not grp_equiv_featues is None:
                pnts = torch.cat([pnts.unsqueeze(1).tile(1,grp_equiv_featues.shape[1],1,1),grp_equiv_featues],dim=2).transpose(1,2)
                weights = torch.cat([weights,weights.mean(1,True).tile(1,grp_equiv_featues.shape[2],1)],dim=1)
            else:
                pnts = pnts.unsqueeze(2)
            center = (pnts + (0.0 if self.training else 0.0)*torch.randn_like(pnts)) * (weights / (weights.sum(1,True) + 1e-6)).unsqueeze(-1)
            #center = pnts.mean(1,True)
            center = center.sum(1,True).transpose(1,2)
            pnts_centered = pnts.transpose(1,2) - center
            # res = self.frame_pred((weights / weights.sum(1,True)).transpose(1,2).unsqueeze(-1) * pnts_centered)
            # F = res['pred'].view(pnts.shape[0],weights.shape[-1],3,3)
            #pnts_centered = (weights.unsqueeze(0).unsqueeze(-1) * pnts_centered).transpose(1,2)
            #pnts_centered = pnts_centered.transpose(1,2)
            R = torch.einsum('bkij,bkil->bkjl',
                             pnts_centered[...,:(2 + int(not self.is_2d_frame))],
                            (weights).transpose(1,2).unsqueeze(-1) * pnts_centered[...,:(2 + int(not self.is_2d_frame))])
            if self.is_2d_frame:
                # a = R[:,:,0,0]
                # b = R[:,:,1,1]
                # c = R[:,:,0,1]
                # delta = torch.sqrt(4 * c**2 + (a - b)**2)
                # lambda1 = (a+b - delta)/2
                # lambda2 = (a+b + delta)/2
                # v1 = torch.cat([torch.where(c.abs() < 1e-5, torch.where(a>b, torch.ones_like(c),torch.zeros_like(c)) , ((lambda2 - b)/c)).unsqueeze(-1),torch.where(c.abs() < 1e-5,  torch.where(a>b, torch.zeros_like(c),torch.ones_like(c)),torch.ones_like(c)).unsqueeze(-1)],dim=-1)
                # v2 = torch.cat([torch.where(c.abs() < 1e-5, torch.where(a>b, torch.zeros_like(c),torch.ones_like(c)) , ((lambda1 - b)/c)).unsqueeze(-1),torch.where(c.abs() < 1e-5,  torch.where(a>b, torch.ones_like(c),torch.zeros_like(c)),torch.ones_like(c)).unsqueeze(-1)],dim=-1)
                
                # V_ = torch.cat([v1.unsqueeze(-1),v2.unsqueeze(-1)],dim=-1)
                lambdas,V_ = torch.linalg.eigh(((weights.sum(1) > 1).float().unsqueeze(-1).unsqueeze(-1).detach() * R + (weights.sum(1) <= 1).float().unsqueeze(-1).unsqueeze(-1).detach() * torch.randn_like(R)))
                V_ = torch.cat([torch.cat([V_,torch.zeros(1,2).unsqueeze(0).unsqueeze(0).tile(V_.shape[0],V_.shape[1],1,1).to(V_)],dim=-2),torch.tensor([0,0,1]).view(1,1,3,1).tile(V_.shape[0],V_.shape[1],1,1).to(V_)],dim=-1)
                #print (V_)
            else:
            
                lambdas,V_ = torch.linalg.eigh(((weights.sum(1) > 1).float().unsqueeze(-1).unsqueeze(-1).detach() * R + (weights.sum(1) <= 1).float().unsqueeze(-1).unsqueeze(-1).detach() * torch.randn_like(R)))
                #V_ = torch.cat([torch.cat([V_,torch.zeros(1,2).unsqueeze(0).unsqueeze(0).tile(V_.shape[0],V_.shape[1],1,1).to(V_)],dim=-2),torch.tensor([0,0,1]).view(1,1,3,1).tile(V_.shape[0],V_.shape[1],1,1).to(V_)],dim=-1)
            # # F, B, vmax = l1pca_sbfk(pnts_centered.T, 3, 50, False)
            F =  V_#.to(R)
            
            
            if self.is_detach_frame:
                return F.detach(), center.detach(),lambdas
            else:
                return F,center,lambdas
            
    # def get_frame(self,pnts,override=None):
    #     if override is None:
    #         is_local_frame = self.is_local_frame
    #     else:
    #         is_local_frame = override
    #     if is_local_frame:
    #         batch_size = pnts.size(0)
    #         num_points = pnts.size(1)
            
    #         def knn(x, k):
               
    #             x_squrae = (x**2).sum(-1,True)
    #             pairwise_distance = -(x_squrae -2*torch.bmm(x,x.transpose(1,2))  + x_squrae.transpose(1,2))
            
    #             idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    #             return idx
    #         idx = knn(pnts, k=self.k_size)
    #         device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #         idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    #         idx = idx + idx_base

    #         idx = idx.view(-1)
             
    #         pnts = pnts.view(batch_size*num_points, -1)[idx, :].view(batch_size, num_points, self.k_size, 3) 
                
    #         center = pnts.mean(2,False)
    #         pnts_centered = pnts - center.unsqueeze(2)
            
    #         R =  torch.einsum('bpki,bpkj->bpij',pnts_centered,pnts_centered)
    #         lambdas,V_ = torch.linalg.eigh(R.detach().cpu())            
    #         F =  V_.to(R)
    #         if self.is_detach_frame:
    #             return F.detach(), center.detach(),pnts_centered.detach()
    #         else:
    #             return F,center,pnts_centered

    #     else:
    #         center = pnts.mean(1,True)
    #         pnts_centered = pnts - center
    #         R = torch.bmm(pnts_centered.transpose(1,2),pnts_centered)
    #         lambdas,V_ = torch.linalg.eigh(R.detach().cpu())            
    #         F =  V_.to(R).unsqueeze(1).repeat(1,pnts.shape[1],1,1)
    #         if self.is_detach_frame:
    #             return F.detach(), center.detach()
    #         else:
    #             return F,center



class APENSegmentation(Frames_Base):
    def __init__(self,with_bn,
                     is_detach_frame,
                     is_rotation_only,
                     weight_threshold,
                     num_layers,
                     number_of_groups,
                     merge_thr,
                     num_point_sample,
                     vote_threshold,
                     pik_thrshold,
                     tmpr,
                     num_classes_pred,
                     local_n_size,
                     rand_thr,
                     sigmas=[0.2 ,0.002,0.005,0.008]):
        super(APENSegmentation, self).__init__(with_bn,is_rotation_only)
                
        self.displacemnt_network = torch.nn.ModuleList([
                                                       
                                                        APENBlock(with_bn=with_bn,
                                                                  is_rotation_only=is_rotation_only,
                                                                  weight_threshold=weight_threshold,
                                                                  number_of_groups=number_of_groups,
                                                                  a_in=0,
                                                                  b_in=0,
                                                                  a_out=8,
                                                                  b_out=8,
                                                                  local_n_size=local_n_size),
                                                        APENBlock(with_bn=with_bn,
                                                                  is_rotation_only=is_rotation_only,
                                                                  weight_threshold=weight_threshold,
                                                                  number_of_groups=number_of_groups,
                                                                  a_in=8,
                                                                  b_in=8,
                                                                  a_out=8,
                                                                  b_out=8,
                                                                  local_n_size=local_n_size),
                                                        APENBlock(with_bn=with_bn,
                                                                  is_rotation_only=is_rotation_only,
                                                                  weight_threshold=weight_threshold,
                                                                  number_of_groups=number_of_groups,
                                                                  a_in=8,
                                                                  b_in=8,
                                                                  a_out=8,
                                                                  b_out=8,
                                                                  local_n_size=local_n_size),
                                                        APENBlock(with_bn=with_bn,
                                                                  is_rotation_only=is_rotation_only,
                                                                  weight_threshold=weight_threshold,
                                                                  number_of_groups=number_of_groups,
                                                                  a_in=8,
                                                                  b_in=8,
                                                                  a_out=32,
                                                                  b_out=32,
                                                                  local_n_size=local_n_size)])
        
        self.pred_net = APENDecoder(with_bn=with_bn,
                                    is_detach_frame=is_detach_frame,
                                    is_rotation_only=is_rotation_only,
                                    weight_threshold=weight_threshold,
                                    number_of_groups=number_of_groups,
                                    in_features_size=128,
                                    num_classes_pred=num_classes_pred,
                                    local_n_size=local_n_size)
        self.clamp_min = 1e-5

        self.number_of_groups = number_of_groups
        self.merge_thr = merge_thr
        self.num_point_sample = num_point_sample
        self.num_layers = num_layers
        self.vote_threshold = vote_threshold
        self.pik_thrshold = pik_thrshold
        self.temperature = tmpr
        self.with_bn = with_bn
        self.rand_thr = rand_thr
        self.sigmas = sigmas
        self.rand_cand_size = 10
        self.R = 16

        logging.debug ("""
        self.number_of_groups = {0}
        self.merge_thr = {1}
        self.num_point_sample {2}
        self.num_layers {3}
        self.vote_threshold {4}
        self.pik_thrshold = {5}
        self.temperature = {6}
        self.with_bn = {7}
        self.rand_cand_size = {8}
        self.sigmas = {9}""".format(self.number_of_groups,
                                    self.merge_thr,
                                    self.num_point_sample,
                                    self.num_layers,
                                    self.vote_threshold,
                                    self.pik_thrshold,
                                    self.temperature,
                                    self.with_bn,
                                    self.rand_cand_size,
                                    self.sigmas))        
    

    def forward(self, point_cloud,normals,epoch,gt=None,gt_seeds=None,vote_mask=None):
        output = {}
        B, N,D = point_cloud.size()
        
        
        sample_size = self.num_point_sample
        point_cloud = point_cloud[...,:3]

        sampled_indices = torch.arange(point_cloud.shape[1]).unsqueeze(0).tile(point_cloud.shape[0],1).to(point_cloud).long()
        
        output["sample_indices"] = sampled_indices
        output["sampled_points"] = point_cloud
        
        idx = fps(point_cloud,(self.number_of_groups - self.rand_cand_size)).long()
        idx = idx.to(point_cloud).long()
        
        candidates = torch.gather(point_cloud,dim=1,index=idx.unsqueeze(-1).tile(1,3))
        
        shuf = np.arange(point_cloud.shape[1])
        all_cand = []
        
        for i in range(point_cloud.shape[0]):
            count = 0
            new_cand = candidates[i]
            while count < self.rand_cand_size:
                np.random.shuffle(shuf)
                idx = torch.tensor(shuf[:max(self.rand_cand_size - count,0)]).to(point_cloud).long()
                c_add = torch.gather(point_cloud[i],dim=0,index=idx.unsqueeze(-1).tile(1,3))
                d = ((c_add.unsqueeze(0) - new_cand.unsqueeze(1))**2).sum(-1).min(0)
                keep = d[0] > self.rand_thr**2
                count += keep.sum().item()
                new_cand = torch.cat([new_cand,c_add[keep]],dim=0)
            all_cand.append(new_cand.unsqueeze(0))
        candidates = torch.cat(all_cand,dim=0)
        output["0_predicted_candidates"] = candidates

        if gt is None:
            current_seeds = point_cloud.unsqueeze(1).sum(1,True)
        else:
            current_seeds = (point_cloud + sampled_gt).unsqueeze(1)
        output["{0}_predicted_seeds".format(0)] = current_seeds
        candidate_mask = torch.ones(candidates.shape[0:2]).to(candidates)
        output["init_candidates"] = candidates 
        
        current_features_equiv = None
        current_features_inv = None
        cand_features_equiv = None
        cand_features_inv = None
        
        insider_iter = 0
        merge_thr = self.merge_thr
        
        sigmas = self.sigmas #[0.2 ,0.001,0.003,0.005]
        fix_sigma = sigmas[0]
        pik_new = (1./self.number_of_groups) * torch.ones(self.number_of_groups).unsqueeze(0).tile(point_cloud.shape[0],1).to(point_cloud)
        output["{0}_predicted_pik".format(0)] = pik_new
        var_k = ((fix_sigma)*torch.ones(3)).unsqueeze(0).unsqueeze(0).tile(point_cloud.shape[0],self.number_of_groups,1).to(point_cloud)
        for iter in range(self.num_layers):
            pik_thrshold = self.pik_thrshold * 2**(iter//self.R)
            output["{0}_candidate_mask".format(iter)] = candidate_mask.clone()
            
            # E - step
            delta = (current_seeds -  candidates.unsqueeze(2))
            N_nk =  (-3./2. * np.log(2 * np.pi) - 0.5*torch.log(var_k).sum(-1).unsqueeze(-1) - 0.5 * (delta * (delta * torch.reciprocal(var_k.unsqueeze(2)))).sum(-1)) +  torch.where(candidate_mask == 0, -torch.inf*torch.ones_like(pik_new),torch.log(pik_new)).unsqueeze(-1)
            nonnormalized_weights = N_nk.transpose(1,2)
            output["{0}_nonormalized_weights".format(iter)] = nonnormalized_weights
            N_nk = torch.exp(N_nk - torch.logsumexp(N_nk,dim=1,keepdim=True))
            weights = N_nk.transpose(1,2)
            output["{0}_weights".format(iter)] = weights

            if iter > 0:
            
                # M - step            
                candidates = (weights.transpose(1,2).unsqueeze(-1) * current_seeds).sum(2)/(weights.sum(1).unsqueeze(-1) + 1e-5)
                merge = True

                if iter in [i * self.R + (self.R//2) for i in range(self.num_layers//self.R)]:
                    
                    while merge and iter in [i * self.R + (self.R//2) for i in range(self.num_layers//self.R)]:
                            
                        
                        d = ((candidates.unsqueeze(1) - candidates.unsqueeze(2))**2)                        
                        d = (d * (d * torch.reciprocal(var_k.unsqueeze(2)))).sum(-1)
                        d = (-torch.log(torch.bmm(candidate_mask.unsqueeze(-1),candidate_mask.unsqueeze(1))) + 1) * (d+1e-7)
                        f1 = d.topk(dim=2,k=2,largest=False)
                        f2 = f1.values[...,-1].min(-1)
                        nnn = utils.vector_gather(f1[1],f2[1])
                        ss = utils.vector_gather(pik_new.unsqueeze(-1),nnn).squeeze(-1).sum(-1,True)
                        pik_new = (f2.values >= merge_thr**2).float().unsqueeze(-1) * pik_new +  (f2.values < merge_thr**2).float().unsqueeze(-1) * pik_new.scatter(dim=1,index=nnn,src=torch.cat([ss,torch.zeros_like(nnn[:,0:1]).to(pik_new)],dim=-1))
                        candidate_mask.scatter_(dim=1,index=nnn[:,1:2],src=(f2.values.unsqueeze(-1) >= merge_thr**2).to(candidate_mask))
                        merge=(f2.values < merge_thr**2).any().item()
                    #logging.debug ((iter,candidate_mask.sum(-1)))
                    var_k = fix_sigma*torch.ones_like(var_k) 
                    
                else:
                    
                    var_k = fix_sigma*torch.ones_like(var_k) 
                    pik_new = (weights.sum(1)/point_cloud.shape[1]) 

                    if iter in [4 + i * self.R + (self.R//2) for i in range(self.num_layers//self.R)]:
                        candidate_mask = candidate_mask * (pik_new > pik_thrshold).float()
                        pik_new = pik_new * (-(pik_new <= pik_thrshold).float() + 1)
                
                output["{0}_predicted_var".format(iter)] = var_k
                output["{0}_predicted_candidates".format(iter)] = candidates
                output["{0}_predicted_pik".format(iter)] = pik_new
                
            
            candidates = candidates.detach()
            var_k = var_k.detach()
            pik_new = pik_new.detach()

            if iter % self.R == 0:
                if iter > 0:
                
                    delta = (current_seeds -  candidates.unsqueeze(2))
                    N_ik_nominator =  (-3./2.) * np.log(2 * np.pi) - 0.5*torch.log(var_k).sum(-1).unsqueeze(-1) - 0.5 * (delta * (delta * torch.reciprocal(var_k.unsqueeze(2)))).sum(-1)
                    N_ik_denominator = torch.logsumexp(N_ik_nominator + torch.where(candidate_mask == 0, -torch.inf*torch.ones_like(pik_new),torch.log(pik_new)).unsqueeze(-1),dim=1,keepdim=True)
                    delta_pik = torch.exp(N_ik_nominator - N_ik_denominator)
                    cov_pik = -(delta_pik.unsqueeze(2) * delta_pik.unsqueeze(1)).mean(-1)
                    delta_pik_mean = candidate_mask * (delta_pik.mean(-1) - 1)

                    N_nk =  (-3./2. * np.log(2 * np.pi) - 0.5*torch.log(var_k).sum(-1).unsqueeze(-1) - 0.5 * (delta * (delta * torch.reciprocal(var_k.unsqueeze(2)))).sum(-1)) +   torch.where(candidate_mask == 0, -torch.inf*torch.ones_like(pik_new),torch.log(pik_new)).unsqueeze(-1)
                    delta_candidates = (torch.exp(N_nk - torch.logsumexp(N_nk,dim=1,keepdim=True))).unsqueeze(-1) * (torch.reciprocal(var_k).unsqueeze(2) * delta)
                    delta_candidates_mean = candidate_mask.unsqueeze(-1).tile(1,1,3) * delta_candidates.mean(2)
                    
                    b = -(delta_candidates.transpose(1,2).flatten(-2).unsqueeze(-1) * delta_candidates.transpose(1,2).flatten(-2).unsqueeze(-2)).mean(1)

                    new = delta_candidates.unsqueeze(3) * (torch.reciprocal(var_k).unsqueeze(2) * delta).unsqueeze(-1) - (torch.exp(N_nk - torch.logsumexp(N_nk,dim=1,keepdim=True))).unsqueeze(-1).unsqueeze(-1) * torch.diag_embed(torch.reciprocal(var_k)).unsqueeze(2)
                    delta_cand_cand = torch.diag_embed(new.permute([0,2,3,4,1])).permute([0,1,5,2,4,3]).flatten(-2).view(new.shape[0],new.shape[2],-1,new.shape[1]*new.shape[3]).mean(1) + b

                    tt = (delta_candidates_mean/ pik_new.unsqueeze(-1))
                    delta_pik_candidate = torch.diag_embed(tt.transpose(1,2)).permute([0,3,2,1]).flatten(-2) - (delta_candidates.transpose(1,2).flatten(-2).unsqueeze(2) * delta_pik.transpose(1,2).unsqueeze(-1)).mean(1)
                    
                    mask = (torch.cat([candidate_mask, candidate_mask.unsqueeze(-1).tile(1,1,3).view(candidate_mask.shape[0],-1)],dim=-1).unsqueeze(-1) * torch.cat([candidate_mask, candidate_mask.unsqueeze(-1).tile(1,1,3).view(candidate_mask.shape[0],-1)],dim=-1).unsqueeze(-2)).bool()
                    fisher = torch.cat([torch.cat([cov_pik,delta_pik_candidate],dim=-1),torch.cat([delta_pik_candidate.transpose(1,2),delta_cand_cand],dim=-1)],dim=1)
                    fisher_inv = []
                    nonnormalized_weights_ = []
                    weights_ = []
                    candidates_ = []
                    
                    for i in range(fisher.shape[0]):
                        sel = torch.masked_select(fisher[i], mask[i])
                        sel = sel.view(np.sqrt(sel.shape[0]).astype(int),np.sqrt(sel.shape[0]).astype(int))
                        fisher_inv = -torch.linalg.pinv(sel.detach(),hermitian=True)

                        theta_phi = torch.cat([torch.masked_select(pik_new[i],candidate_mask[i].bool()),torch.masked_select(candidates[i],candidate_mask[i].unsqueeze(-1).tile(1,3).bool())],dim=-1).detach() + torch.mm(fisher_inv,
                                                                                                                                                                                                             (torch.cat([torch.masked_select(delta_pik_mean[i],candidate_mask[i].bool()),torch.masked_select(delta_candidates_mean[i],candidate_mask[i].unsqueeze(-1).tile(1,3).bool())],dim=-1) - torch.cat([torch.masked_select(delta_pik_mean[i],candidate_mask[i].bool()),torch.masked_select(delta_candidates_mean[i],candidate_mask[i].unsqueeze(-1).tile(1,3).bool())],dim=-1).detach()).unsqueeze(-1)).squeeze(-1)
                        cand = theta_phi[candidate_mask[i].sum().long():].view(-1,candidates.shape[2])
                        candidates_.append(torch.zeros_like(candidates[i]).masked_scatter_(candidate_mask[i].bool().unsqueeze(-1).tile(1,candidates.shape[-1]),cand).unsqueeze(0))
                        pik_new_ = theta_phi[:candidate_mask[i].sum().long()]

                        delta = (current_seeds[i].detach() -  cand.unsqueeze(1))
                        N_nk =  (-3./2. * np.log(2 * np.pi) - 0.5*torch.log(torch.masked_select(var_k[i],candidate_mask[i].unsqueeze(-1).tile(1,3).bool()).view(-1,3)).sum(-1).unsqueeze(-1) - 0.5 * (delta * (delta * torch.reciprocal(torch.masked_select(var_k[i],candidate_mask[i].unsqueeze(-1).tile(1,3).bool()).view(-1,3).unsqueeze(1)))).sum(-1)) +  torch.log(pik_new_).unsqueeze(-1)
                        nonnormalized_weights_.append((-torch.inf*torch.ones_like(nonnormalized_weights[i])).masked_scatter_(candidate_mask[i].bool().unsqueeze(0).tile(weights.shape[1],1) ,N_nk.transpose(0,1)).unsqueeze(0))
                        N_nk = torch.exp(N_nk - torch.logsumexp(N_nk,dim=0,keepdim=True))
                        weights_temp = N_nk.transpose(0,1)
                        weights_.append((torch.zeros_like(weights[i])).masked_scatter_(candidate_mask[i].bool().unsqueeze(0).tile(weights.shape[1],1) ,weights_temp).unsqueeze(0))
                    
                    nonnormalized_weights = torch.cat(nonnormalized_weights_,dim=0)
                    weights = torch.cat(weights_,dim=0)
                    candidates = torch.cat(candidates_,dim=0)
                    if not current_features_equiv is None:
                        cand_features_equiv = (weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1) * current_features_equiv.unsqueeze(1)).sum(2)/(weights.sum(1).unsqueeze(-1).unsqueeze(-1) + 1e-5)
                        cand_features_inv = (weights.transpose(1,2).unsqueeze(-1) * current_features_inv.unsqueeze(1)).sum(2)/(weights.sum(1).unsqueeze(-1) + 1e-5)
                        
                    output["{0}_nonormalized_weights".format(iter)] = nonnormalized_weights
                    output["{0}_weights".format(iter)] = weights
                    output["{0}_final_centers".format(iter//self.R)] = candidates
                    output["{0}_objectness_label".format(iter//self.R)] = torch.ones_like(candidates[...,0])
                    output["{0}_final_candidate_mask".format(iter//self.R)] = candidate_mask

                else:
                    output["init_weights".format(iter)] = weights

                if iter < self.num_layers - 1:
                    if not gt is None:
                        output["{0}_predicted_seeds".format(iter//self.R)] = (point_cloud + sampled_gt[...,:3]).unsqueeze(1)
                        output["{0}_predicted_seeds_other".format(iter//self.R)] = (point_cloud).unsqueeze(1)
                        current_seeds = output["{0}_predicted_seeds".format(iter//self.R)].sum(1,True)
                    else:
                        
                        tmpr = self.temperature
                        dr_w = torch.nn.functional.gumbel_softmax(nonnormalized_weights,tau=tmpr,hard=True)
                        output["{0}_weights".format(iter)] = dr_w
                        self.displacemnt_network[iter//self.R](point_cloud=point_cloud,
                                                    normals=normals, 
                                                    dr_w = dr_w,
                                                    weights=weights,# if self.training else weights,#weights,
                                                    iter=iter//self.R,
                                                    output=output,
                                                    cand_features_equiv=cand_features_equiv,
                                                    cand_features_inv=cand_features_inv,                                                    
                                                    epoch=epoch)
                
                        current_seeds = output["{0}_predicted_seeds".format(iter//self.R)].sum(1,True)
                        current_features_equiv = output["{0}_seeds_features_equiv".format(iter//self.R)]
                        current_features_inv = output["{0}_seeds_features_inv".format(iter//self.R)]
                        
                            
                        
                        
                    insider_iter = 0
                    filter_seeds = current_seeds
                    idx_c = fps(filter_seeds.squeeze(1),candidates.shape[1]).long()
                    candidates = utils.vector_gather(filter_seeds.squeeze(1),idx_c)
                    candidate_mask = torch.ones(candidates.shape[0:2]).to(candidates)
                    pik_new = (1./self.number_of_groups) * torch.ones(self.number_of_groups).unsqueeze(0).tile(point_cloud.shape[0],1).to(point_cloud)
                    merge = True
                    fix_sigma = sigmas[max((iter-1)//self.R,0) + 1]
                    var_k = fix_sigma*torch.ones_like(var_k)     
                    while merge:
                        
                        d = ((candidates.unsqueeze(1) - candidates.unsqueeze(2))**2)#.sum(-1)
                        d = (d * (d * torch.reciprocal(var_k.unsqueeze(2)))).sum(-1)
                        d = (-torch.log(torch.bmm(candidate_mask.unsqueeze(-1),candidate_mask.unsqueeze(1))) + 1) * (d+1e-7)
                        f1 = d.topk(dim=2,k=2,largest=False)
                        f2 = f1.values[...,-1].min(-1)
                        nnn = utils.vector_gather(f1[1],f2[1])
                        ss = utils.vector_gather(pik_new.unsqueeze(-1),nnn).squeeze(-1).sum(-1,True)
                        pik_new = (f2.values > merge_thr**2).float().unsqueeze(-1) * pik_new +  (f2.values < merge_thr**2).float().unsqueeze(-1) * pik_new.scatter(dim=1,index=nnn,src=torch.cat([ss,torch.zeros_like(nnn[:,0:1]).to(pik_new)],dim=-1))
                        candidate_mask.scatter_(dim=1,index=nnn[:,1:2],src=(f2.values.unsqueeze(-1) > merge_thr**2).to(candidate_mask))
                        merge=(f2.values < merge_thr**2).any().item()
                    #logging.debug ((iter,candidate_mask.sum(-1)))
                    output["{0}_predicted_candidates".format(iter)] = candidates
                    var_k = fix_sigma*torch.ones_like(var_k) 
                    output["{0}_predicted_var".format(iter)] = var_k
                    output["{0}_predicted_pik".format(iter)] = pik_new
                    
            insider_iter = insider_iter + 1
        tmpr = self.temperature# - np.linspace(0,self.temperature - 1,3000)[min(epoch,2999)]
        dr_w = torch.nn.functional.gumbel_softmax(nonnormalized_weights,tau=tmpr,hard=True)
        self.pred_net(point_cloud=point_cloud,
                                                normals=normals, 
                                                dr_w = dr_w,
                                                weights=weights,# if self.training else weights,#weights,
                                                iter=iter//self.R,
                                                output=output,
                                                cand_features_equiv=cand_features_equiv,
                                                cand_features_inv=cand_features_inv,
                                                epoch=epoch)

            #Score computation
            #delta_pik = (weights / (pik_new.unsqueeze(1) * weights).sum(-1,True)).mean(1)
            
            
            #print (iter,delta_pik)


        output["final_center"] = candidates
    
        return output


