from random import random
import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import math
import utils.compute as utils
import numpy as np
from pointnet2_repo.ops_utils import furthest_point_sample as fps
import MinkowskiEngine as ME
import logging

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
                
                lambdas,V_ = torch.linalg.eigh(((weights.sum(1) > 1).float().unsqueeze(-1).unsqueeze(-1).detach() * R + (weights.sum(1) <= 1).float().unsqueeze(-1).unsqueeze(-1).detach() * torch.randn_like(R)))
                V_ = torch.cat([torch.cat([V_,torch.zeros(1,2).unsqueeze(0).unsqueeze(0).tile(V_.shape[0],V_.shape[1],1,1).to(V_)],dim=-2),torch.tensor([0,0,1]).view(1,1,3,1).tile(V_.shape[0],V_.shape[1],1,1).to(V_)],dim=-1)
                
            else:
            
                lambdas,V_ = torch.linalg.eigh(((weights.sum(1) > 1).float().unsqueeze(-1).unsqueeze(-1).detach() * R + (weights.sum(1) <= 1).float().unsqueeze(-1).unsqueeze(-1).detach() * torch.randn_like(R)))
            
            F =  V_
            
            if self.is_detach_frame:
                return F.detach(), center.detach(),lambdas
            else:
                return F,center,lambdas
            

class APENBlock(Frames_Base):
    def __init__(self,with_bn,
                      is_rotation_only,
                      weight_threshold,
                      number_of_groups,
                      a_in,
                      b_in,
                      a_out,
                      b_out,
                      local_n_size):
        super(APENBlock, self).__init__(with_bn,is_rotation_only)
        
        self.a_in = a_in
        self.b_in = b_in
        self.a_out = a_out
        self.b_out = b_out
        self.fetaure_size = lambda a,b: a + 3 * b
        self.in_features_size = self.fetaure_size(self.a_in,self.b_in)
        self.out_features_size = self.fetaure_size(self.a_out,self.b_out) + 3
        self.local_n_size = local_n_size
        
        if self.in_features_size == 0:
            self.a_in = 4
            self.b_in = 4
        self.conv1 = ME.MinkowskiLinear(self.in_features_size + self.local_n_size*3 + 3 ,self.fetaure_size(self.a_in*2,self.b_in*2) )# torch.nn.Conv1d(3, 64, 1)
        self.conv2 = ME.MinkowskiLinear(self.fetaure_size(self.a_in*2,self.b_in*2),self.fetaure_size(self.a_in*3,self.b_in*3))#torch.nn.Conv1d(64, 128, 1)
        self.conv3 = ME.MinkowskiLinear(self.fetaure_size(self.a_in*3,self.b_in*3),self.fetaure_size(self.a_in*4,self.b_in*4))#torch.nn.Conv1d(128, 128, 1)
        self.conv4 = ME.MinkowskiLinear(self.fetaure_size(self.a_in*4,self.b_in*4),self.fetaure_size(self.a_in*5,self.b_in*5))#torch.nn.Conv1d(128, 512, 1)
        self.conv5 = ME.MinkowskiLinear(self.fetaure_size(self.a_in*5,self.b_in*5),self.fetaure_size(self.a_in*6,self.b_in*6))#torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = ME.MinkowskiBatchNorm(self.fetaure_size(self.a_in*2,self.b_in*2))
        self.bn2 = ME.MinkowskiBatchNorm(self.fetaure_size(self.a_in*3,self.b_in*3))
        self.bn3 = ME.MinkowskiBatchNorm(self.fetaure_size(self.a_in*4,self.b_in*4))
        self.bn4 = ME.MinkowskiBatchNorm(self.fetaure_size(self.a_in*5,self.b_in*5))
        self.bn5 = ME.MinkowskiBatchNorm(self.fetaure_size(self.a_in*6,self.b_in*6))
        self.convs1 = ME.MinkowskiLinear(np.array([self.fetaure_size(self.a_in*i,self.b_in*i) for i in list(range(2,7)) + [6]]).sum(),256)# torch.nn.Conv1d(4944 - 16, 256, 1)
        
            
        self.convs2 = ME.MinkowskiLinear(256, 256)#torch.nn.Conv1d(256, 256, 1)
        self.convs3 = ME.MinkowskiLinear(256, 128) #torch.nn.Conv1d(256, 128, 1)
        self.convs4 = ME.MinkowskiLinear(128, self.out_features_size) #torch.nn.Conv1d(128, num_part, 1)
        
        self.bns1 = ME.MinkowskiBatchNorm(256)
        self.bns2 = ME.MinkowskiBatchNorm(256)
        self.bns3 = ME.MinkowskiBatchNorm(128)
        
        self.max_pool = ME.MinkowskiGlobalSumPooling()
        self.concat_op = ME.MinkowskiBroadcastConcatenation()
        self.activation = ME.MinkowskiReLU()
        self.activation_dense = nn.ReLU()
        self.weight_threshold = weight_threshold
        

    def forward(self,point_cloud,normals,dr_w,weights,iter,output,cand_features_equiv,cand_features_inv,epoch):
        B, N, D = point_cloud.size()

        #weights = torch.nn.functional.gumbel_softmax(weights,tau=0.01,hard=True)
        #normals = 0*normals
        weights = dr_w
        frames,centers,lambdas = self.get_frame(point_cloud,weights)
        output["debug_{0}_lambdas".format(iter)] = lambdas
        output["debug_{0}_frames".format(iter)] = frames
        
        if self.is_rotation_only:
            F_ops = self.ops.unsqueeze(0).unsqueeze(2) * (torch.sign(torch.det(frames[...,:])).unsqueeze(-1).unsqueeze(-1) * frames).unsqueeze(1)
        else:
            F_ops = self.ops.unsqueeze(0).unsqueeze(2) * frames[...,:].unsqueeze(1)
        
        d = (weights.unsqueeze(2) * weights.unsqueeze(1)).sum(-1)*((point_cloud.unsqueeze(2) - point_cloud.unsqueeze(1))**2).sum(-1) + (1-(weights.unsqueeze(2) * weights.unsqueeze(1)).sum(-1))*1000
        idx = d.topk(dim=-1,largest=False,k=self.local_n_size)[1]

        pnts_n = torch.gather(input=point_cloud.unsqueeze(2).tile(1,1,idx.shape[-1],1),index=idx.unsqueeze(-1).tile(1,1,1,3),dim=1)
        
        v1 = ((1-weights).transpose(1,2).unsqueeze(-1).unsqueeze(-1) * centers.unsqueeze(-2).tile(1,1,1,self.local_n_size,1) + (weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1) * pnts_n.unsqueeze(1)) - centers.unsqueeze(-2))
        output["debug_{0}_input_before".format(iter)] = v1
        framed_input1 = torch.matmul(F_ops.transpose(3,4).unsqueeze(3).tile(1,1,1,v1.shape[2],1,1),v1.unsqueeze(1).tile(1,F_ops.shape[1],1,1,1,1).transpose(-1,-2)).transpose(-1,-2)
        output["debug_{0}_input_aftrer".format(iter)] = framed_input1
        
        normals_ = normals.unsqueeze(2)
        
        v2 = normals_.unsqueeze(1) * weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1) + (1-weights).transpose(1,2).unsqueeze(-1).unsqueeze(-1) * ((normals_.unsqueeze(1) * weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1)).sum(2,True)/(weights.sum(1,True).transpose(1,2).unsqueeze(-1).unsqueeze(-1) + 1e-6))
        if not cand_features_equiv is None:
            v2 = torch.cat([v2,cand_features_equiv.unsqueeze(2).tile(1,1,v2.shape[2],1,1)],dim=3)
        
        output["debug_{0}_v2_before".format(iter)] = v2
        framed_input2 = torch.matmul(F_ops.transpose(3,4).unsqueeze(3).tile(1,1,1,v2.shape[2],1,1),v2.unsqueeze(1).tile(1,F_ops.shape[1],1,1,1,1).transpose(-1,-2)).transpose(-1,-2)#torch.einsum('bokij,bkpl->bokpi',F_ops.transpose(3,4),v2)#.unsqueeze(-1)
        output["debug_{0}_v2_after".format(iter)] = framed_input2

        
        
        framed_input = torch.cat([framed_input1.flatten(-2),
                                  framed_input2.flatten(-2),
                                  ],dim=-1)
        if not cand_features_inv is None:
            framed_input = torch.cat([framed_input,
                                      cand_features_inv.unsqueeze(1).unsqueeze(-2).tile(1,F_ops.shape[1],1,point_cloud.shape[1],1)],
                                      dim=-1)
        
        weights_t = weights.unsqueeze(-1).transpose(1,2).tile(1,1,1,framed_input.shape[1])
        
        # Find all points with weight above threshold. 
        ind = (weights_t.view(-1) >= self.weight_threshold).nonzero()
        # filter input points according to ind
        filtered_input = framed_input.permute([0,2,3,1,4]).reshape(-1, self.in_features_size + self.local_n_size*3 + 3)[ind].squeeze(1)
        # Flatten batch, weights dimensions and other dimensions are points and frames. ind_map are the filtered indices.
        # i.e., ind_map[i](a,b,c) a is the index of the filtered batch,weight b is the index of the point and c is the index of the frame
        ind_map = (weights_t  >= self.weight_threshold).nonzero().int().contiguous()
        sparse_input = ME.SparseTensor(coordinates=ind_map,features=filtered_input)
        weights_sparse = ME.SparseTensor(coordinates=sparse_input.coordinates,
                                         features = weights_t[(weights_t.detach() >= self.weight_threshold)].unsqueeze(-1),
                                         coordinate_manager=sparse_input.coordinate_manager)
        out1 = self.activation(self.bn1(self.conv1(sparse_input)) if self.with_bn else self.conv1(sparse_input))
        out2 = self.activation(self.bn2(self.conv2(out1)) if self.with_bn else self.conv2(out1))
        out3 = self.activation(self.bn3(self.conv3(out2)) if self.with_bn else self.conv3(out2))
        out4 = self.activation(self.bn4(self.conv4(out3)) if self.with_bn else self.conv4(out3))
        out5 = self.bn5(self.conv5(out4)) if self.with_bn else self.conv5(out4)
        
        kernel_size = [1,ind_map[:,2].max().item()+1,1]
        sum_pool_nom = ME.MinkowskiSumPooling(kernel_size = kernel_size,stride=kernel_size,dimension=3)(out5 * weights_sparse)
        sum_pool_denom = ME.MinkowskiSumPooling(kernel_size = kernel_size,stride=kernel_size,dimension=3)(weights_sparse)
        sum_pool_denom = sum_pool_denom + ME.SparseTensor(coordinates=sum_pool_denom.coordinates,features=1e-5*torch.ones_like(sum_pool_denom.features),coordinate_manager=sum_pool_denom.coordinate_manager)
        out_pool = sum_pool_nom/sum_pool_denom
        
        
        cnt = ME.cat([out1,out2,out3,out4,out5])
        
        concat = self.conc(cnt,out_pool)
        
        outs1 = self.activation(self.bns1(self.convs1(concat)) if self.with_bn else self.convs1(concat))
        outs2 = self.activation(self.bns2(self.convs2(outs1)) if self.with_bn else self.convs2(outs1))    
        outs3 = self.activation(self.bns3(self.convs3(outs2)) if self.with_bn else self.convs3(outs2))

        outs4 = self.convs4(outs3)

        ff = ME.MinkowskiToDenseTensor()(outs4).permute([0,2,3,4,1])
        
        ff_inv = ff[...,:self.a_out].mean(3)
        gg = ME.MinkowskiToDenseTensor()(ME.SparseTensor(coordinates=ind_map,features=F_ops.flatten(-2).permute([0,2,1,3]).unsqueeze(2).tile(1,1,point_cloud.shape[1],1,1).reshape(-1,9)[ind].squeeze(1))).permute([0,2,3,4,1]).view(B,-1,point_cloud.shape[1],F_ops.shape[1],3,3)
        seeds_displacment = torch.matmul(gg,ff[...,self.a_out:].view(ff.shape[0],ff.shape[1],ff.shape[2],ff.shape[3],-1,3).transpose(-1,-2)).mean(3).transpose(-1,-2)
        output['debug_{0}_equiv_output'.format(iter)] = seeds_displacment
        output["{0}_seeds_features_inv".format(iter)] = ((ME.MinkowskiToDenseTensor()(weights_sparse))[...,0].permute([0,2,3,1]) * ff_inv).sum(1)
        
        current_features_equiv = (ME.MinkowskiToDenseTensor()(weights_sparse))[...,0].permute([0,2,3,1]).unsqueeze(-1) * (seeds_displacment[:,:,:,1:])
        output["{0}_seeds_features_equiv".format(iter)] = current_features_equiv.sum(1)
        seeds_displacment = seeds_displacment[:,:,:,0]
        seeds_pred = (ME.MinkowskiToDenseTensor()(weights_sparse))[...,0].permute([0,2,3,1]) * (ME.MinkowskiToDenseTensor()(ME.SparseTensor(coordinates=ind_map,features=point_cloud.unsqueeze(1).unsqueeze(3).tile(1,weights.shape[2],1,F_ops.shape[1],1).view(-1,3)[ind].squeeze(1)))[...,0].permute([0,2,3,1]) + seeds_displacment  )

        output["{0}_predicted_seeds".format(iter)] = seeds_pred
        

class APENDecoder(Frames_Base):
    def __init__(self,with_bn,is_detach_frame,is_rotation_only,weight_threshold,number_of_groups,in_features_size,num_classes_pred,local_n_size):
        super(APENDecoder, self).__init__(with_bn,is_rotation_only)
        
        
        self.in_features_size = in_features_size
        self.local_n_size = local_n_size
        
        self.conv1 = ME.MinkowskiLinear(self.in_features_size+self.local_n_size*3 + 3,self.in_features_size*3)# torch.nn.Conv1d(3, 64, 1)
        self.conv2 = ME.MinkowskiLinear(self.in_features_size*3,self.in_features_size*4)#torch.nn.Conv1d(64, 128, 1)
        self.conv3 = ME.MinkowskiLinear(self.in_features_size*4,self.in_features_size*5)#torch.nn.Conv1d(128, 128, 1)
        self.conv4 = ME.MinkowskiLinear(self.in_features_size*5,self.in_features_size*6)#torch.nn.Conv1d(128, 512, 1)
        self.conv5 = ME.MinkowskiLinear(self.in_features_size*6,self.in_features_size*7)#torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = ME.MinkowskiBatchNorm(self.in_features_size*3)
        self.bn2 = ME.MinkowskiBatchNorm(self.in_features_size*4)
        self.bn3 = ME.MinkowskiBatchNorm(self.in_features_size*5)
        self.bn4 = ME.MinkowskiBatchNorm(self.in_features_size*6)
        self.bn5 = ME.MinkowskiBatchNorm(self.in_features_size*7)
        self.convs1 = ME.MinkowskiLinear(self.in_features_size*((3+7)*5//2+7),256)# torch.nn.Conv1d(4944 - 16, 256, 1)
        
            
        self.convs2 = ME.MinkowskiLinear(256, 256)#torch.nn.Conv1d(256, 256, 1)
        self.convs3 = ME.MinkowskiLinear(256, 128) #torch.nn.Conv1d(256, 128, 1)
        self.convs4 = ME.MinkowskiLinear(128, num_classes_pred)#torch.nn.Conv1d(128, num_part, 1)
        
        self.bns1 = ME.MinkowskiBatchNorm(256)
        self.bns2 = ME.MinkowskiBatchNorm(256)
        self.bns3 = ME.MinkowskiBatchNorm(128)
        
        self.max_pool = ME.MinkowskiGlobalSumPooling()
        self.concat_op = ME.MinkowskiBroadcastConcatenation()
        self.activation = ME.MinkowskiReLU()
        self.activation_dense = nn.ReLU()
        self.weight_threshold = weight_threshold

    def forward(self,point_cloud,normals,dr_w,weights,iter,output,cand_features_equiv,cand_features_inv,epoch):
        B, N, D = point_cloud.size()

        
        weights = dr_w
        frames,centers,lambdas = self.get_frame(point_cloud,weights)
        output["debug_{0}_lambdas".format(iter)] = lambdas
        output["debug_{0}_frames".format(iter)] = frames
        
        if self.is_rotation_only:
            F_ops = self.ops.unsqueeze(0).unsqueeze(2) * (torch.sign(torch.det(frames[...,:])).unsqueeze(-1).unsqueeze(-1) * frames).unsqueeze(1)
        else:
            F_ops = self.ops.unsqueeze(0).unsqueeze(2) * frames[...,:].unsqueeze(1)
        
        d = (weights.unsqueeze(2) * weights.unsqueeze(1)).sum(-1)*((point_cloud.unsqueeze(2) - point_cloud.unsqueeze(1))**2).sum(-1) + (1-(weights.unsqueeze(2) * weights.unsqueeze(1)).sum(-1))*1000
        idx = d.topk(dim=-1,largest=False,k=self.local_n_size)[1]

        pnts_n = torch.gather(input=point_cloud.unsqueeze(2).tile(1,1,idx.shape[-1],1),index=idx.unsqueeze(-1).tile(1,1,1,3),dim=1)
        
        
        v1 = ((1-weights).transpose(1,2).unsqueeze(-1).unsqueeze(-1) * centers.unsqueeze(-2).tile(1,1,1,self.local_n_size,1) + (weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1) * pnts_n.unsqueeze(1)) - centers.unsqueeze(-2))
        output["debug_{0}_input_before".format(iter)] = v1
        
        framed_input1 = torch.matmul(F_ops.transpose(3,4).unsqueeze(3).tile(1,1,1,v1.shape[2],1,1),v1.unsqueeze(1).tile(1,F_ops.shape[1],1,1,1,1).transpose(-1,-2)).transpose(-1,-2)
        output["debug_{0}_input_aftrer".format(iter)] = framed_input1
        
        normals_ = normals.unsqueeze(2)
        
        v2 = normals_.unsqueeze(1) * weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1) + (1-weights).transpose(1,2).unsqueeze(-1).unsqueeze(-1) * ((normals_.unsqueeze(1) * weights.transpose(1,2).unsqueeze(-1).unsqueeze(-1)).sum(2,True)/(weights.sum(1,True).transpose(1,2).unsqueeze(-1).unsqueeze(-1) + 1e-6))
        v2 = torch.cat([v2,cand_features_equiv.unsqueeze(2).tile(1,1,v2.shape[2],1,1)],dim=3)
        
        output["debug_{0}_v2_before".format(iter)] = v2
        framed_input2 = torch.matmul(F_ops.transpose(3,4).unsqueeze(3).tile(1,1,1,v2.shape[2],1,1),v2.unsqueeze(1).tile(1,F_ops.shape[1],1,1,1,1).transpose(-1,-2)).transpose(-1,-2)#torch.einsum('bokij,bkpl->bokpi',F_ops.transpose(3,4),v2)#.unsqueeze(-1)
        output["debug_{0}_v2_after".format(iter)] = framed_input2

        
        # B x F x W x P x 6
        framed_input = torch.cat([framed_input1.flatten(-2),
                                  framed_input2.flatten(-2),
                                  cand_features_inv.unsqueeze(1).unsqueeze(-2).tile(1,F_ops.shape[1],1,point_cloud.shape[1],1)],dim=-1)
    
        weights_t = weights.unsqueeze(-1).transpose(1,2).tile(1,1,1,framed_input.shape[1])

        # Find all points with weight above threshold. 
        ind = (weights_t.view(-1) >= self.weight_threshold).nonzero()
        # filter input points according to ind
        filtered_input = framed_input.permute([0,2,3,1,4]).reshape(-1, self.local_n_size*3 + 3 + self.in_features_size)[ind].squeeze(1)
        # Flatten batch, weights dimensions and other dimensions are points and frames. ind_map are the filtered indices.
        # i.e., ind_map[i](a,b,c) a is the index of the filtered batch,weight b is the index of the point and c is the index of the frame
        ind_map = (weights_t  >= self.weight_threshold).nonzero().int().contiguous()
        sparse_input = ME.SparseTensor(coordinates=ind_map,features=filtered_input)
        weights_sparse = ME.SparseTensor(coordinates=sparse_input.coordinates,
                                         features = weights_t[(weights_t.detach() >= self.weight_threshold)].unsqueeze(-1),
                                         coordinate_manager=sparse_input.coordinate_manager)
        out1 = self.activation(self.bn1(self.conv1(sparse_input)) if self.with_bn else self.conv1(sparse_input))
        out2 = self.activation(self.bn2(self.conv2(out1)) if self.with_bn else self.conv2(out1))
        out3 = self.activation(self.bn3(self.conv3(out2)) if self.with_bn else self.conv3(out2))
        out4 = self.activation(self.bn4(self.conv4(out3)) if self.with_bn else self.conv4(out3))
        out5 = self.bn5(self.conv5(out4)) if self.with_bn else self.conv5(out4)
        
        kernel_size = [1,ind_map[:,2].max().item()+1,1]
        sum_pool_nom = ME.MinkowskiSumPooling(kernel_size = kernel_size,stride=kernel_size,dimension=3)(out5 * weights_sparse)
        sum_pool_denom = ME.MinkowskiSumPooling(kernel_size = kernel_size,stride=kernel_size,dimension=3)(weights_sparse)
        sum_pool_denom = sum_pool_denom + ME.SparseTensor(coordinates=sum_pool_denom.coordinates,features=1e-5*torch.ones_like(sum_pool_denom.features),coordinate_manager=sum_pool_denom.coordinate_manager)
        out_pool = sum_pool_nom/sum_pool_denom
        
        
        cnt = ME.cat([out1,out2,out3,out4,out5])
        
        concat = self.conc(cnt,out_pool)
        
        outs1 = self.activation(self.bns1(self.convs1(concat)) if self.with_bn else self.convs1(concat))
        outs2 = self.activation(self.bns2(self.convs2(outs1)) if self.with_bn else self.convs2(outs1))    
        outs3 = self.activation(self.bns3(self.convs3(outs2)) if self.with_bn else self.convs3(outs2))

        outs4 = self.convs4(outs3)

        ff = ME.MinkowskiToDenseTensor()(outs4).permute([0,2,3,4,1])
        

        ff_inv = ff.mean(3)
        
        output["seed_part_logits"] = ((ME.MinkowskiToDenseTensor()(weights_sparse))[...,0].permute([0,2,3,1]) * ff_inv).sum(1)


