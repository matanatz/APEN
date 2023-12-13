import torch
import logging
import scipy
import utils.compute as utils
import torch.nn.functional as F
import numpy as np
from utils.nn_distance import huber_loss

class SegLoss(torch.nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()
        self.ltype=kwargs['type']
        self.only_last=kwargs['only_last']
        self.seg_weight = kwargs['seg_weight']
        
        logging.debug ("""
        seg_weight = {0}
                       """.format(self.seg_weight))



    def forward(self, network_outputs,labels,epoch):
        debug = {}
        
        
        en = 4

        seed_gt_votes_mask = torch.gather(input=labels["vote_mask"],dim=1,index=network_outputs["sample_indices"]).float()
        point_gt_mask = torch.gather(input=labels["point_mask"],dim=1,index=network_outputs["sample_indices"]).float()   
        point_gt_mask = point_gt_mask.flatten(0, 1).long()  
        seed_gt_votes =  torch.gather(input=labels["point_clouds"],dim=1,index=network_outputs["sample_indices"].unsqueeze(-1).tile(1,1,labels["point_clouds"].shape[-1]))[...,:3] + torch.gather(input=labels["vote_label"],dim=1,index=network_outputs["sample_indices"].unsqueeze(-1).tile(1,1,labels["vote_label"].shape[-1]))[...,:3]
        
        debug["seeds_gt_votes"] = seed_gt_votes
        debug["seeds_gt_votes_mask"] = seed_gt_votes_mask

        
        weights = np.linspace(1,1,en)
        
        votes_loss = 0.0
        
        for i in range(en - 1 if self.only_last else 0,en):
            
            delta = network_outputs["{0}_predicted_seeds".format(i)].sum(1) - seed_gt_votes
            if self.ltype == 'l1':
                error = delta.abs().sum(-1)
                error_other = ((network_outputs["{0}_predicted_seeds".format(i)].sum(1) - torch.tensor([0,0,-1]).view(1,1,3).to(network_outputs["{0}_predicted_seeds".format(i)])).abs()).sum(-1)
            elif self.ltype == 'l2':
                error = (delta**2).sum(-1)
                error_other = ((network_outputs["{0}_predicted_seeds".format(i)].sum(1) - torch.tensor([0,0,-1]).view(1,1,3).to(network_outputs["{0}_predicted_seeds".format(i)]))**2).sum(-1)
            c_vote_loss = weights[i]*(1*(seed_gt_votes_mask * error ).sum(-1) / seed_gt_votes_mask.sum(-1) + 1*((-seed_gt_votes_mask + 1) * error_other).sum(-1)/((-seed_gt_votes_mask + 1).sum(-1) + 1e-5) ).mean()
            
            votes_loss += c_vote_loss
            debug['vote_{0}_loss'.format(i)] = c_vote_loss
            
        criterion_point_part_cls = torch.nn.CrossEntropyLoss(reduction='mean')
        point_part_loss =criterion_point_part_cls(network_outputs['seed_part_logits'].view(-1,network_outputs['seed_part_logits'].shape[-1]),labels['point_cls_label'].long().flatten())
        debug['point_part_loss'] = point_part_loss
        
        total_loss =  votes_loss  + self.seg_weight*point_part_loss
        debug['total_loss'] = total_loss
        
        return {"loss": total_loss,"loss_monitor":debug}