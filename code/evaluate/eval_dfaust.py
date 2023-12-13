import torch
from pyhocon import ConfigFactory
import sys
sys.path.append("../code")
import os
import argparse
import utils.compute as utils
import plotly.offline as offline
from datetime import datetime
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from evaluate.trace import get_scene_traces
import pickle
from dataclasses import dataclass

@dataclass
class SegResult:
    xyz: np.ndarray
    gt_label: np.ndarray
    pd_label: np.ndarray

seg_map = ['Pelvis',
           'Left Hip',
           'Right Hip',
           'Spine 1',
           'Left Knee',
           'Right Knee',
           'Spine 2',
           'Left Ankle',
           'Right Ankle',
           'Spine 3',
           'Left foot',
           'Right foot',
           'Neck',
           'Left Collar',
           'Right Collar',
           'Head',
           'Left Shouldar',
           'Right Shouldar',
           'Left Elbow',
           'Right Elbow',
           'Left Wrist',
           'Right Wrist',
           'Left Hand',
           'Right Hand'
           ]


def evaluate(network,conf,epoch,base_path,expname,no_vis,is_mpi_sorted):
    
    res_iou = []
    conf.get_config('train.dataset.properties')['mode'] = 'test'
    if is_mpi_sorted:
        conf.get_config('train.dataset.properties')['dataset_file'] = conf.get_config('train.dataset.properties')['dataset_file'].split('class.npz')[0] + 'mpi_sorted_class.npz'
    
    network.eval()
        
    ds = utils.get_class(conf.get_string('train.dataset.class'))(**conf.get_config('train.dataset.properties'))
    eval_dataloader = torch.utils.data.DataLoader(ds,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=0, drop_last=False)
        
    all_res = []
    output = os.path.join(base_path,"results/")
    print (output)
    utils.mkdir_ifnotexists(output)
    with torch.no_grad():
        print (len(eval_dataloader))
        for i,data in enumerate(eval_dataloader):        
        

            for key in data:        
                data[key] = utils.get_cuda_ifavailable(data[key]).contiguous()
                
                                    
            outputs = network(data["point_clouds"],data["normals"],epoch,vote_mask=data["vote_mask"])
            seg_pred = outputs['seed_part_logits'].argmax(-1)
                
            gt_segm = data['point_cls_label'].reshape(1, -1).detach()                  # (B, N)
        
            s = conf.get_int('network.num_classes_pred')
            
            onehot_gt = torch.eye(s,device=gt_segm.device)[gt_segm.long()]
            onehot_pred = torch.eye(s,device=seg_pred.device)[seg_pred.long()]
            
            matching_score = (onehot_gt.unsqueeze(-1) * onehot_pred.unsqueeze(-2)).sum(1).long() # B x s x s
            union_score = torch.sum(onehot_gt, dim=1).unsqueeze(-1) + \
                            torch.sum(onehot_pred, dim=1, keepdim=True) - matching_score      # (B, s, s)
            iou_score = matching_score / (union_score + 1e-8)
            print (i,(torch.trace(iou_score[0]))/(s*1.0))
            all_res.append(SegResult(xyz=data["point_clouds"][0].detach().cpu().numpy(),gt_label=gt_segm[0].cpu().numpy(),pd_label=seg_pred[0].detach().cpu().numpy()))
            res_iou.append((torch.trace(iou_score[0])).cpu().item()/(s*1.0))
            
            if not no_vis:
                traces = get_scene_traces(pnts=outputs["sampled_points"],
                                                                        outputs=outputs,
                                                                        normals = utils.vector_gather(data["normals"],outputs["sample_indices"]),
                                                                        cap="",
                                                                        layers=conf.get_int('network.num_layers'),
                                                                        seg_map=seg_map
                                                                        )
                                
                fig = go.Figure()
                b = 2.5
                fig.layout.scene.update(dict(#camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),
                                                            xaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                            yaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                            zaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                            aspectratio=dict(x=1, y=1, z=1)))
                [fig.add_trace(a) for a in traces]

                

                filename = os.path.join(output,"{0}_{1}_{2}".format(expname,epoch,i))
                offline.plot(fig, filename=filename + '.html', auto_open=False)
            
            
    pd.DataFrame({'iou':res_iou}).to_csv(os.path.join(output,"iou_{0}_{1}.csv".format(expname,epoch)))
    pd.DataFrame({'iou':np.array(res_iou).mean(keepdims=True),'iou_std':np.array(res_iou).std(keepdims=True)}).to_csv(os.path.join(output,"iou_mean_{0}_{1}.csv".format(expname,epoch)))
    filename = os.path.join(output,"{0}_{1}".format(expname,epoch))
    with open('{0}.pkl'.format(filename), 'wb') as f:
        pickle.dump(all_res, f)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--expsfolder", type=str, default="")
    parser.add_argument("--expname", type=str,default="dfaust")
    parser.add_argument("--timestamp", type=str, default="latest")
    parser.add_argument("--root", type=str, default="../")
    parser.add_argument("--epoch", type=str, default="latest")
    parser.add_argument("--no_vis",default=False,action="store_true")
    parser.add_argument("--old_form",default=False,action="store_true")
    parser.add_argument("--mpi_sorted",default=False,action="store_true")

    

    opt = parser.parse_args()

    folder_name = opt.expsfolder
    expname = opt.expname
    root = opt.root
    timestamp = opt.timestamp
    epoch = opt.epoch
    no_vis = opt.no_vis
    old_form = opt.old_form
    mpi_sorted = opt.mpi_sorted


    

    base_path = os.path.join(root,'exps/{0}/{1}'.format(folder_name,expname))#,timestamp))
    if timestamp =='latest':
        timestamp = sorted([(datetime.strptime(t, '%Y_%m_%d_%H_%M_%S'),t) for t in  os.listdir(base_path) if not 'DS_Store' in t and not 'last_plot' in t],reverse=True,key=lambda x: x[0])[0][1]

    base_path = os.path.join(base_path,timestamp)

    print (base_path)
    conf_path = os.path.join(base_path,'code','confs','runconf.conf')

    with open(conf_path) as f:
        lines = f.readlines()

    lines[0] = lines[0].replace("""../../""","")

    with open(os.path.join(base_path,'code','confs','runconf1.conf'), "w") as f:
        f.writelines(lines)
        
    conf = ConfigFactory.parse_file(os.path.join(base_path,'code','confs','runconf1.conf'))

    if old_form:
        conf.get_config('train.dataset.properties')['dataset_file'] =  '/'.join(conf.get_config('train.dataset.properties')['dataset_file'].split('/')[:-1] +  ['_' + conf.get_config('train.dataset.properties')['dataset_file'].split('/')[-1]])
    
    #conf.get_config('network')['pik_thrshold']  = 0.01
    #conf.get_config('network')['merge_thr']  = 1e-1
    #conf.get_config('train.dataset.properties')['shape_path'] = '.' + conf.get_config('train.dataset.properties')['shape_path']
    
    #/Volumes/sambashare/gen_equiv/exps/toy/2023_07_29_10_15_45/checkpoints/ModelParameters/latest.pth
    saved_model_state = torch.load(os.path.join(base_path,'checkpoints', 'ModelParameters', epoch + ".pth"))
    #saved_model_state = torch.load(os.path.join(base_path,, epoch + ".pth"))
    #
    network = utils.get_class(conf.get_string('train.network_class'))(**conf.get_config('network')).cuda()
    network.load_state_dict(saved_model_state['model_state_dict'])
    epoch = saved_model_state['epoch']
    network.eval()
    print (epoch)
    print ("current dir {0}".format(os.getcwd()))

    evaluate(network,conf,epoch,base_path,expname,no_vis,mpi_sorted)
            