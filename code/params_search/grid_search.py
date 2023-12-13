import numpy as np
import os
import json
import sys
import numpy as np
sys.path.append('.')
from params_search.base_runner import BaseRunner
import itertools
from git import Repo

class GridSearch(BaseRunner):

    def __init__(self, 
                folder,
                 parameters,
                 params_all,
                 conf_name,
                 name,
                 gpu_num=1,
                 retries=1,
                 class_trainer='train_v2',
                 cluster='waic',
                 gpu='auto',
                 workers=16,
                 batch_size=64,
                 cancel_parallel=False,
                 exps_folder='exps',
                 nepochs=10000,
                 is_continue=True,
                 base_path='',
                 gpu_mem='16',
                 ):
 
        super().__init__(gpu_num=gpu_num,retries=retries,class_trainer=class_trainer,cluster=cluster,gpu=gpu,workers=workers,batch_size=batch_size,cancel_parallel=cancel_parallel,exps_folder=exps_folder,nepochs=nepochs,is_continue=is_continue,base_path=base_path,gpu_mem=gpu_mem)
        self.parameters = parameters
        self.params_all = params_all
        self.folder = folder
        self.conf_name = conf_name
        self.name = name

    def prepare_splits_and_conf(self):
        params_list = list(self.parameters.items())
        runs = []
        for i,params in enumerate(list(itertools.product(*[[(k,x,v[1]) for x in v[0]] for k,v in params_list]))):
        
            str = ["""include "../../{0}.conf" """.format(self.conf_name)]
            str = str + self.params_all
            run_name = ''
            for k,v,w in params:
                str.append('{0} = {1}'.format(k,v))
                if 'learning_rate_schedule' in k:
                    run_name += '{0}_{1}_'.format(w,v[1]['Initial'])
                elif 'dataset_file' in k:
                    run_name += '{0}_{1}_'.format(w,v.split('/')[-1].split('.npz')[0])
                elif type(v) is list:
                    run_name += '_'.join(["{0}".format(i) for i in v])
                else:
                    run_name += '{0}_{1}_'.format(w,v)
            
            run_name = '{0}_{1}'.format(self.name,run_name[:-1])
            run_name = run_name.replace('[','').replace(']','').replace(', ','_')
            str.append('train.expname = {0}'.format(run_name))
            

            with open(os.path.join(self.folder,'{0}.conf'.format(run_name)), "w") as text_file:
                text_file.write('\n'.join(str).replace('\'',"\""))
            #utils.mkdir_ifnotexists("{0}/exps_{1}".format(all_exps_folder,exps_folder))
            self.runs_conf.append([os.path.join(self.folder,'{0}.conf'.format(run_name)),run_name])
            self.runs.append(dict(self.params,**{'conf':os.path.join(self.folder,'{0}.conf'.format(i))}))
        
        repo = Repo('../')
        repo.git.add('./code/confs')
        repo.index.commit('grid confs commit')
        repo.git.push()
        