import utils.compute as utils
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import time
import numpy as np
import json
import logging
import pandas as pd
import torch
import GPUtil
from utils.visdomer import Visdomer
import socket
import plotly.graph_objs as go
#from pytorch3d.transforms import  Rotate, random_rotations
import utils.plots as  plt
import plotly.offline as offline
from plotly.subplots import make_subplots
import functools
import plotly.express as px
from evaluate.trace import get_scene_traces


class BaseTrainRunner():
    def __init__(self,**kwargs):

        if (type(kwargs['conf']) == str):
            self.conf = ConfigFactory.parse_file(kwargs['conf'])
            self.conf_filename = kwargs['conf']
        else:
            self.conf = kwargs['conf']
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.expnameraw = self.conf.get_string('train.expname')
        self.expname = self.conf.get_string('train.expname') +  kwargs['expname']
        debug_conf = {}
        debug_conf['batch_size'] = str(self.batch_size)

        self.step_log = {}
        self.epoch_log = {}

        base_path = kwargs['base_path']#self.conf.get_string('train.base_path')


        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(base_path,kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join(base_path,kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = None
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']


        self.adjust_lr = self.conf.get_bool('train.adjust_lr')

        if kwargs['gpu_index'] == "auto":
            deviceIDs = GPUtil.getAvailable(
                order="memory",
                limit=1,
                maxLoad=0.5,
                maxMemory=0.5,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
            gpu = deviceIDs[0]
        else:
            gpu = kwargs['gpu_index']
        self.GPU_INDEX = gpu
        self.exps_folder_name = kwargs['exps_folder_name']

        utils.mkdir_ifnotexists(os.path.join(base_path,self.exps_folder_name))

        self.expdir = os.path.join(base_path, self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        self.log_dir = log_dir
        utils.mkdir_ifnotexists(log_dir)
        utils.configure_logging(kwargs['debug'],kwargs['quiet'],os.path.join(self.log_dir,'log.txt'))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        self.last_plot_dir = os.path.join(base_path, self.exps_folder_name,self.expname,"last_plot")
        utils.mkdir_ifnotexists(self.last_plot_dir)
        self.visdom_env = '~'.join([self.exps_folder_name.replace('_','-'), self.expname.replace('_','-'), self.timestamp.replace('_','-')])
        utils.mkdir_ifnotexists(self.plots_dir)
        # self.visdomer = Visdomer(self.conf.get_string('train.visdom_server'), expname=self.expname, timestamp=self.timestamp,port=self.conf.get_int('train.visdom_port'), do_vis=kwargs['vis'])
        # self.window = [None,None]
        is_loaded = False

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_subddir = "SchedulerParamaeters"
        
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path,self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_subddir))

        # Backup code
        self.code_path = os.path.join(self.expdir, self.timestamp, 'code')
        utils.mkdir_ifnotexists(self.code_path)
        for folder in ['training','preprocess','evaluate','utils','model','datasets','confs']:
            utils.mkdir_ifnotexists(os.path.join(self.code_path, folder))
            os.system("""cp -r ./{0}/* "{1}" """.format(folder,os.path.join(self.code_path, folder)))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.code_path, 'confs/runconf.conf')))

        logging.info('shell command : {0}'.format(' '.join(sys.argv)))

        self.train_noise_level = self.conf.get_float('train.train_noise_level')
        self.save_learning_log_freq = self.conf.get_int('train.save_learning_log_freq')
        self.learning_log_epoch_path = os.path.join(self.log_dir, 'learning_epoch_log.csv')
        self.learning_log_step_path = os.path.join(self.log_dir, 'learning_step_log.csv')
        self.debug_log_conf_path = os.path.join(self.log_dir, 'debug_conf.csv')
        self.train_rot = self.conf.get_string('train.train_rot')
        self.test_rot = self.conf.get_string('train.test_rot')
        self.workers = kwargs['workers']
        
        logging.info('after creating data set')
        self._set_loader()

        self.ds_len = len(self.ds)
        logging.info("data set size : {0}".format(self.ds_len))

        self.parallel = kwargs['parallel']
        self.grad_clip = self.conf.get_float('train.grad_clip')
        self.num_forwards = self.conf.get_int('train.num_forwards')
        self.num_layers = self.conf.get_int('network.num_layers')
        self._set_network_loss()
        if self.parallel:
            self.network = torch.nn.DataParallel(self.network)
            logging.info("GPU parallel mode")
        else:
            logging.info("no parallel")
        
        self.network = utils.get_cuda_ifavailable(self.network)
        self.loss = utils.get_cuda_ifavailable(self.loss)

        self._set_optimizer()
        self.epoch_checkpoint_frequency = self.conf.get_int('train.epoch_checkpoint_frequency')
        self.cancel_visdom = self.conf.get_bool('train.cancel_visdom')
        if not self.cancel_visdom:
            self.visdomer = Visdomer(self.conf.get_string('train.visdom_server'), expname=self.expname, timestamp=self.timestamp,port=self.conf.get_int('train.visdom_port'), do_vis=kwargs['vis'])
            self.window = [None,None]
        is_loaded = False
        self.start_epoch = -1
        if is_continue:

            if kwargs['timestamp'] == 'latest':
                potential_timestamps = ['{:%Y_%m_%d_%H_%M_%S}'.format(t) for t in sorted([datetime.strptime(t, '%Y_%m_%d_%H_%M_%S') for t in  os.listdir(os.path.join(base_path,kwargs['exps_folder_name'],self.expname)) if not 'DS_Store' in t and not 'last_plot' in t],reverse=True)]
            else:
                potential_timestamps = [timestamp]

            i = 0
            while not is_loaded and i < len(potential_timestamps):
                timestamp = potential_timestamps[i]
                old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

                if os.path.isfile(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
                    try:
                        
                            
                        logging.info('after loading late vec')
                        saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
                        self.network.load_state_dict(saved_model_state['model_state_dict'])
                        

                        logging.info('after loading model')
                        data = torch.load(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
                        self.optimizer.load_state_dict(data["optimizer_state_dict"])

                        if not self.scheduler is None:
                            data = torch.load(os.path.join(old_checkpnts_dir, self.scheduler_subddir, str(kwargs['checkpoint']) + ".pth"))
                            self.scheduler.load_state_dict(data['scheduler_state_dict'])


                        logging.info('after loading optimizer')
                        self.start_epoch = saved_model_state['epoch']
                        is_loaded = True

                        if os.path.isfile(os.path.join(self.expdir, timestamp, 'log', 'learning_epoch_log.csv')):
                            self.epoch_log = pd.read_csv(
                                os.path.join(self.expdir, timestamp, 'log', 'learning_epoch_log.csv')).to_dict()
                            self.epoch_log.pop('Unnamed: 0', None)
                            for k in self.epoch_log.keys():
                                self.epoch_log[k] = list(self.epoch_log[k].values())
                        else:
                            self.epoch_log = {}

                        if os.path.isfile(os.path.join(self.expdir, timestamp, 'log', 'learning_step_log.csv')):
                            self.step_log = pd.read_csv(
                                os.path.join(self.expdir, timestamp, 'log', 'learning_step_log.csv')).to_dict()
                            for k in self.step_log.keys():
                                self.step_log[k] = list(self.step_log[k].values())

                            self.step_log.pop('Unnamed: 0', None)
                        else:
                            self.step_log = {}
                    except Exception as e:
                        logging.info ('something went wrong in load timestamp : {0}'.format(timestamp))
                        logging.info (str(e))
                        i = i + 1
                else:
                    i = i + 1

        #pd.DataFrame(debug_conf).to_csv(self.debug_log_conf_path)
        if not is_loaded:
            logging.info("---------NO TIMESTAMP LOADED------------------")

        #self.code_reg_lambda = self.conf.get_float('train.code_reg_lambda')
        #self.test_epochs = self.conf.get_list('train.test_after')

        logging.info('hostname : {0}'.format(socket.gethostname()))



    # def latent_size_reg(self, latent):
    #     latent_loss = 0.0
        
    #     latent_loss = torch.mean(latent.pow(2))
    #     return latent_loss
    
    def _set_optimizer(self):
        self.lr_schedules = BaseTrainRunner.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))

        opt_params = [{"params": self.network.parameters(), "lr": self.lr_schedules[0].get_learning_rate(0)}]
        self.optimizer = torch.optim.AdamW(opt_params)
        self.scheduler = None
        
    def _set_network_loss(self):
        logging.info ("network class {0}".format(self.conf.get_string('train.network_class')))
        self.network = utils.get_class(self.conf.get_string('train.network_class'))(**self.conf.get_config('network'))
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))
        

    def _set_loader(self):

        
        self.ds = utils.get_class(self.conf.get_string('train.dataset.class'))(**self.conf.get_config('train.dataset.properties'))
        d = self.conf.get_config('train.dataset.properties')
        d['mode'] = "test"
        self.ds_eval = utils.get_class(self.conf.get_string('train.dataset.class'))(**d)
        
        
            

        self.dataloader = torch.utils.data.DataLoader(self.ds,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=self.workers,drop_last=True,pin_memory=True)
        self.eval_dataloader = torch.utils.data.DataLoader(self.ds_eval,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=0, drop_last=True)
                                                      
    def get_learning_rate_schedules(schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self):
        d = self.epoch_checkpoint_frequency
        if self.epoch % d == 0:
            torch.save(
                {"epoch": self.epoch, "model_state_dict": self.network.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(self.epoch) + ".pth"))
        torch.save(
            {"epoch": self.epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        if self.epoch % d == 0:
            torch.save(
                {"epoch": self.epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(self.epoch) + ".pth"))
        torch.save(
            {"epoch": self.epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        
        if not self.scheduler is None:
            if self.epoch % d == 0:
                torch.save(
                    {"epoch": self.epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                    os.path.join(self.checkpoints_path, self.scheduler_subddir, str(self.epoch) + ".pth"))

            torch.save(
                    {"epoch": self.epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                    os.path.join(self.checkpoints_path, self.scheduler_subddir, "latest.pth"))
        

    def save_learning_log(self, epoch_log,step_log):
        pd.DataFrame(epoch_log).to_csv(self.learning_log_epoch_path)
        pd.DataFrame(step_log).to_csv(self.learning_log_step_path)



class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)),1.0e-5)


class TrainRunner(BaseTrainRunner):
    
    def run(self):
        win = None
        win_surface = None
        timing_log = []
        loss_log_epoch = []
        lr_log_epoch = []
        logging.debug("*******************running*********")
        self.epoch = 0
        j = 0
        for epoch in range(self.start_epoch+1, self.nepochs + 2):
            self.epoch = epoch
            
            start_epoch = time.time()
            batch_loss = 0.0

            if (epoch % self.conf.get_int('train.save_checkpoint_frequency') == 0 or epoch == self.start_epoch) and epoch >= 0:
                self.save_checkpoints()
            if epoch % self.conf.get_int('train.plot_frequency') == 0 and epoch >= 0:
                logging.debug("in plot")
                self.network.eval()
                # if self.conf.get_bool('train.eval_ensm'):
                #     evaluate_ens(self.network,self.conf,epoch,os.path.join(self.expdir, self.timestamp),self.expname)
                # else:
                #     evaluate(self.network,self.conf,epoch,os.path.join(self.expdir, self.timestamp),self.expname)
                if True: #with torch.no_grad():
                    
                    data = next(iter(self.eval_dataloader))
                    
                    shuf = np.arange(data["point_clouds"].shape[1])
                    np.random.shuffle(shuf)
                    for key in data:
                        # if key in ["point_clouds","normals","vote_label","vote_mask","point_mask","pose_label","pose_angle","point_cls_label","sym_label",
                        #             "point_cloud_color"]:
                        #     #data[key] = data[key][:,shuf]
                        #     pass
                        data[key] = utils.get_cuda_ifavailable(data[key]).contiguous()
                    
                    
                    outputs = self.network(data["point_clouds"][0:1],data["normals"][0:1],epoch,vote_mask=data["vote_mask"][0:1])
                    #loss = self.loss(network_outputs=outputs,labels=data,epoch=epoch)
                    #diff = loss["loss_monitor"]["diff"]

                    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"},{"type": "scene"}],
                                                            [{"type": "scene"},{"type": "scene"}]],
                                                            subplot_titles=("Pred", "a","b","c"))
                    
                    b = 2
                    fig.layout.scene.update(dict(#camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),
                                                xaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                yaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                zaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
                    fig.layout.scene2.update(dict(#camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),
                                                xaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                yaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                zaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
                    fig.layout.scene3.update(dict(#camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),
                                                xaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                yaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                zaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
                    fig.layout.scene4.update(dict(#camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),
                                                xaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                yaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                zaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                aspectratio=dict(x=1, y=1, z=1)))  
                    
                    fig = go.Figure()
                    fig.layout.scene.update(dict(#camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),
                                                xaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                yaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                zaxis=dict(range=[-b, b], autorange=False,visible=True,showbackground=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
                    
                    logging.debug("before plot")
                    
                    traces = get_scene_traces(pnts=outputs["sampled_points"],
                                                            outputs=outputs,
                                                            normals = data["normals"],
                                                            cap="",
                                                            layers=self.num_layers,
                                                            )
                    
                    
                    #[fig.add_trace(a,row=1,col=1) for a in traces]
                    [fig.add_trace(a) for a in traces]

                    filename = '{0}/res.html'.format(self.plots_dir)
                    filename_last = '{0}/res.html'.format(self.last_plot_dir)
                    
                    logging.debug("saving file : {0}".format(filename))
                    offline.plot(fig, filename=filename, auto_open=False)
                    offline.plot(fig, filename=filename_last, auto_open=False)
                    logging.debug("after plot")           
                    

            self.network.train()
            if (self.adjust_lr):
                self.adjust_learning_rate(epoch)
            logging.debug('before data loop {0}'.format(time.time()-start_epoch))
            before_data_loop = time.time()
            data_index = 0
            j = 0
            
            for data in self.dataloader:
                logging.debug('in loop data {0}'.format(time.time()-before_data_loop))
                start = time.time()
                shuf = np.arange(data["point_clouds"].shape[1])
                if j % 1 == 0:
                    shuf_ = []
                    for j in range(self.batch_size):
                        np.random.shuffle(shuf)
                        shuf_.append(np.expand_dims(shuf,0).copy())
                    shuf = np.concatenate(shuf_,axis=0)
                for key in data:
                    if key in ["point_clouds","normals","vote_label","vote_mask","point_mask","pose_label","pose_angle","point_cls_label","sym_label",
                                "point_cloud_color"]:
                        data[key] = utils.vector_gather(data[key],torch.tensor(shuf))
                        
                    data[key] = utils.get_cuda_ifavailable(data[key]).contiguous()
                
                j = j + 1
                
                outputs = self.network(data["point_clouds"] + self.train_noise_level*torch.randn_like(data["point_clouds"]) ,
                                       data["normals"],
                                       epoch,
                                       vote_mask=data["vote_mask"])
                
                loss_res = self.loss(network_outputs=outputs, labels=data,epoch=epoch)#,centers=centers,pnts=pnts,normals=normals)
                    
                loss = loss_res["loss"]#.mean()
                loss.backward()
                
                logging.debug('after backward  {0}'.format(time.time()-start))
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
                self.optimizer.step()

                if 'total_loss' in self.step_log:
                    len_step_loss = len(self.step_log['total_loss'])
                else:
                    len_step_loss = 0
                for k in loss_res['loss_monitor'].keys():
                    if k in self.step_log:
                        self.step_log[k].append(loss_res['loss_monitor'][k].mean().item())
                    else:
                        if len_step_loss > 0:
                            self.step_log[k] = [0.0]*len_step_loss + [loss_res['loss_monitor'][k].mean().item()]
                        else:
                            self.step_log[k] =[loss_res['loss_monitor'][k].mean().item()]
                if "loss_mean" in self.step_log:
                    tt = np.array([(p.grad**2).mean().item() for p in self.network.parameters() if not p.grad is None])
                    self.step_log["loss_grad_mean"].append(tt.mean())
                else:
                    if len_step_loss > 0:

                        tt = np.array([(p.grad**2).mean().item() for p in self.network.parameters() if not p.grad is None])
                        self.step_log["loss_grad_mean"] = [0.0]*len_step_loss + [tt.mean()]

                    else:
                        tt = np.array([(p.grad**2).mean().item() for p in self.network.parameters() if not p.grad is None])
                        self.step_log["loss_grad_mean"] = [tt.mean() + tt.mean()]

                batch_loss += loss.item()
                logging.debug("expname : {0}".format(self.expname))
                logging.debug("timestamp: {0} , epoch : {1}, data_index : {2} , loss_mean : {3} , vote_10_loss{4} ".format(self.timestamp,
                                                                                                                                                epoch,
                                                                                                                                                data_index,
                                                                                                                                                loss.item(),
                                                                                                                                                loss_res["loss_monitor"]['total_loss'].mean().item()))
                self.optimizer.zero_grad(True)
                for param in self.network.parameters():
                    param.grad = None
                
                data_index = data_index + 1
                

                
            lr_log_epoch.append(self.optimizer.param_groups[0]["lr"])
            loss_log_epoch.append(batch_loss / (self.ds_len // self.batch_size))
            end = time.time()
            seconds_elapsed_epoch = end - start
            timing_log.append(seconds_elapsed_epoch)

            if (epoch % self.save_learning_log_freq == 0):
                trace_steploss = []
                selected_stepdata = pd.DataFrame(self.step_log)
                for x in selected_stepdata.columns:
                    if 'loss' in x:
                        trace_steploss.append(
                            go.Scatter(x=np.arange(len(selected_stepdata)), y=selected_stepdata[x], mode='lines',
                                       name=x))

                fig = go.Figure(data=trace_steploss)

                env = '/'.join([self.expname, self.timestamp])
                if not self.cancel_visdom:
                    if win is None:
                        win = self.visdomer.plot_plotly(fig, env=env)
                    else:
                        self.visdomer.plot_plotly(fig, env=env,win=win)

                self.save_learning_log(epoch_log=dict(epoch=range(self.start_epoch+1, epoch + 1),
                             loss_epoch=loss_log_epoch,
                             time_elapsed=timing_log,
                             lr_epoch=lr_log_epoch),
                                       step_log=self.step_log)

