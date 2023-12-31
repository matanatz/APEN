from scipy.spatial import cKDTree as KDTree
import argparse
import sys
# python training/exp_runner.py --batch_size 2 --expname _uniform --workers 0 --nepoch 100000 --gpu auto --conf ./confs/dfaust_fix_latent_small_local.conf
sys.path.append("../code")
import os
import GPUtil


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Input batch size.")
    
    parser.add_argument(
        "--nepoch", type=int, default=100, help="Number of epochs to train."
    )
    parser.add_argument("--conf", type=str, default="./confs/dfaust.conf")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--expsfolder", type=str, default="exps")
    parser.add_argument(
        "--gpu", type=str, default="auto", help="GPU to use [default: GPU auto]."
    )
    parser.add_argument(
        "--parallel",
        default=False,
        action="store_true",
        help="If set, indicaties running on multiple gpus.",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Data loader number of workers."
    )
    parser.add_argument(
        "--is_continue",
        default=False,
        action="store_true",
        help="If set, indicates continuing from a previous run.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default='latest',
        help="The timestamp of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='latest',
        help="The checkpoint epoch number of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--debug",
        default=True,
        action="store_true",
        help="If set, debugging messages will be printed.",
    )

    parser.add_argument(
        "--cancel_vis",
        default=False,
        action="store_true",
        help="If set, cancel visualize plots in visdom.",
    )

    parser.add_argument(
        "--quiet",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed.",
    )

    parser.add_argument(
        "--trainer",
        dest="trainer",
        default='train_dfaust',
        type=str
    )

    parser.add_argument(
        "--base_path",
        dest="base_path",
        default='../',
        type=str
    )

    #os.environ['OMP_NUM_THREADS'] = "1"
    opt = parser.parse_args()

    if opt.gpu == "auto":
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
        gpu = opt.gpu

    if (not opt.gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    else:
        print ("No gpu selected")
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
    import torch
    torch.set_num_threads(4)
    print ("torch available gpus : {0}".format(torch.cuda.device_count()))
    print ("torch available gpus : {0}".format(torch.torch.version.cuda))
    import utils.general as utils

    trainrunner = utils.get_class("training.{0}.TrainRunner".format(opt.trainer))(
        conf=opt.conf,
        batch_size=opt.batch_size,
        nepochs=opt.nepoch,
        expname=opt.expname,
        gpu_index=opt.gpu,
        exps_folder_name=opt.expsfolder,
        parallel=opt.parallel,
        workers=opt.workers,
        is_continue=opt.is_continue,
        timestamp=opt.timestamp,
        checkpoint=opt.checkpoint,
        debug=opt.debug,
        quiet=opt.quiet,
        vis=not opt.cancel_vis,
        base_path=opt.base_path
    )

    trainrunner.run()
