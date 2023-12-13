import sys
import numpy as np
sys.path.append('.')
from params_search.grid_search import GridSearch
import utils.general as utils


if __name__ == "__main__":

    name = 'dfaust'
    folder = './confs/grid_search/{0}/'.format(name)
    utils.mkdir_ifnotexists(folder)
    parameters = {
                'train.dataset.properties.dataset_file': ([
                    '/workspace/dfaust/_x_random_pose.npz',
                    '/workspace/dfaust/_x_unseen_random_pose.npz',
                    '/workspace/dfaust/_x_unseen_same_pose.npz'],'split'),
                }

    #runner(runs[0])
    a = GridSearch(
                   folder=folder,
                   parameters=parameters,
                   params_all=[],
                   conf_name="dfaust_grid",
                   class_trainer='train_dfaust',
                   batch_size=2,
                   workers=4,
                   exps_folder='exps/{0}'.format(name),
                   name=name,
                   cancel_parallel=True,
                   gpu_num=1,
                   retries=10,
                   is_continue=True,
                   cluster='ngc',
                   base_path='/workspace',
                   gpu_mem='32',nepochs=100)
    a.run()

   