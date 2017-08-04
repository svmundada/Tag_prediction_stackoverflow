import os
import sys
import json
from datetime import datetime

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    max_length = 58 # longest sequence to parse
    n_classes = 16
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 64
    n_epochs = 10
    lr = 0.001
    dir_path = "/home/iota/Downloads/do_not_open/v1/top16/" # path of directory in which the all code is.
    output_path = dir_path+"results/{:%Y%m%d_%H%M%S}/".format(datetime.now()) # path of directory where results for
                                                                                #current exp are stored.

    previous_path = "/home/iota/Downloads/do_not_open/v1/top16/results/20170803_204948/" # If reusing weights

    # previous_path = None

    eval_path = previous_path # For eval path, use when validation being performed for development set.
                                                                              

    def __init__(self, result_folder_bool=True):

        self.result_folder = result_folder_bool
        # create results directory.
        if not os.path.exists(Config.dir_path+"results/"):
            os.makedirs(Config.dir_path+"results/")

        # create folder in results directory for storing experiments according to date and time.
        

    def create_output_folder(self):
        if self.result_folder and not os.path.exists(Config.output_path):
            os.makedirs(Config.output_path)



    def store_config_info(self):
        store = {'max_length':Config.max_length,
        'n_classes':Config.n_classes,
        'dropout':Config.dropout,
        'hidden_size':Config.hidden_size,
        'batch_size':Config.batch_size,
        'n_epochs':Config.n_epochs,
        'lr':Config.lr,
        "dir_path" : Config.dir_path,
        "previous_path" : Config.previous_path,
        'eval_path' : Config.eval_path}

        self.create_output_folder()

        with open(Config.output_path+'Config_dict.json', 'w') as f:
            json.dump(store, f)


    def use_config_info(self, configpath):
        with open(configpath, 'r') as f:
            store = json.load(f)

            Config.max_length = store['max_length']
            Config.n_classes = store['n_classes']
            Config.dropout = store['dropout']
            Config.hidden_size = store['hidden_size']
            Config.n_epochs = store['n_epochs']
            Config.lr = store['lr']
            Config.dir_path = store['dir_path']
            Config.previous_path = store['previous_path']
            Config.eval_path = store['eval_path']
