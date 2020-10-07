# coding: utf-8
import os.path

CONFIG_JSON = 'config.json'
DATA = 'data'
GRAPH_JSON = 'graph.json'
MODEL = 'model'

TASKSDIR = os.path.dirname(os.path.realpath(__file__)) # set to the /sd_lib/task/
MODEL_DIR_NEW = TASKSDIR + '/../../../models/'

class TaskPaths:
    '''
    This is a class for creating and using paths to configuration files and working directoris in working progress
    '''
    MODEL_DIR = MODEL_DIR_NEW
    MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, CONFIG_JSON)
    MODEL_CONFIG_NAME = CONFIG_JSON
