import os,json,cortex
import numpy as np
from config.dir import DATA_DIR, REPO_DIR,RESULTS_DATA_DIR,FEATURE_DATA_DIR
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.utils import generate_leave_one_run_out
from sklearn.model_selection import check_cv
import matplotlib.pyplot as plt
from utils.npp import zscore


def open_json(subj,file):
    
    dir=os.path.join( RESULTS_DATA_DIR,subj)
    with open(os.path.join(dir,file), "r")  as f:
        data = json.load(f)
    return  np.array(data, dtype=float)



