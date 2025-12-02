import os,json,argparse,logging,itertools
import numpy as np
from config.dir import DATA_DIR, EM_DATA_DIR,RESULTS_DATA_DIR,FEATURE_DATA_DIR
import matplotlib.pyplot as plt
import sklearn.metrics


def load_primal_coef(subjs,sess,target,model,save_voxel):
    if subjs == 'sub-UTS02':
        file_name = subjs+'_'+sess+model+'_primal_coef'
        directory=os.path.join(RESULTS_DATA_DIR,subjs,sess,model+'_model')

    else: 
        file_name = subjs+"_"+target+'_'+sess+model+'_primal_coef'
        directory=os.path.join(RESULTS_DATA_DIR,subjs,sess+'_'+target,model+'_model')
    best_voxels = np.load(os.path.join(RESULTS_DATA_DIR,subj,save_voxel,'semantic_model/best_voxels.npy'))
    print(f"Loading {file_name}.npy from {dir}")
    primal_coef_r=np.load(os.path.join(directory,file_name+'_r.npy'))
    primal_coef_r2=np.load(os.path.join(directory,file_name+'_r2.npy'))
    #return primal_coef_r2[:, best_voxels], primal_coef_r[:, best_voxels]
    return primal_coef_r2, primal_coef_r

def compute_r(weight,weight2,sess):
    """
    Compute pairwise Pearson correlations between weight maps for labels 'a','b','c'.
    Returns:
      r_dict: dict with keys like 'ab', 'ac', 'bc' and correlation values
      average_r: mean of the three correlations
    """
    labels = ['a', 'b', 'c']
    r = []
    r2 =[]

    for i, j in itertools.combinations(labels, 2):
        key1 = f'{sess}{i}'
        key2 = f'{sess}{j}'
        print(f"Computing correlation between {key1} and {key2}")
        w1 = weight.get(key1)
        w2 = weight.get(key2)
        w1_2 = weight2.get(key1)
        w2_2 = weight2.get(key2)
        if w1 is None or w2 is None:
            raise KeyError(f"Missing weights for keys: {key1} or {key2}")
        corr = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]
        corr2= sklearn.metrics.r2_score(w1_2.ravel(), w2_2.ravel())
        r.append(corr)
        r2.append(corr2)
    average_r = float(np.mean(r))
    average_r2 = float(np.mean(r2))
    
    return average_r, average_r2

def plot_r(sess_r_values,sess,subjs,target,model,name):
    sess_list = list(sess_r_values.keys())
    r_values = list(sess_r_values.values())
    x = np.array(sess_list)  # X-axis 
    y = r_values # Y-axis

    plt.plot(x, y)  
    plt.xlabel('Number of training stories')
    plt.ylabel(f'Mean similarities of estimated weights({name})')
    directory=os.path.join(RESULTS_DATA_DIR,subjs,sess+target,model+'_model')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory,f'mean_similarities_weights_{name}.png'))
    np.save(os.path.join(directory,name+"-values.npy"), sess_r_values)
    plt.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  nargs='+', type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=int, required=True)
    parser.add_argument("--model", choices=['converter', 'converted','semantic'], required=True, help='Select model type.')
    parser.add_argument("--save_voxel", type=str, default='27a', help='Path to saved voxel selection.')
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    globals().update(args.__dict__)

    assert len(subject) <= 2 and len(subject) >=1, "1 <= subjects <= 2"
    sessions = list(map(str, sessions))
    subject = list(map(str, subject))
    sess = '_'.join(sessions)
    subjs = '_'.join(subject)
    dict_r_values ={}
    dict_r2_values ={}
    for j in sessions: 

        sess_r_values = []
        sess_r2_values = []
        weight={}
        weight2={}
        for i in ['a','b','c']:
            primal_coef_r,primal_coef_r2 =load_primal_coef(subjs,j+i,target,model,save_voxel)
            weight[f'{j}{i}']=primal_coef_r
            weight2[f'{j}{i}']=primal_coef_r2
        r_value,r2_value=compute_r(weight,weight2,j)
        sess_r_values.append(r_value)
        sess_r2_values.append(r2_value)
        print("r values of j : ",sess_r_values)
        dict_r_values[j]=sess_r_values
        dict_r2_values[j]=sess_r2_values
    plot_r(dict_r_values,sess,subjs,target,model,"r")
    plot_r(dict_r2_values,sess,subjs,target,model,"r2")
