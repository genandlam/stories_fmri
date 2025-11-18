import os,json,argparse,logging,itertools
import numpy as np
from config.dir import DATA_DIR, EM_DATA_DIR,RESULTS_DATA_DIR,FEATURE_DATA_DIR
import matplotlib.pyplot as plt
import sklearn.metrics


def load_primal_coef(subjs,sess,target,model):
    #if subjs == '2':
    #    file_name = subjs+"_"+target+'_'+sess+model+'_primal_coef.npy'
    file_name = subjs+"_"+target+'_'+sess+model+'_primal_coef.npy'
    directory=os.path.join(RESULTS_DATA_DIR,subjs,sess+'_'+target,model+'_model')
    print(f"Loading {file_name}.npy from {dir}")
    primal_coef=np.load(os.path.join(directory,file_name) )
    return primal_coef

def compute_r(weight, sess):
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
        if w1 is None or w2 is None:
            raise KeyError(f"Missing weights for keys: {key1} or {key2}")
        corr = np.corrcoef(w1.ravel(), w2.ravel())[0, 1]
        corr2= sklearn.metrics.r2_score(w1.ravel(), w2.ravel())
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
    #plt.grid(True)
    directory=os.path.join(RESULTS_DATA_DIR,subjs,sess+target,model+'_model')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory,f'mean_similarities_weights_{name}.png'))
    plt.close()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  nargs='+', type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=int, required=True)
    parser.add_argument("--model", choices=['converter', 'converted','semantic'], required=True, help='Select model type.')
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
        #primal_coef,dir=load_primal_coef(subjs,j,target,model)
        sess_r_values = []
        sess_r2_values = []
        weight={}
        for i in ['a','b','c']:
            primal_coef=load_primal_coef(subjs,j+i,target,model)
            weight[f'{j}{i}']=primal_coef
        r_value,r2_value=compute_r(weight,j)
        sess_r_values.append(r_value)
        sess_r2_values.append(r2_value)
        print("r values of j : ",sess_r_values)
        dict_r_values[j]=sess_r_values
        dict_r2_values[j]=sess_r2_values
    plot_r(dict_r_values,sess,subjs,target,model,"r")
    plot_r(dict_r2_values,sess,subjs,target,model,"r2")
