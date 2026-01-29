import os,argparse,logging,itertools
import numpy as np
from config.dir import DATA_DIR, EM_DATA_DIR,RESULTS_DATA_DIR,FEATURE_DATA_DIR
import matplotlib.pyplot as plt
import sklearn.metrics

def check_file(file):
    if not os.path.isfile(file):
        return False
    else:
        return True
    
def open_npy(subj,sess,file,dir=RESULTS_DATA_DIR):
    data = np.load(os.path.join(dir,subj,sess,file))
    print(f"Loaded {file} with shape: ", data.shape)
    return data

def load_primal_coef(subjs,sess,target,model,save_voxel,threshold):

    if subjs == 'sub-UTS02':
        file_name = subjs+'_'+sess+model+'_primal_coef'
        primal_coef_r=open_npy(subjs,sess,model+'_model/'+file_name+'_r.npy')
        primal_coef_r2= open_npy(subjs,sess,model+'_model/'+file_name+'_r2.npy')
   
        # using the save_voxel best voxel selection for UTS02
        scores_train = open_npy(subjs,save_voxel,model+'_model'+'/scores_train.npy')
        best_voxels = np.argsort(scores_train)[::-1][:threshold]

    else: 

        file_name = subjs+"_"+target+'_'+sess+model+'_primal_coef'
        primal_coef_r=open_npy(subjs,sess+'_'+target,model+'_model/'+file_name+'_r.npy')
        primal_coef_r2=open_npy(subjs,sess+'_'+target,model+'_model/'+file_name+'_r2.npy')

        if check_file(os.path.join(RESULTS_DATA_DIR,subjs,save_voxel+'_'+target,model+'_model/best_voxels.npy')) == False:
            scores_train = open_npy(subjs,save_voxel+'_'+target,model+'_model/scores_train.npy')
            best_voxels = np.argsort(scores_train)[::-1][:threshold]
            np.save(os.path.join(RESULTS_DATA_DIR,subjs,save_voxel+'_'+target,model+'_model/best_voxels.npy'), best_voxels)
        else:
            best_voxels = open_npy(subjs,save_voxel+'_'+target,model+'_model/best_voxels.npy')

    return primal_coef_r2[:, best_voxels], primal_coef_r[:, best_voxels]


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
    return float(np.mean(r)), float(np.mean(r2)), float(np.round(np.std(r), 2)), float(np.round(np.std(r2),2))

def plot_r(sess_r_values,var,sess,subjs,target,model,name):

    x = np.array(list(sess_r_values.keys()))  # X-axis 
    y = np.array(list(sess_r_values.values())).flatten() # Y-axis
    vars_r_values = np.array(list(var.values())).flatten()
    print("x values: ",x)
    print("y values: ",y)
    print("var values: ",vars_r_values)
    plt.plot(x, y)  
    plt.errorbar(x, y, yerr = vars_r_values,
             fmt ='o', capsize=5)
    plt.xlabel('Number of training stories')
    plt.ylabel(f'Mean similarities of estimated weights({name})')
    directory=os.path.join(RESULTS_DATA_DIR,subjs,sess+target,model+'_model')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory,f'mean_similarities_weights_{name}_error.png'))
    np.save(os.path.join(directory,name+"-values.npy"), sess_r_values)
    np.save(os.path.join(directory,name+"-values(std).npy"), var)
    plt.close()

def plot_r2(sess,subjs,target,model,name):

    directory=os.path.join(RESULTS_DATA_DIR,subjs[0],sess+target,model+'_model')
    for subj in subjs:
        file = f'{model}_model/{name}-values.npy'
        r_value = np.load(os.path.join(RESULTS_DATA_DIR,subj,sess+target,file), allow_pickle=True, encoding='latin1').tolist()
        std_file = f'{model}_model/{name}-values(std).npy'
        std_value = np.load(os.path.join(RESULTS_DATA_DIR,subj,sess+target,std_file), allow_pickle=True, encoding='latin1').tolist()
        print("r values of ",subj,": ",r_value)
        x = np.array(list(r_value.keys()))  # X-axis 
        y = np.array(list(r_value.values())).flatten() # Y-axis
        vars_r_values = np.array(list(std_value.values())).flatten()
        plt.plot(x, y, label = subj)
        plt.errorbar(x, y, yerr = vars_r_values,
             fmt ='o', capsize=5)

    plt.legend()
    plt.xlabel('Number of training stories')
    plt.ylabel(f'Mean similarities of estimated weights({name})')
    plt.savefig(os.path.join(directory,f'compare_mean_similarities_weights_{name}_error.png'))
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  nargs='+', type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=int, required=True)
    parser.add_argument("--model", choices=['converter','converted', 'semantic'], required=True, help='Select model type.')
    parser.add_argument("--save_voxel", type=str, default='7a', help='Path to saved voxel selection.')
    parser.add_argument("--threshold", type=int, default=10000)
    parser.add_argument("--mode", choices=['single', 'multi'], required=True, help='Select mode single/multi .')
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    globals().update(args.__dict__)

    assert len(subject) <= 3 and len(subject) >=1, "1 <= subjects <= 3"
    sessions = list(map(str, sessions))
    subject = list(map(str, subject))
    sess = '_'.join(sessions)
    if mode == 'single':
        subjs = '_'.join(subject)
    else:
        subjs = subject
    if mode == 'multi':
        
        plot_r2(sess,subjs,target,model,name="r")
        plot_r2(sess,subjs,target,model,name="r2")
        exit()

    dict_r_values ={}
    dict_r2_values ={}
    dict_vars_r_values ={}
    dict_vars_r2_values ={}
    for j in sessions: 

        sess_r_values = []
        sess_r2_values = []
        vars_r_values = []
        vars_r2_values = []
        weight={}
        weight2={}
        for i in ['a','b','c']:
            primal_coef_r,primal_coef_r2 = load_primal_coef(subjs,j+i,target,model,save_voxel,threshold)
            weight[f'{j}{i}']=primal_coef_r
            weight2[f'{j}{i}']=primal_coef_r2
        r_value,r2_value,r_vars_value,r2_vars_value=compute_r(weight,weight2,j)
        sess_r_values.append(r_value)
        sess_r2_values.append(r2_value)
        vars_r_values.append(r_vars_value)
        vars_r2_values.append(r2_vars_value)
        print("r values of j : ",sess_r_values)
        dict_r_values[j]=sess_r_values
        dict_r2_values[j]=sess_r2_values
        dict_vars_r_values[j]=vars_r_values
        dict_vars_r2_values[j]=vars_r2_values

    plot_r(dict_r_values,dict_vars_r_values,sess,subjs,target,model,"r")
    plot_r(dict_r2_values,dict_vars_r2_values,sess,subjs,target,model,"r2")
