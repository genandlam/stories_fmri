import os,json,cortex,argparse,logging,joblib
import numpy as np
from config.dir import DATA_DIR, EM_DATA_DIR,RESULTS_DATA_DIR,FEATURE_DATA_DIR
from sklearn.preprocessing import StandardScaler
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from voxelwise_tutorials.delayer import Delayer
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.utils import generate_leave_one_run_out
from sklearn.model_selection import check_cv
import matplotlib.pyplot as plt
from utils.npp import zscore
from himalaya.viz import plot_alphas_diagnostic
from sklearn.decomposition import PCA
from voxelwise_tutorials.wordnet import scale_to_rgb_cube
from utils.SemanticModel import SemanticModel
from voxelwise_tutorials.viz import plot_hist2d

def open_json(subj,sess,file,dir=FEATURE_DATA_DIR):
    print(os.path.join(dir,subj,sess,file))
    with open(os.path.join(dir,subj,sess,file), "r")  as f:
        data = json.load(f)
    print(f"Loaded {file} with shape: ", np.array(data).shape)
    return  np.array(data, dtype=float)

def open_npy(subj,sess,file,dir=RESULTS_DATA_DIR):
    data = np.load(os.path.join(dir,subj,sess,file))
    print(f"Loaded {file} with shape: ", data.shape)
    return data

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_file(file):
    if not os.path.isfile(file):
        return False
    else:
        return True
    
def save_predict(pipeline,X_train,X_test,dir,Y_No=True):

    train_predict=pipeline.predict(X_train)
    test_predict=pipeline.predict(X_test)

    if Y_No:
        print("train predict saving:", train_predict[:3])
        np.save(os.path.join(dir,'train_predict'), train_predict )
        np.save(os.path.join(dir,'test_predict'), test_predict )   
    
    else:
        return train_predict,test_predict
    
def check_mean_sf(X_train,X_test):

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    if np.mean(X_train) >0.0001 or np.std(X_train) >1.0001:
        print("train are not standardized properly.")
        X_train= zscore(np.array(X_train, dtype=float))

    if np.mean(X_test) >0.0001 or np.std(X_test) <1:
        print("Warning: test are not standardized properly.")
        X_test= zscore(np.array(X_test, dtype=float))
        print("Mean of new test :",np.mean(X_test))
        print("Standard deviation new test: ",np.std(X_test))

    return X_train,X_test

def semantic_features(subject,sess):
        
        X_train = open_json(subject,sess,'features_train.json')
        X_test = open_json(subject,sess,'features_test.json')
        print("(n_samples_train, n_features) =", X_train.shape)
        print("(n_samples_test, n_features) =", X_test.shape)
        X_train,X_test=check_mean_sf(X_train,X_test)

        return X_train,X_test


def select_voxels(subj,sess,save_voxel,threshold,model,target):

    if model == "converter":
        X_train = open_json(subj,sess,'fmri_train.json')
        X_test = open_npy(subj,save_voxel,'semantic_model/test_predict.npy',dir=RESULTS_DATA_DIR)

    elif model == "converter_same":
        X_train = open_json(subj,sess,'fmri_train.json')
        X_test = open_npy(subj,sess,'semantic_model/test_predict.npy',dir=RESULTS_DATA_DIR)

    elif model == "converted":
        pipeline,dir, _ = load_model(subj,save_voxel,target,"semantic")
        X_train,X_test = semantic_features(subj,sess)
        X_train,X_test = save_predict(pipeline,X_train,X_test,dir,Y_No=False)
        X_test = open_npy(subj,save_voxel,'semantic_model/test_predict.npy',dir=RESULTS_DATA_DIR)
        
    elif model == "converted_same":
        X_train = open_npy(subj,sess,'semantic_model/train_predict.npy',dir=RESULTS_DATA_DIR)
        X_test = open_npy(subj,save_voxel,'semantic_model/test_predict.npy',dir=RESULTS_DATA_DIR)

    X_train,X_test=check_mean_sf(X_train,X_test)

    if check_file(os.path.join(RESULTS_DATA_DIR,subj,save_voxel,'semantic_model/best_voxels.npy')) == False:
        scores_train = open_npy(subj,save_voxel,'semantic_model/scores_train.npy',dir=RESULTS_DATA_DIR)
        best_voxels = np.argsort(scores_train)[::-1][:threshold]
        np.save(os.path.join(RESULTS_DATA_DIR,subj,save_voxel,'semantic_model/best_voxels.npy'), best_voxels)
 
    else:
        best_voxels = open_npy(subj,save_voxel,'semantic_model/best_voxels.npy',dir=RESULTS_DATA_DIR)
    
    print("(n_samples_train, n_features) =", X_train[:, best_voxels].shape)
    print("(n_samples_test, n_features) =", X_test[:, best_voxels].shape)

    return X_train[:, best_voxels],X_test[:, best_voxels]

def create_run_on_set(subj,sess):

    run_onsets=open_json(subj,sess,'run_on.json')
    run_onsets=list(map(int, run_onsets))
    return run_onsets

def save_model(pipeline,subj,sess,target,model):

    file_name,directory=get_model_filename_dir(subj,sess,target,model)
    check_dir(directory)
    if check_file(os.path.join(directory,file_name)):
        file_name = subj+"_"+target+'_'+sess+model+'_model_1.pkl'
    joblib.dump(pipeline, os.path.join(directory,file_name), compress=True) 
    return directory

def train_model(subj,sess,X_train,Y_train,model):
    run_onsets= create_run_on_set(subj,sess)
    if len(run_onsets) > 1 : 
        n_samples_train = X_train.shape[0]
        cv = generate_leave_one_run_out(n_samples_train, run_onsets)
        cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list
    else:
        cv = None
        print(" 1 run only - defaulting to no cv")
    if model == "semantic":
        print("Using semantic features ...")
        delay= Delayer(delays=[1, 2, 3, 4])
    else: delay = None 
    print (delay)
    X_train= X_train.astype("float32")
    alphas = np.logspace(1, 20, 20)
    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)
    pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    delay,
    KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100)),
    )
    _ = pipeline.fit(X_train, Y_train)

    return pipeline,alphas,backend

def get_model_filename_dir(subj,sess,target,model):

    if model == 'semantic':
        file_name = subj+'_'+sess+str.capitalize(model)+'_model.pkl'
        directory=os.path.join(RESULTS_DATA_DIR,subj,sess,model+'_model')
    else:
        file_name = subj+"_"+target+'_'+sess+model+'_model.pkl'
        directory=os.path.join(RESULTS_DATA_DIR,subj,sess+'_'+target,model+'_model')
    return file_name,directory

def save_histogram(title,scores,dir):

    plt.clf()
    plt.hist(scores , bins=50, log=True)
    plt.title("Histogram of "+title+" R-squared values")
    plt.ylabel("Frequency")
    plt.xlabel("R-squared")
    plt.savefig(dir+'/'+title+'_histogram.png')

def save_scores(pipeline,backend,X_train,Y_train,X_test,Y_test,save_score=True):

    scores_train = backend.to_numpy(pipeline.score(X_train,Y_train))
    print("(n_voxels train,) =", scores_train.shape)
    scores_test = backend.to_numpy(pipeline.score(X_test, Y_test))
    print("(n_voxels test,) =", scores_test.shape)

    if save_score:
        print("score saving ...")
        np.save(os.path.join(dir,'scores_train'), scores_train)
        np.save(os.path.join(dir,'scores_test'), scores_test)

    return scores_test,scores_train

def save_rscore(dir,X_train,X_test):
    predict_train=np.load(os.path.join(dir,'train_predict.npy'))
    predict_test=np.load(os.path.join(dir,'test_predict.npy'))
    corr_train = np.corrcoef(X_train.ravel(), predict_train.ravel())[0, 1]
    corr_test = np.corrcoef(X_test.ravel(), predict_test.ravel())[0, 1]
    np.save(os.path.join(dir,'rscore_train'), corr_train )
    np.save(os.path.join(dir,'rscore_test'), corr_test ) 

def load_model(subj,sess,target,model):

    Backend = set_backend("torch_cuda", on_error="warn")
    print(Backend)
    file_name,directory=get_model_filename_dir(subj,sess,target,model)
    pipeline= joblib.load(os.path.join(directory,file_name)) 
    return pipeline,directory,Backend

def set_pycortex_store(filestore):
    """Set the pycortex store to the specified directory."""
    cortex.database.default_filestore = filestore
    cortex.db.filestore = filestore
    cortex.db.reload_subjects()
    print(f"pycortex store set to {filestore}")

def save_cortex(title,subj,scores,dir,model):
    set_pycortex_store(os.path.join(DATA_DIR, 'ds003020/derivative/pycortex-db'))
    subject = subj.split('-')[1]
    xfm = subject+'_auto'
    # First create example voxel data for this subject and transform
    voxel_data = scores 
    voxel_vol = cortex.Volume(voxel_data, subject, xfm,vmin=0,cmap="inferno")

    # Then we have to get a mapper from voxels to vertices for this transform
    mapper = cortex.get_mapper(subject, xfm, 'line_nearest', recache=True)

    # Just pass the voxel data through the mapper to get vertex data
    vertex_map = mapper(voxel_vol)

    # You can plot both as you would normally plot Volume and Vertex data
    cortex.quickshow(voxel_vol, with_rois=False)
    plt.savefig(os.path.join(dir,subj+title+'_'+model+'_model_voxel.png'))
    plt.clf()

def plot_alphas(backend,dir,alphas):

    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
    plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
    plt.savefig(os.path.join(dir,'alphas_diagnostic.png'))

def pca_com(average_coef):
    
    pca = PCA(n_components=4)
    pca.fit(average_coef.T)
    components = pca.components_
    print("(n_components, n_features) =", components.shape)
    print("PCA explained variance =", pca.explained_variance_ratio_)
    return components,pca

def print_voxel_words_pca(voxnum,components):
    # find_words_like_vec returns 10 words most correlated with the given vector, and the correlations
    eng1000 = SemanticModel.load(os.path.join(EM_DATA_DIR, "english1000sm.hf5"))
    voxwords = eng1000.find_words_like_vec(components[voxnum,:])
    print ("Best words for voxel %d :" % (voxnum))
    print(voxwords)

def compare_hist(dir,subjs,model,scores,target):
    dir1,_=get_model_filename_dir(target,sess,target,'semantic')
    scores_baseline= np.load(os.path.join(dir1,'scores_test'+'.npy'))
    ax = plot_hist2d(scores_baseline, scores,vmin=-0.2, vmax=0.4)
    ax.set(title='Generalization R2 scores', ylabel='Baseline model',
        xlabel='Transform model')
    plt.savefig(os.path.join(dir,subjs,model+'_model_hist_compare.png'))

def compare_heatmap(dir,subjs,model,scores,target):
    dir1,_=get_model_filename_dir(target,sess,target,'semantic')
    scores_baseline= np.load(os.path.join(dir1,'scores_test'+'.npy'))
    subject = 'UTS02'
    xfm = 'UTS02_auto'
    vol_data = cortex.Volume2D(scores_baseline, scores, subject, xfm,
                            #cmap="GreenWhiteBlue_2D",
                            vmin=0,vmin2=0, vmax=0.2, vmax2=0.2
                            )
    cortex.quickshow(vol_data, with_rois=False )# with_colorbar=False,
    plt.savefig(os.path.join(dir,subjs,model+'_model_voxel_compare.png'))

def get_primal_coef(scores_test,pipeline,target,dir,backend,model,subjs,sess):

    rscore_test=np.load(os.path.join(dir,'rscore_test.npy'))
    primal_coef = backend.to_numpy(pipeline[-1].get_primal_coef())
    primal_coef /= np.linalg.norm(primal_coef, axis=0)
    primal_coef_r2= np.copy(primal_coef)
    primal_coef_r= np.copy(primal_coef)
    primal_coef_r2 *= np.sqrt(np.maximum(0, scores_test ))
    primal_coef_r *= np.sqrt(np.maximum(0, rscore_test ))
    print("(n_features, n_voxels) =", primal_coef.shape)

    if model =='semantic':
        delayer = pipeline.named_steps['delayer']
        primal_coef_per_delay = delayer.reshape_by_delays(primal_coef_r2, axis=0)
        print("(n_delays, n_features, n_voxels) =", primal_coef_per_delay.shape)
        # average over delays
        average_coef = np.mean(primal_coef_per_delay, axis=0)
        print("(n_features, n_voxels) =", average_coef.shape)
        file_name= subjs+'_'+sess+model+'_primal_coef'
        primal_coef_r2= average_coef
    else:
        file_name = subjs+"_"+target+'_'+sess+model+'_primal_coef'
    np.save(os.path.join(dir,file_name+'_r'),primal_coef_r)
    np.save(os.path.join(dir,file_name+'_r2'),primal_coef_r2)

    return primal_coef_r2

def plot_RGB(scores_test,pipeline,target,dir,backend,model,subjs,sess):

    primal_coef=get_primal_coef(scores_test,pipeline,target,dir,backend,model,subjs,sess)
    set_pycortex_store(os.path.join(DATA_DIR, 'ds003020/derivative/pycortex-db'))
    # perform PCA on the voxel coefficients
    components,pca =pca_com(primal_coef)
    # transform with the fitted PCA
    average_coef_transformed = pca.transform(primal_coef.T).T
    print("(n_components, n_voxels) =", average_coef_transformed.shape)
    del primal_coef
    # We make sure vmin = -vmax, so that the colormap is centered on 0.
    vmax = np.percentile(np.abs(average_coef_transformed), 99.9)
    voxel_colors = scale_to_rgb_cube(average_coef_transformed[1:4].T, clip=3).T
    print("(n_channels, n_voxels) =", voxel_colors.shape)
    subject = target.split('-')[1]
    xfm = subject+'_auto'
    # Scaling the three datasets to be between 0-255
    test1_scaled = voxel_colors[0] / np.max(voxel_colors[0]) * 255
    test2_scaled = voxel_colors[1] / np.max(voxel_colors[1]) * 255
    test3_scaled = voxel_colors[2] / np.max(voxel_colors[2]) * 255
    red = cortex.Volume(test1_scaled.astype(np.uint8), subject, xfm)
    green = cortex.Volume(test2_scaled.astype(np.uint8),  subject, xfm)
    blue = cortex.Volume(test3_scaled.astype(np.uint8),  subject, xfm)
    vol_data = cortex.VolumeRGB(red, green, blue, subject,vmin=0, vmax=1, vmin2=0, vmax2=1, vmin3=0, vmax3=1)
    cortex.quickshow(vol_data, with_colorbar=False)
    plt.savefig(os.path.join(dir,subjs+'RGB_'+model+'_model_voxel.png'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  nargs='+', type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=str, required=True)
    parser.add_argument("--model", choices=['converter', 'converted','converted_same','semantic'], required=True, help='Select model type.')
    parser.add_argument("--savemodel", type=bool, default=False)
    parser.add_argument("--threshold", type=int, default=10000)
    parser.add_argument("--save_voxel", type=str, default='27a', help='Session name where to save/load best voxels.')
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    globals().update(args.__dict__)
    assert len(subject) <= 2 and len(subject) >=1, "1 <= subjects <= 2"

    sessions = list(map(str, sessions))
    subject = list(map(str, subject))
    sess = '_'.join(sessions)
    subjs = '_'.join(subject)
    
    Y_train = open_json(target,sess,'fmri_train.json') # (n_train_stories, n_voxels)
    Y_test = open_json(target,sess,'fmri_test.json') # (n_test_stories, n_voxels)
    Y_train,Y_test=check_mean_sf(Y_train,Y_test)
    
    if model == "semantic":

        print("Semantic model selected ...")
        X_train, X_test = semantic_features(subjs,sess)

    else:

        X_train = np.empty((0, threshold)) # (n_train_stories * n_subjects, n_voxels)
        X_test = np.empty((0, threshold)) # (n_test_stories * n_subjects, n_voxels)  
        y_train = np.empty((0, Y_train.shape[1]))
        y_test = np.empty((0, Y_test.shape[1]))
        
        for i in range (len(subject)): 
 
            X_train_best,X_test_best= select_voxels(subject[i],sess,save_voxel,threshold,model,target)
            X_train=np.vstack([X_train, X_train_best])
            X_test=np.vstack([X_test, X_test_best])
            y_train = np.vstack([y_train, Y_train])
            y_test = np.vstack([y_test, Y_test])
            print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

        Y_test = y_test
        Y_train = y_train

    if savemodel == True:

        pipeline,alphas,backend = train_model(subjs,sess,X_train,Y_train,model)
        dir = save_model(pipeline,subjs,sess,target,model)
        print(f"Model saved in {dir}")
        save_predict(pipeline,X_train ,X_test,dir)
        scores_test,scores_train=save_scores(pipeline,backend,X_train,Y_train,X_test,Y_test)
        save_rscore(dir,Y_train,Y_test)
        save_histogram("Train Data",scores_train,dir)
        save_histogram("Test Data",scores_test,dir)
        save_cortex("Train Data",target,scores_train,dir,model)
        save_cortex("Test Data",target,scores_test,dir,model)
        plot_alphas(backend,dir,alphas)
        plot_RGB(scores_test,pipeline,target,dir,backend,model,subjs,sess)

    elif check_file(os.path.join(get_model_filename_dir(subjs,sess,target,model)[1],get_model_filename_dir(subjs,sess,target,model)[0])):
        print("Loading existing model...")
        pipeline,dir,backend = load_model(subjs,sess,target,model)
        scores_test,scores_train=save_scores(pipeline,backend,X_train,Y_train,X_test,Y_test)
        save_rscore(dir,Y_train,Y_test)
        plot_RGB(scores_test,pipeline,target,dir,backend,model,subjs,sess)

    else:

        pipeline,alphas,backend  = train_model(subjs,sess,X_train,Y_train,model) 
        save_predict(pipeline,X_train ,X_test,dir,False)
        print(f'Trained Data shapes: X_train: {X_train.shape}, X_test: {X_test.shape}')
