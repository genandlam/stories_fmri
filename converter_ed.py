import os,json,cortex,argparse,logging,joblib
import numpy as np
from config.dir import DATA_DIR, EM_DATA_DIR,RESULTS_DATA_DIR,FEATURE_DATA_DIR
from sklearn.preprocessing import StandardScaler
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.backend import set_backend
from sklearn.pipeline import make_pipeline
from voxelwise_tutorials.utils import generate_leave_one_run_out
from sklearn.model_selection import check_cv
import matplotlib.pyplot as plt
from utils.npp import zscore
from himalaya.viz import plot_alphas_diagnostic
from sklearn.decomposition import PCA
from voxelwise_tutorials.wordnet import scale_to_rgb_cube
from utils.SemanticModel import SemanticModel


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

def select_voxels(subj,sess,threshold,model):

    if model == "converter":
        X_train = open_json(subj,sess,'fmri_train.json')
    elif model == "converted":
        X_train = open_npy(subj,sess,'semantic_model/train_predict.npy',dir=RESULTS_DATA_DIR)

    X_test = open_npy(subj,sess,'semantic_model/test_predict.npy',dir=RESULTS_DATA_DIR)
    X_train,X_test=check_mean_sf(X_train,X_test)

    scores_train = open_npy(subj,sess,'semantic_model/scores_train.npy',dir=RESULTS_DATA_DIR)
    best_voxels = np.argsort(scores_train)[::-1][:threshold]

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

def train_model(subj,sess,X_train,Y_train,X_test,Y_test):
    run_onsets= create_run_on_set(subj,sess)
    if len(run_onsets) > 1 : 
        n_samples_train = X_train.shape[0]
        cv = generate_leave_one_run_out(n_samples_train, run_onsets)
        cv = check_cv(cv)  # copy the cross-validation splitter into a reusable list
    else:
        cv = None
        print(" 1 run only - defaulting to no cv")

    X_train= X_train.astype("float32")
    alphas = np.logspace(1, 20, 20)
    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)
    pipeline = make_pipeline(
    StandardScaler(with_mean=True, with_std=False),
    KernelRidgeCV(
        alphas=alphas, cv=cv,
        solver_params=dict(n_targets_batch=500, n_alphas_batch=5,
                           n_targets_batch_refit=100)),
    )
    _ = pipeline.fit(X_train, Y_train)

    scores_train = pipeline.score(X_train,Y_train)
    scores_train = backend.to_numpy(scores_train)
    print("(n_voxels train,) =", scores_train.shape)
    scores_test = pipeline.score(X_test, Y_test)
    scores_test = backend.to_numpy(scores_test)
    print("(n_voxels test,) =", scores_test.shape)

    return pipeline,scores_train,scores_test,alphas,backend

def get_model_filename_dir(subj,sess,target,model):
    file_name = subj+"_"+target+'_'+sess+model+'_model.pkl'
    directory=os.path.join(RESULTS_DATA_DIR,subj,sess+'_'+target,model+'_model')
    return file_name,directory

def save_histogram(title,scores,dir):

    plt.clf()
    plt.hist(scores , bins=50, log=True)
    plt.title("Histogram of "+title+" R-squared values")
    plt.ylabel("Frequency")
    plt.xlabel("R-squared")
    plt.show()
    plt.savefig(dir+'/'+title+'_histogram.png')

def save_scores(scores_train,scores_test,dir):
    print("score saving:", scores_train)
    np.save(os.path.join(dir,'scores_train'), scores_train)
    np.save(os.path.join(dir,'scores_test'), scores_test )
    
def save_predict(pipeline,X_train,X_test,dir):

    train_predict=pipeline.predict(X_train)
    test_predict=pipeline.predict(X_test)
    np.save(os.path.join(dir,'train_predict'), train_predict )
    np.save(os.path.join(dir,'test_predict'), test_predict )    

def load_model(subj,sess,target,model):

    Backend = set_backend("torch_cuda", on_error="warn")
    print(Backend)
    file_name = subj+"_"+target+'_'+sess+model+'_model.pkl'
    directory=os.path.join(RESULTS_DATA_DIR,subj,sess+'_'+target,model+'_model')
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
    plt.show()
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

def get_primal_coef(scores_test,pipeline,target,dir,backend,model,subjs,sess):

    primal_coef = pipeline[-1].get_primal_coef()
    primal_coef = backend.to_numpy(primal_coef)
    primal_coef /= np.linalg.norm(primal_coef, axis=0)
    primal_coef *= np.sqrt(np.maximum(0, scores_test ))
    print("(n_features, n_voxels) =", primal_coef.shape)
    file_name = subjs+"_"+target+'_'+sess+model+'_primal_coef'
    np.save(os.path.join(dir,file_name),primal_coef)

    return primal_coef

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
    plt.show()
    plt.savefig(os.path.join(dir,subjs+'RGB_'+model+'_model_voxel.png'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  nargs='+', type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--sessions", nargs='+', type=str, required=True)
    parser.add_argument("--model", choices=['converter', 'converted'], required=True, help='Select model type.')
    parser.add_argument("--savemodel", type=bool, default=False)
    parser.add_argument("--threshold", type=int, default=1000)
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    globals().update(args.__dict__)

    assert len(subject) <= 2 and len(subject) >=1, "1 <= subjects <= 2"
    sessions = list(map(str, sessions))
    subject = list(map(str, subject))
    sess = '_'.join(sessions)
    subjs = '_'.join(subject)

    X_train_best,X_test_best= select_voxels(subject[0],sess,threshold,model)
    X_train_best_2,X_test_best_2= select_voxels(subject[1],sess,threshold,model)

    X_train_concat = np.concatenate([X_train_best, X_train_best_2], axis=0)
    X_test_concat = np.concatenate([X_test_best, X_test_best_2], axis=0)
    
    Y_train = open_json(target,sess,'fmri_train.json')
    Y_test = open_json(target,sess,'fmri_test.json')
    Y_train,Y_test=check_mean_sf(Y_train,Y_test)
    Y_train_concat = np.concatenate([Y_train, Y_train], axis=0)
    Y_test_concat = np.concatenate([Y_test, Y_test], axis=0)

    if savemodel == True:

        pipeline,scores_train,scores_test,alphas,backend = train_model(subjs,sess,X_train_concat,Y_train_concat,X_test_concat,Y_test_concat)
        dir = save_model(pipeline,subjs,sess,target,model)
        print(f"Model saved in {dir}")
        save_predict(pipeline,X_train_concat,X_test_concat,dir)
        save_scores(scores_train,scores_test,dir)
        save_histogram("Train Data",scores_train,dir)
        save_histogram("Test Data",scores_test,dir)
        save_cortex("Train Data",target,scores_train,dir,model)
        save_cortex("Test Data",target,scores_test,dir,model)
        plot_alphas(backend,dir,alphas)
        plot_RGB(scores_test,pipeline,target,dir,backend,model,subjs,sess)

    elif check_file(os.path.join(get_model_filename_dir(subjs,sess,target,model)[1],get_model_filename_dir(subjs,sess,target,model)[0])):
        print("Loading existing model...")
        pipeline,dir,backend = load_model(subjs,sess,target,model)
        scores_train = pipeline.score(X_train_concat,Y_train_concat)
        print("(n_voxels train,) =", scores_train.shape)
        scores_test = pipeline.score(X_test_concat,Y_test_concat)
        print("(n_voxels test,) =", scores_test.shape)
        scores_test = backend.to_numpy(scores_test)
        scores_train = backend.to_numpy(scores_train)

        save_cortex("Train Data",target,scores_train,dir,model)
        save_cortex("Test Data",target,scores_test,dir,model)
        plot_RGB(scores_test,pipeline,target,dir,backend,model,subjs,sess)

    else:

        pipeline,scores_train,scores_test,alphas,backend  = train_model(subjs,sess,X_train_concat,Y_train_concat,X_test_concat,Y_test_concat)
        print(pipeline.predict(X_train_concat))
        print(pipeline.predict(X_test_concat))

