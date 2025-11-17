import os,json,h5py
import numpy as np
from config.dir import DATA_DIR, EM_DATA_DIR ,REPO_DIR
from utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from utils.dsutils import make_semantic_model, make_word_ds
from utils.npp import zscore
from utils.SemanticModel import SemanticModel
from utils.interpdata import lanczosinterp2D
import argparse
import logging


def get_story_wordseqs(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open( os.path.join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_word_ds(grids, trfiles)
	return wordseqs


def downsample_word_vectors(stories, word_vectors, wordseqs):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times, 
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

def get_eng1000_vectors(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(os.path.join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = sm.data
	return downsample_word_vectors(allstories, vectors, wordseqs)


def get_feature_space(feature, *args):
	_FEATURE_CONFIG = {
        "eng1000": get_eng1000_vectors,
    }
	return _FEATURE_CONFIG[feature](*args)


def convert_to_serializable(downsampled_feat):
    """Convert downsampled feature dictionary to a serializable format."""
    
    serializable_dict = downsampled_feat.tolist()

    return serializable_dict


def apply_zscore_and_hrf(stories, downsampled_feat, trim):
	"""Get (z-scored and delayed) stimulus for train and test stories.
	The stimulus matrix is delayed (typically by 2,4,6,8 secs) to estimate the
	hemodynamic response function with a Finite Impulse Response model.

	Args:
		stories: List of stimuli stories.

	Variables:
		downsampled_feat (dict): Downsampled feature vectors for all stories.
		trim: Trim downsampled stimulus matrix.
		delays: List of delays for Finite Impulse Response (FIR) model.

	Returns:
		delstim: <float32>[TRs, features * ndelays]
	"""
	stim = [zscore(downsampled_feat[s][5+trim:-trim]) for s in stories]
	stim = np.vstack(stim)

	return stim

def get_response(stories, subject,run_on_set=None,remove=None):

	"""Get the subject"s fMRI response for stories."""
	
	subject_x = subject.split('-')[1]
	subject_dir = os.path.join(DATA_DIR, "ds003020/derivative/preprocessed_data/%s" % subject_x)
	base = subject_dir
	resp = []
	if run_on_set is None:
		run_on_set = [0]
	for story in stories:
		resp_path = os.path.join(base, "%s.hf5" % story)
		hf = h5py.File(resp_path, "r")
		resp.extend(hf["data"][:])
		if not run_on_set:
			run_on_set.append(hf["data"][:].shape[0])
		else:
			run_on_set.append(run_on_set[-1]+hf["data"][:].shape[0])
		#print(hf["data"][:].shape[0], "for story:", story)
		hf.close()
	return np.array(resp), run_on_set[:-1] if remove is None else run_on_set

if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", nargs='+', type=str, required=True)
	parser.add_argument("--trim", type=int, default=5)
	parser.add_argument("--feature", type=str, default="eng1000")
	parser.add_argument("--sessions", nargs='+', type=str, required=True)
	parser.add_argument("--stories", type=int, default=1)
	logging.basicConfig(level=logging.INFO)

	args = parser.parse_args()
	globals().update(args.__dict__)
	assert len(subject) <= 2 and len(subject) >=1, "1 <= subjects <= 2"
	subject = list(map(str, subject))
	sessions = list(map(str, sessions))
	s = '_'.join(sessions)
	if len(subject) > 1:
		remove = 1
		subjects = '_'.join(subject)
		save_location = os.path.join(REPO_DIR, "feature",feature, subjects,s)
	else:
		remove = None
		save_location = os.path.join(REPO_DIR, "feature",feature, subject[0],s)

	os.makedirs(save_location, exist_ok=True)

	with open(os.path.join(EM_DATA_DIR, f"sess_{stories}.json"), "r") as f:
		sess_to_story = json.load(f)

	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory)
	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"
	allstories = list(set(train_stories) | set(test_stories))

	downsampled_feat = get_feature_space(feature, allstories)
	delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat, trim)
	delTest = apply_zscore_and_hrf(test_stories, downsampled_feat, trim)
	print('Stimulus trainset:',delRstim.shape)
	print('Stimulus testset:',delTest.shape)

	# Response
	if remove is not None:
		zRresp,run_on_set = get_response(train_stories, subject[0],remove=remove)
		zRresp_2,run_on_set = get_response(train_stories, subject[1],run_on_set,remove=remove)
		run_on_set = run_on_set[:-1]
	else:
		zRresp,run_on_set = get_response(train_stories, subject[0])
	print("zRresp: ", zRresp.shape)
	print("length of run_on_set:", len(run_on_set), "with values:", run_on_set)

	test_resp,_ = get_response(test_stories, subject[0])
	print("zRresp Test: ", test_resp.shape)

	print("Saving features to:", save_location)

	with open(save_location+'/run_on.json', "w") as file:
		json.dump(run_on_set,file, indent=4)

	with open(save_location+'/fmri_train.json', "w") as file:
		json.dump(convert_to_serializable(zRresp),file, indent=4)
	with open(save_location+'/features_train.json', "w") as file:
		json.dump(convert_to_serializable(delRstim),file, indent=4)

	with open(save_location+'/features_test.json', "w") as file:
		json.dump(convert_to_serializable(delTest),file, indent=4)
	with open(save_location+'/fmri_test.json', "w") as file:
		json.dump(convert_to_serializable(test_resp),file, indent=4)



