"""Extract contour features from a directory of contour files"""

import numpy as np
import pandas as pd
import pickle
import os

from motif.feature_extractors import bitteli
from motif.core import CONTOUR_EXTRACTOR_REGISTRY
from motif.core import FEATURE_EXTRACTOR_REGISTRY


def load_contour_filelist(input_dir):
    '''Load list of contour csv files from a directory.
    
    Parameters
    ----------
    input_dir : str
        The path to a directory containing contour csv files. 

    Returns
    -------
    filelist : list
        Contour files to be processed. 
    '''    
    filelist = []
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            filelist.append(os.path.join(input_dir, file))
    return filelist


def get_contours(filename):
    '''Read contour data from csv and return the time, pitch, salience estimates.
    
    Parameters
    ----------
    filename : str
        The path to a csv file with contour data.

    Returns
    -------
    time : list
        Time stamps for each contour stored as a list of numpy arrays.
    pitch : list
        Pitch estimates for each contour stored as a list of numpy arrays.
    salience : list
        Salience estimates for each contour stored as a list of numpy arrays.
    '''
    d = pd.read_csv(filename, header=None)
    contour_index = d.iloc[:, 0].get_values()
    contour_time_fr = d.iloc[:, 1].get_values()
    contour_pitch_f0 = d.iloc[:, 2].get_values()
    contour_salience = d.iloc[:, 3].get_values()

    pitch = []
    time = []
    salience = []
    for index in np.unique(contour_index):
        inds = np.where(contour_index==index)[0]
        pitch.append(contour_pitch_f0[inds])
        time.append(contour_time_fr[inds])
        salience.append(contour_salience[inds])
    return time, pitch, salience


def extract_features_for_contour_filelist(filelist):
    '''Extract contour features for a list of contour files.
    
    Parameters
    ----------
    filelist : list
        Contour files to be processed.
    
    Returns
    -------
    contour_features_list : list
        Contour features for each contour. 
    contour_files_list : list
        File name for each processed contour. 
    '''   
    # load contours and compute features
    sample_rate = 128.
    bitli = bitteli.BitteliFeatures()
    n_files = len(filelist)
    
    contour_files_list = []
    contour_features_list = []
    for i in range(n_files):
        if not os.path.exists(filelist[i]):
            continue
        with open(filelist[i],'r') as f:
            lines = f.readlines()
        if len(lines)<1:  # if file is empty
            continue
        print str(i+1) + " of " + str(n_files)+str(filelist[i])
        time, pitch, salience = get_contours(filelist[i])
        n_contours = len(pitch)
        for j in range(n_contours):
            if len(pitch[j])>1:
                pass
            try:
                features = bitli.get_feature_vector(time[j], pitch[j], salience[j], sample_rate)
            except:
                continue
            contour_features_list.append(features[:,None].T)
            contour_files_list.append(os.path.basename(filelist[i]))
    
    return contour_features_list, contour_files_list


def features_from_contours(input_dir, pickle_file=None):
    '''Save contour data to pickle file.
    
    Parameters
    ----------
    pickle_file : str
        The path to a pickle file to save the contour data. 
    '''  
    filelist = load_contour_filelist(input_dir)
    features_list, files_list = extract_features_for_contour_filelist(filelist)
    contour_features = np.concatenate(features_list)
    contour_files = np.array(files_list)
    
    if pickle_file is not None:
        pickle.dump([contour_features, contour_files], open(pickle_file, 'wb'))


def load_audio_filelist(input_dir):
    '''Load list of contour csv files from a directory.
    
    Parameters
    ----------
    input_dir : str
        The path to a directory containing contour csv files. 

    Returns
    -------
    filelist : list
        Contour files to be processed. 
    '''    
    filelist = []
    for file in os.listdir(input_dir):
        if file.endswith(".mp3") or file.endswith(".wav"):
            filelist.append(os.path.join(input_dir, file))

    return filelist
    

def extract_features_for_audio_filelist(audio_files):
    '''Extract contour features for a list of contour files.
    
    Parameters
    ----------
    filelist : list
        Contour files to be processed.
    
    Returns
    -------
    contour_features_list : list
        Contour features for each contour. 
    contour_files_list : list
        File name for each processed contour. 
    '''    
    contour_extractor = CONTOUR_EXTRACTOR_REGISTRY['salamon']()
    feature_extractor = FEATURE_EXTRACTOR_REGISTRY['bitteli']()
    
    features_list = []
    files_list = []
    for audio_filepath in audio_files:
        ctr = contour_extractor.compute_contours(audio_filepath)
        X = feature_extractor.compute_all(ctr)
        features_list.append(X)
        file_name = os.path.basename(ctr.audio_filepath)
        files_list.append(np.repeat(file_name, len(X)))
    
    return features_list, files_list


def features_from_audio(input_dir, pickle_file=None):
    '''Save contour data to pickle file.
    
    Parameters
    ----------
    pickle_file : str
        The path to a pickle file to save the contour data. 
    '''  
    filelist = load_audio_filelist(input_dir)    

    features_list, files_list = extract_features_for_audio_filelist(filelist)
    contour_features = np.concatenate(features_list)
    contour_files = np.array(files_list)
    
    if pickle_file is not None:
        pickle.dump([contour_features, contour_files], open(pickle_file, 'wb'))

  
if __name__ == '__main__':
    input_dir = '../data/VocalContours'
    #pickle_file = '../data/contour_data.pickle'
    pickle_file = None
    
    features_from_contours(input_dir, pickle_file)