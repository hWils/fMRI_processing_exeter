cg# -*- coding: utf-8 -*-

"""
Created on Thu Apr  1 11:19:17 2021

This script takes as input .json files, and carries out hyperalignment on them.
Returns hyperalignment as output, which consists of indivudals transformed matrices,
as well as transformation matrices. These should be saved as .json files.

@author: hlw69
"""


####### THIS WORKS, FIND OUT EXACTLY WHICH SCRIPTS IT IS CALLING AND SAVE TO GITHUB 
##### MAKE CHANGE IN NITT from decomp svd to linalg non gpu svd!!
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from nibabel import save
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.experimental_paradigm import  BlockParadigm # EventRelatedParadigm
from os import mkdir, path
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import sklearn
import glob
import nibabel
import os
from nltools.stats import align
#from stats_hw import align
import matplotlib.pyplot as plt
import warnings
import json
#from mvpa2.suite import *
import nibabel as nib
from nilearn.masking import apply_mask as nimask


scale_data = True
stimuli = 2 #this would be just the first two faces
features = 10000
num_subjects = 15
warnings.simplefilter('ignore')


def load_data(roi):
    filename = 'dicts/faces_perceive_roi_'+ str(roi) + '_D1.json'
    with open(filename) as fp:
        dict_faces_perception = json.load(fp)
    print("Data is loaded")
    return dict_faces_perception

def process_data(dict_faces_perception):
    all_data = []
    for subject in range(30, 59):
        if subject == 39:
            print("skip 39")
        else:
            print(subject)
            data = np.array(dict_faces_perception[str(subject)])
            data = data.astype(np.float32)
            if scale_data:
                scaled = sklearn.preprocessing.minmax_scale(data, feature_range=(0, 1), axis=0, copy=True)
                print(np.min(scaled), np.max(scaled))
                all_data.append(scaled)
            else:
                all_data.append(data)
    print("The length  of all_data represents subjects: ", len(all_data))
    print("The stimuli of one subject in all_data is:", all_data[0].shape[0])
    print("For each stimulus, num_features is: ", all_data[0].shape[1], all_data[0].dtype)
    return all_data

def do_hyperalign(num_subs, num_stim, num_feats):
    print("In this instance, there are ", num_subs, 'ppts, and ', num_stim, ' stimuli, with all stimuli')
    lst = [item[:num_stim, :] for item in all_data[:num_subs]]
    print(len(lst))
    print("data i processed into right format, wit h scaling, will begin hyperalignment")
    hyperalign = align(lst, method='procrustes')
    print("hyperalignment is complete")
    return hyperalign

def save_hyperaligned_data(hyper):
    with open(hyper_json_fn, 'wb') as f:
        np.save(f, np.asarray(hyper['transformed']))
  #  with open(hyper_json_fn, 'w') as fp:
      #  print("opened")
       # new= {}
     #   new['data'] = hyper['transformed']
    #    print(type(hyper))
     #   json.dump(hyper, fp)
        print("finished")
        #fp.close()
       # print("closed")



# npy save insteaad of json
# SOMETHING NOT RIGHT AS SHOULD BE DIFFERENT LENGTHS OF FEATURES DEPENDING ON ROI. Perhaps masks are identical?
r = ['18']
for roi in r:
    hyper_json_fn = 'C:/Users/hlw69/Documents/fMRI_Exeter/hyperaligned/hyper_fp_roi'+roi+'.npy'
    data_dict = load_data(roi)
    all_data = process_data(data_dict)
    hyper = do_hyperalign(num_subjects, stimuli, features)
    save_hyperaligned_data(hyper)

#print(hyper['transformed'].shape)

