# processing fMRI data
# find path to fmri files
# find path to SPM.MAT files


#changed mask in fmrilinear to None
# stopped mask being applied in applymask function


# THIS CURRENTLY WORKS FOR FACILITATING HYPERALIGNMENT WITHIN CONDITIONS, so classifying withn faces, or within places
# This needs to be altered to be able to classify between faces and places

import scipy.io as spio
from nibabel import save
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm  # EventRelatedParadigm
from os import mkdir, path
from itertools import chain
import glob
import nibabel
import matplotlib.pyplot as plt
#from mvpa2.suite import *
import numpy as np
import nibabel as nib
import json
import os
#from jsonEncoder import NumpyEncoder
n_scans = 330  # dynamics
tr = 3 # sampling rate
last_stim = 58
first_stim = 30
stim_num = 36
subject = 34
condition = 'imagine'
exclude_subjects = [39]
task = 'faces'
task_conditions = [type]
roi = '20_37'
dilation = '1'
#fn = 'C:/Users/hlw69/Documents/fMRI_Exeter/dicts/'+task+'_perceive_all_roi_ppt30_59_dumps.json'
#fn = 'C:/Users/hlw69/Documents/fMRI_Exeter/dicts/faces_perceive_roi_' + roi+'_D'+dilation+'.json'  #the json file needs to be created first in dicts
maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//bounded'+roi+'_d1.nii'  # rdilation_one_3D.nii


# submethod of extracting the perception onsets and durations
def block_to_stimuli_onset(block_ons, dur, stim_num=36):
    perception_ons = np.zeros(stim_num)
    perception_duration = 7 * (np.ones(stim_num))
    i = 0
    for block in block_ons:
        for stim in range(4):
            if stim == 0:
                perception_ons[i] = block
            else:
                perception_ons[i] = block + (dur * stim)
            i += 1
    return perception_ons, perception_duration


# create the correct filepath for retreving the 4D fmri file for a particular subject and condition
def condition_type_filepath(condition, subject, types):
    main_filepath = 'C://Users//hlw69//Documents//temp_fmri_exeter//Imagine_' + subject
    if condition == 'faces':
        filepath = main_filepath + '//Faces//'
        if types == 'spm':
            filepath = filepath + 'SPM.mat'
        elif types == '4D':
            filepath = filepath + 'Segmented//Motion_corrected//4D.nii'
    elif condition == 'places':
        filepath = main_filepath + '//Places//'
        if types == 'spm':
            filepath = filepath + 'SPM.mat'
        elif types == '4D':
            filepath = filepath + 'Segmented//Motion_corrected//4D.nii'
            print("4d filepath is ", filepath)
    return filepath


# given an spm file path, extract the onset and duration times for each part of the experiment, including control conditions
def getOnsetDurationDurImgPercSPM(condition_filepath, stim_num=36):
    mat = spio.loadmat(condition_filepath, squeeze_me=True)
    spm = mat['SPM'][()]

    imagination = spm['Sess']['U'].item()['ons'][0]
    imagination_dur = spm['Sess']['U'].item()['dur'][0]

    name = spm['Sess']['U'].item()['ons'][1]
    name_dur = spm['Sess']['U'].item()['dur'][1]

    look_text = spm['Sess']['U'].item()['ons'][2]
    look_text_dur = spm['Sess']['U'].item()['dur'][2]

    imagine_text = spm['Sess']['U'].item()['ons'][4]
    imagine_text_dur = spm['Sess']['U'].item()['dur'][4]

    control_imagine = spm['Sess']['U'].item()['ons'][5]
    control_imagine_dur = spm['Sess']['U'].item()['dur'][5]

    control_name = spm['Sess']['U'].item()['ons'][6]
    control_name_dur = spm['Sess']['U'].item()['dur'][6]

    look_control_text = spm['Sess']['U'].item()['ons'][7]
    look_control_text_dur = spm['Sess']['U'].item()['dur'][7]

    control_imagine_text = spm['Sess']['U'].item()['ons'][9]
    control_imagine_text_dur = spm['Sess']['U'].item()['dur'][9]

    block_perception_ons = spm['Sess']['U'].item()['ons'][3]
    control_block_perception_ons = spm['Sess']['U'].item()['ons'][8]
    perceive, perceive_dur = block_to_stimuli_onset(block_perception_ons, dur=7)
    control_perceive, control_perceive_dur = block_to_stimuli_onset(control_block_perception_ons, dur=7)

    conditions = [['imagination'] * stim_num, ['name'] * stim_num, ['look_text'] * 9, ['perceive'] * stim_num,
                  ['imagine_text'] * 9, ['control_imagine'] * stim_num, ['control_name'] * stim_num,
                  ['look_control_text'] * 9, ['control_perceive'] * stim_num, ['control_imagine_text'] * 9]

    onsets = [imagination, name, look_text, perceive,
              imagine_text, control_imagine, control_name, look_control_text, control_perceive, control_imagine_text]

    durations = [imagination_dur, name_dur, look_text_dur, perceive_dur,
                 imagine_text_dur, control_imagine_dur, control_name_dur,
                 look_control_text_dur, control_perceive_dur, control_imagine_text_dur]
    return conditions, onsets, durations


# for exemplar level, find onsets and durations given a particular filepath and conditions
def getOnsetExemplarLevelM(condition_filepath, task, stim_num=36):
    # get imagination onset and duration
    mat = spio.loadmat(condition_filepath, squeeze_me=True)
    spm = mat['SPM'][()]
    exemplars = np.arange(1, 37)
    exemplar_condition_names = []
    exemplar_onsets = []
    exemplar_durations = []

    if task == 'imagine':
        onset = spm['Sess']['U'].item()['ons'][0]
        duration = spm['Sess']['U'].item()['dur'][0]
    elif task == 'perceive':
        block_perception_ons = spm['Sess']['U'].item()['ons'][3]
        onset, duration = block_to_stimuli_onset(block_perception_ons, dur=7)

    for i in range(36):
        name = task + '_' + str(exemplars[i])
        exemplar_onsets.append(onset[i])
        exemplar_durations.append(duration[i])
        exemplar_condition_names.append(name)
        print("onsets, durations: ", exemplar_onsets, " ", exemplar_durations)

    return exemplar_condition_names, exemplar_onsets, exemplar_durations


# Input = condition, fMRI .nii files, spm .mat files. Output = designmatrix.png, GLM model, contrast images .nii
# Takes in the .nii files for each dynamic at the subject level, extracts the timings/condition from the .mat file
#  creates and saves a png of the design matrix. Then obtains the 4D fMRI data, and fits a GLM
# defines contrasts, then carries out contrasts, which produces a contrast .nii file and saves

def first_level_analysis(subject, faceOrplace, stim_num='36'):
    if faceOrplace != 'faces' and faceOrplace != 'places':
        print("You need to enter either Faces or Places as a parameter to this function")
    else:
        print("Conducting single_level_analysis for " + faceOrplace + "condition, for subject number " + subject)
    # get filepaths for faces and places
    filepath = condition_type_filepath(faceOrplace, subject, types='spm')
    print(filepath)

    # onsets and durations for both places and faces for a given subject
    condition_list, onset, duration = getOnsetDurationDurImgPercSPM(filepath)  ##########

    # flatten lists # WHY ARE WE FLATTENING?!
    flattened_conditions = list(chain.from_iterable(condition_list))
    flattened_onsets = list(chain.from_iterable(onset))
    flattened_durations = list(chain.from_iterable(duration))
    print("These two numbers should be equal, if not there is an error ", len(flattened_conditions),
          len(flattened_onsets))
    paradigm = BlockParadigm(con_id=flattened_conditions, onset=flattened_onsets, duration=flattened_durations)

    n_scans = 330  # dynamics
    tr = 3

    frametimes = np.arange(0, n_scans * tr, tr)
    hrf_model = 'canonical'
    drift_model = "cosine"
    hfcut = 128  # what does this refer to? bandpass
    design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model, hfcut=hfcut)
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    resultdir = 'C://Users//hlw69//Documents//fMRI_Exeter//masked_results//'
    filename = subject + "_" + faceOrplace
    write_dir = path.join(resultdir, filename)
    if not path.exists(write_dir):
        mkdir(write_dir)
    plt.savefig(path.join(write_dir, 'design_matrix.png'))

    # obtain the neural data, this will be in a 4D format nii.gz
    spm4Dfilepath = condition_type_filepath(faceOrplace, subject, types='4D')

    img_4d = nib.load(spm4Dfilepath)
    print("Shape of img_4d is ", img_4d.shape)
    # maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//bounded17_18_19_dilated_1.nii' #r_testfirstAttemptDilated.nii'
    roi_mask = nib.load(maskDirectory)
    masked_data = nimask(img_4d,roi_mask) # this returns a 2D array

    ##  Create the fmri linear model
    fmri_glm = FMRILinearModel(img_4d, design_matrix.matrix,
                               mask='compute')
    fmri_glm.fit(do_scaling=True, model='ar1')

    contrasts = {}
    contrasts = {
        "imagination": np.zeros(25),
        "perception": np.zeros(25),
        "Perc_Imag": np.zeros(25), 'Imag_Perc': np.zeros(25)}
    contrasts['imagination'][4] = 1
    contrasts['perception'][9] = 1
    contrasts['Perc_Imag'][9] = 1
    contrasts['Perc_Imag'][4] = -1
    contrasts['Imag_Perc'][4] = 1
    contrasts['Imag_Perc'][9] = -1

    print("length of contrasts = ", len(contrasts))
    print("Contrasts include : ", contrasts)
    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        # print('  Contrast % 2i out of %i: %s' %
        #            (index + 1, len(contrasts), contrast_id))
        # save the z_image
        image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
        z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
        save(z_map, image_path)

        # Create snapshots of the contrasts
        vmax = max(-z_map.get_data().min(), z_map.get_data().max())
        plot_map(z_map.get_data(), z_map.get_affine(),
                 cmap=cm.cold_hot, vmin=-vmax, vmax=vmax,
                 slicer='z', black_bg=True, threshold=2.5,
                 title=contrast_id)
        plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

    print("All the  results were witten in %s" % write_dir)






def first_level_exemplar(subject, faceOrplace, imagineOrPerceive): # obtain contrast images
    if faceOrplace != 'Faces' and faceOrplace != 'Places':
        print("You need to enter either Faces or Places as a parameter to this function")
    else:
        print(
            "Conducting single_level_analysis at the exemplar level for " + faceOrplace + "condition, for subject number " + subject)
    # get filepaths for faces and places
    filepath = condition_type_filepath(faceOrplace, subject, types='spm')

    condition_list, onset, duration =  getOnsetExemplarLevelM(filepath, task=imagineOrPerceive)  ##
    print(condition_list)

    paradigm = BlockParadigm(con_id=condition_list, onset=onset, duration=duration)

    n_scans = 330  # dynamics
    tr = 3
    frametimes = np.arange(0, n_scans * tr, tr)
    hrf_model = 'canonical'
    drift_model = "cosine"
    hfcut = 128  # bandpass
    design_matrix = make_dmtx(frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model, hfcut=hfcut)
    ax = design_matrix.show()
    ax.set_position([.05, .25, .9, .65])
    ax.set_title('Design matrix')
    resultDir = 'C://Users//hlw69//Documents//fMRI_Exeter//masked_results//'
    filename = subject + "_" + faceOrplace + '//'
    write_dir = path.join(resultDir, filename)
    if not path.exists(write_dir):
        mkdir(write_dir)
    write_dir = path.join(write_dir, (faceOrplace + '_contrasts//'))
    print(write_dir)
    if not path.exists(write_dir):
        mkdir(write_dir)
    write_dir = path.join(write_dir, (imagineOrPerceive))
    print(write_dir)
    if not path.exists(write_dir):
        mkdir(write_dir)
    plt.savefig(path.join(write_dir, 'design_matrix_exemplar_level.png'))

    spm4Dfilepath = condition_type_filepath(faceOrplace, subject, types='4D')
    print(spm4Dfilepath)  #
    img_4d = nib.load(spm4Dfilepath)
    print(" Shape of img_4d is ", img_4d.shape)

    ##  Create the fmri linear model
    # fmri_glm = FMRILinearModel(spm4Dfilepath, design_matrix.matrix,
    fmri_glm = FMRILinearModel(img_4d, design_matrix.matrix,
                               mask='compute')
    fmri_glm.fit(do_scaling=True, model='ar1')
    print("image before contrast is ", img_4d.shape)

    identity = np.identity(51, dtype=int).tolist()
    contrasts = {}
    for x in range(1, stim_num + 1):
        contrasts[x] = np.array(identity[x - 1])  #
    print(len(contrasts))
    print(contrasts.keys())
    # print(contrasts.get('1'))
    # print(contrasts)

    print('Computing contrasts...')
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        print('  Contrast % 2i out of %i: %s' %
              (index + 1, len(contrasts), contrast_id))
        # save the z_image
        image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
        z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
        print("z_map shape ", z_map.shape)
        print("Image path is ", image_path)
        save(z_map, image_path)

        # Create snapshots of the contrasts
        vmax = max(-z_map.get_data().min(), z_map.get_data().max())
#        plot_map(z_map.get_data(), z_map.get_affine(),
#                 cmap=cm.cold_hot, vmin=-vmax, vmax=vmax,
#                 slicer='z', black_bg=True, threshold=2.5,
#                 title=contrast_id)
#        plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

    print("All the  results were witten in %s" % write_dir)
    # plt.show()
    print("shape of contrasts is ", contrasts[5])
    print(contrasts[5].shape, "SHAPE")
    return contrasts  ## ZMAP SHOULD BE RETURNED OR US


# get all the contrast files for faces the .nii z_map pictures, then also get the mask from directory
def getContrastFilesMask(faceOrPlace, subject, condition, mask):
    print(faceOrPlace, "condition")
    directory = 'C://Users//hlw69//Documents//fMRI_Exeter//masked_results//' + str(
        subject) + '_' + faceOrPlace + '//' + faceOrPlace + '_contrasts//' + condition + '//'
    os.chdir(directory)
    contrastfiles = []
    for file in glob.glob("*.nii"):
        contrastfiles.append(file)
    os.listdir()
    contrastfiles.sort(key=lambda x: float(x.strip('_z_map.nii')))
    if mask == None:
        print("No mask will be loaded as it is set to none")
    #  maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//masks_roi//bounded17_18_19_dilated_1.nii'#rdilation_one_3D.nii'
    # maskDirectory = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_'+str(subject)+'//'+mask+'//mask.img'
    else:
        print("A mask had been selected, the mask directory is ", maskDirectory)
        mask = nibabel.load(maskDirectory).get_data()
        mask = mask.astype(bool)
    return contrastfiles, directory, mask

# gets .nii data and applies mask, if mask is equal to None, then just loads data normally
def applyMaskData(mask, contrastfiles, directory):
    contrast_array = []
    for i,contrast in enumerate(contrastfiles):
        print(i)
        img_nifti = nibabel.load(directory + contrast)
        img = img_nifti.get_data()
        X = img
        if mask == None:
            print("No mask will be applied")
        else:
            print("Shape of mask is ", mask.shape)
            print("Shape of img in applymaskdata is ,", img.shape)
            X = img[mask]
        print(X.shape)
        # WHY HAVE I CHANGED THIS TO BE A LIST??
        contrast_array.append(list(X))
        print("X : ", type(X))
        print("contrast_array : ", type(contrast_array))
    return contrast_array

# this is to get the .nii files associated with each perception condition for one task condition (i.e faces)
# no contrasting is done here, so there should be around 3 files for each exemplar
# to create a 3D, rather than 4D (time) for each exemplar, will take mean of the 3 timesteps
from nilearn import plotting, surface
for subject in range(first_stim, last_stim+1): #59
    if subject == 39:
        print("skip 39")
    else:
        print(subject)
        _4Dfilepath = condition_type_filepath("faces", str(subject), types='4D')
        img_4d = nib.load(_4Dfilepath).get_fdata()
       # print(img_4d.shape)
        subject = str(subject)
        filepath = condition_type_filepath("faces", str(subject), types='spm')
        condition_list, onset, duration = getOnsetExemplarLevelM(filepath, task='imagine')  # will give 4 blocks of 7, then a big gap
        frametimes = np.arange(0, n_scans * tr, tr)
        stimulus_end = [x + duration[0] for x in onset]
        i = 1/tr #to bring down to 0, rather than 0.333333
        # getOnset returns in seconds, TR is 3, so needs to be divided into 3 to get closest sampling point
        tr_onset = [round((x / tr)-i) for x in onset]         # need to round as trs are now numbers with decimal points
        tr_end = [round((x / tr)) for x in stimulus_end]
        # iterate through and get associated .nii files
        mainf = 'C://Users//hlw69//Documents//temp_fmri_exeter//fmri_npy//'
        averaged_nii = []
        for start, stop, condition in zip(tr_onset, tr_end, condition_list):
            print("to be averaged")
            for sample in range(start, stop):
                t_slice = img_4d[..., start:stop]
                average_slice = np.mean(t_slice, axis=3)
                print(average_slice.dtype, average_slice.shape)

                # subject # task # condition # exemplar
                newfile = mainf + str(subject) + "face" + condition
                np.save(newfile, average_slice)
                # 4d
              #  averaged_nii.append(average_slice)


#print(tr_onset)
#print(tr_end)

       # contrast = first_level_exemplar(subject, type, 'perceive')
#       contrast = first_level_exemplar(subject, 'Places', 'perceive')
# contrast = first_level_exemplar(subject, 'Faces', 'imagine')
#  contrast = first_level_exemplar(subject, 'Places', 'imagine')
"""
# all files are now created for the first level analysis, should not need to get exemplar level files
# when performing logistic regression
# all need is to extract the contrast files.. such as perception
# then convert  them into the correct format.. masked and array form.
a_ContrastFiles, a_Directory, a_Mask = getContrastFilesMask(type, subject, 'perceive', mask='faces')
# b_ContrastFiles, b_Directory, b_Mask = getContrastFilesMask('Places', subject, 'perceive', mask = 'Faces')
# apply mask - and needs to be in array form for logistic regression
a_contrast_arrays = np.array(applyMaskData(a_Mask, a_ContrastFiles, a_Directory))
# b_contrast_arrays = np.array(applyMaskData(b_Mask, b_ContrastFiles, b_Directory))
"""

# each subject one by one, creates perception contrast arrays for faces and places - should do this for imagination too, then store them somewhere















"""
dict_faces_perception = {}
dict_places_perception = {}
for subject in range(first_stim, last_stim): #59
    # ignore any subjects to be excluded, for example '39' needs to be fully processed
    if subject in exclude_subjects:
        dict_faces_perception[str(subject)] = 'NULL'
        dict_places_perception[str(subject)] = 'NULL'
        print("skip")
    else:
        print("subject is ", subject)
        if 'faces' in task_conditions:
            a_ContrastFiles, a_Directory, a_Mask = getContrastFilesMask(type, subject, 'perceive', mask='faces')
            Perceive_Faces_contrast_arrays = np.array(applyMaskData(a_Mask, a_ContrastFiles, a_Directory))
            print(Perceive_Faces_contrast_arrays.shape)
           # print(len(Perceive_Faces_contrast_rrays))
           # print("Final length is ", len(Perceive_Faces_contrast_arrays[0]), len(Perceive_Faces_contrast_arrays[1]))
           # print("Final length is ", len(Perceive_Faces_contrast_arrays[3][0][0]))
           # print("Final length is ", len(Perceive_Faces_contrast_arrays[2][0][0][0]))
            dict_faces_perception[str(subject)] = Perceive_Faces_contrast_arrays
        if 'places' in task_conditions:
            b_ContrastFiles, b_Directory, b_Mask = getContrastFilesMask(type, subject, 'perceive', mask = 'Faces')
            Perceive_Places_contrast_arrays = applyMaskData(b_Mask, b_ContrastFiles, b_Directory)
            dict_places_perception[str(subject)] = Perceive_Places_contrast_arrays

# dirname = os.path.dirname(fn)
# if not os.path.exists(dirname):
#    os.makedirs(dirname)
# put the data into .json files

#old_dict = {'name':"Foobar", 'baz':42}
#wanted_keys = ['36', '37', '38', '39', '40', '41']
#new_dict = {k: old_dict[k] for k in set(wanted_keys) & set(old_dict.keys())}

dumped = json.dumps(dict_faces_perception, cls=NumpyEncoder)
with open(fn, 'w') as f:
    json.dumps(dumped, f)
"""

"""
import scipy.io as sio
sio.savemat(fn, dict_faces_perception)
pydict = sio.loadmat(fn)
print(pydict.keys())


with open(fn, 'w') as fp:
    print("opened")
    json.dump(dict_faces_perception, fp)
    print("finished")
# with open(fn, 'r') as fp:
#    contrasts_dict_faces_perception = fp.readline()
#    contrasts_dict = json.loads(contrasts_dict_faces_perception)
fp.close()
print("closed")
"""
