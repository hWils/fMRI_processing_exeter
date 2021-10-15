"""
# This script is used for obtaining contrasts between Face and Place conditions
# It can be extended to save these into json format, however it currently saves
# .nii and .png files for each subject for face minus place; place minus face; faces; places
# This is carried out over all exemplars together, lose the temporal dimension and exemplar distinctions

"""

import scipy.io as spio
from nibabel import save
from nipy.modalities.fmri.glm import FMRILinearModel
from nipy.modalities.fmri.design_matrix import make_dmtx
from nipy.labs.viz import plot_map, cm
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm  # EventRelatedParadigm
from os import mkdir, path
from itertools import chain
import matplotlib.pyplot as plt
#from mvpa2.suite import *
import numpy as np
import nibabel as nib

# faces and places start in separate files in each subjects folder i.e 'Imagine_30' -> 'Faces' 'Places'


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


# create the correct FILEPATH for retreving the 4D fmri file for a particular subject and condition.
# There is one produced for each of the 320 dynamics. Combines all these files into one.
def condition_type_filepath(condition, subject, types):
    main_filepath = 'C://Users//hlw69//Documents//fMRI_Exeter//Imagery_Raw_and_processed_data//Imagery_Raw_and_processed_data//Imagine_' + subject
    if condition == 'Faces':
        filepath = main_filepath + '//Faces//'
        if types == 'spm':
            filepath = filepath + 'SPM.mat'
        elif types == '4D':
            filepath = filepath + 'Segmented//Motion_corrected//4D.nii.gz'
    elif condition == 'Places':
        filepath = main_filepath + '//Places//'
        if types == 'spm':
            filepath = filepath + 'SPM.mat'
        elif types == '4D':
            filepath = filepath + 'Segmented//Motion_corrected//4D.nii.gz'
            print("4d filepath is ", filepath)
    return filepath


# given an spm file path, extract the onset and duration times for each part of the experiment, including control conditions.
# This function has to be computed twice, once for the Faces 4d.nii and once for the Places
def getOnsetDurationDurImgPercSPM(condition_filepath, stim_num=36, face_place = 'Faces'):
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

    conditions = [['imagination' + face_place] * stim_num, ['name'+face_place] * stim_num, ['look_text'+face_place] * 9, ['perceive'+face_place] * stim_num,
                  ['imagine_text'+face_place] * 9, ['control_imagine'+face_place] * stim_num, ['control_name'+face_place] * stim_num,
                  ['look_control_text'+face_place] * 9, ['control_perceive'+face_place] * stim_num, ['control_imagine_text'+face_place] * 9]

    onsets = [imagination, name, look_text, perceive,
              imagine_text, control_imagine, control_name, look_control_text, control_perceive, control_imagine_text]

    durations = [imagination_dur, name_dur, look_text_dur, perceive_dur,
                 imagine_text_dur, control_imagine_dur, control_name_dur,
                 look_control_text_dur, control_perceive_dur, control_imagine_text_dur]
    return conditions, onsets, durations


# Input = condition, fMRI .nii files, spm .mat files. Output = designmatrix.png, GLM model, contrast images .nii
# Takes in the .nii files for each dynamic at the subject level, extracts the timings/condition from the .mat file
#  creates and saves a png of the design matrix. Then obtains the 4D fMRI data, and fits a GLM
# defines contrasts, then carries out contrasts, which produces a contrast .nii file and saves

def first_level_analysis(subject, imaginationOrPerception):
    if imaginationOrPerception != 'imagine' and imaginationOrPerception != 'perceive':
        print("You need to enter either Imagine or Perceive as a parameter to this function")
    else:
        print("Conducting single_level_analysis for " + imaginationOrPerception + "condition, for subject number " + subject)
    # get filepaths for faces and places

    facefilepath = condition_type_filepath('Faces', subject, types='spm')
    placefilepath = condition_type_filepath('Places', subject, types='spm')
    print(facefilepath, placefilepath)

    # onsets and durations for both places and faces for a given subject
    face_condition_list, face_onset, face_duration = getOnsetDurationDurImgPercSPM(facefilepath, 36, 'Faces')  ##########
    place_condition_list, place_onset, place_duration = getOnsetDurationDurImgPercSPM(placefilepath, 36, 'Places')

   # print(place_onset)

#    place_onset = place_onset + 330 # add this on so can append after places

    # flatten lists # FACES
    face_flattened_conditions = list(chain.from_iterable(face_condition_list))
    face_flattened_onsets = list(chain.from_iterable(face_onset))
    face_flattened_durations = list(chain.from_iterable(face_duration))

    # flatten lists # PLACES
    place_flattened_conditions = list(chain.from_iterable(place_condition_list))
    place_flattened_onsets = list(chain.from_iterable(place_onset))
    print(place_flattened_onsets)
   # from operator import add
    highest = max(face_flattened_onsets)
   # place_flattened_onsets = list(map(add, face_flattened_onsets, place_flattened_onsets))
    place_flattened_onsets = [x+highest for x in place_flattened_onsets]
    print(place_flattened_onsets)
    place_flattened_durations = list(chain.from_iterable(place_duration))

    flattened_conditions = face_flattened_conditions + place_flattened_conditions
    flattened_onsets = face_flattened_onsets + place_flattened_onsets
    flattened_durations = face_flattened_durations + place_flattened_durations

    print("These two numbers should be equal, if not there is an error ", len(flattened_conditions),
          len(flattened_onsets))
    paradigm = BlockParadigm(con_id=flattened_conditions, onset=flattened_onsets, duration=flattened_durations)
    # paradigm needs to be changed so faces and places are put together
    n_scans = 330 + 330  # dynamics. 330 + 330 as faces and places
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
    filename = subject + "_" + imaginationOrPerception
    write_dir = path.join(resultdir, filename)
    if not path.exists(write_dir):
        mkdir(write_dir)
    plt.savefig(path.join(write_dir, 'facesVsPlace_design_matrix.png'))

 # obtain the neural data, this will be in a 4D format nii.gz
    facesspm4Dfilepath = condition_type_filepath('Faces', subject, types='4D')
    placesspm4Dfilepath = condition_type_filepath('Places', subject, types='4D')

    faces_img_4d = nib.load(facesspm4Dfilepath)#.get_data()
    places_img_4d= nib.load(placesspm4Dfilepath)#.get_data()
    # append the places img to the end of the face img

    print("Shape of img_4d is ", faces_img_4d.shape, places_img_4d.shape)
    img_4d = nib.funcs.concat_images((faces_img_4d, places_img_4d),axis=3)
  #  img_4d = np.concatenate((faces_img_4d, places_img_4d),axis=3)
    #img_4d = np.moveaxis(img_4d, -1, 0) # move time dimension to the beginning
    print("New shape should be 79,95,69,660... is it " , img_4d.shape)
    ##  Create the fmri linear model
#    print(len(img_4d), len(design_matrix.matrix))
    fmri_glm = FMRILinearModel([img_4d], design_matrix.matrix,
                               mask='compute')
    fmri_glm.fit(do_scaling=True, model='ar1')

    contrasts = {}

    contrasts = {
        "faces_perceive": np.zeros(50),
        "places_perceive": np.zeros(50),
        "faces_minus_places_per": np.zeros(50),
        "places_minus_faces_per": np.zeros(50),
    }

    contrasts['faces_perceive'][18] = 1
    contrasts['places_perceive'][19] = 1
    contrasts['faces_minus_places_per'][18] = 1
    contrasts['faces_minus_places_per'][19] = -1
    contrasts['places_minus_faces_per'][18] = -1
    contrasts['places_minus_faces_per'][19] = 1

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


for subject in range(30, 31): #59
    if subject == 39:
        print("skip 39")
    else:
        print(subject)
        subject = str(subject)
        first_level_analysis(subject, 'perceive')
       # contrast = first_level_exemplar(subject, 'Faces', 'perceive')
#       contrast = first_level_exemplar(subject, 'Places', 'perceive')
# contrast = first_level_exemplar(subject, 'Faces', 'imagine')
#  contrast = first_level_exemplar(subject, 'Places', 'imagine')

