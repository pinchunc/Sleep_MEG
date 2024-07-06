import os.path as op
import os
import matplotlib.pyplot as plt
import numpy as np 

import mne
from mne.annotations import Annotations

from mne_bids import (BIDSPath, write_raw_bids, read_raw_bids, 
                      write_meg_calibration, write_meg_crosstalk, 
                      get_anat_landmarks, write_anat)

from FLUXSettings import bids_root,subject,session

## only deal with MEG files for now
# need to add anat later
get_ipython().run_line_magic("matplotlib", "qt")

#%%
# specify specific file names
MEG_data_root  = f'/Volumes/MEGMORI/meg/s{subject}/raw'  # RDS folder for MEG data
bids_root      = '/Volumes/MEGMORI/BIDS-data' #op.join(MEG_data_root, 'BIDS-data')
file_extension = '.fif'

# Define the fine calibration and cross-talk compensation files for BIDS
maxfilter_folder = op.join(bids_root, 'crosstalk')

# files from Anna Camera for our MEG system, same for all our recordings
crosstalk_file   = op.join(maxfilter_folder, 'ct_sparse.fif')  #'reduces interference' 
                                                                #'between Elekta's co-located' 
                                                                #'magnetometer and'
                                                                #'paired gradiometer sensor units'
calibration_file = op.join(maxfilter_folder, 'sss_cal.dat')  #'encodes site-specific'
                                                                #'information about sensor' 
                                                                #'orientations and calibration'

#%%
task    = 'loc'  # name of the task
                                                                # Read the events from stim channel
for run in [1,2]:
    file_name = f's{subject}_{task}_r{run}_raw'
    raw_fname = op.join(MEG_data_root, file_name + file_extension) 
    filename_events = op.join(MEG_data_root, file_name + '-eve' + file_extension)


    # read raw and define the stim channel
    raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True, 
                              verbose=True, preload=False)
    stim_channel = 'STI101'

    events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.001001,
                             consecutive=False, mask=65280,
                             mask_type='not_and')  #' mask removes triggers associated
                                                   # with response box channel 
                                                   # (not the response triggers)'
    if  np.size(events)/3 == 292:
        events = events[6:]
    else:
        break
        
    eventlabel_path = '/Volumes/MEGMORI/behaviour/loc_events'
    eventlabel_file = f's{subject}_trig_{task}_r{run}.csv'
    print(eventlabel_file)
    eventlabel_fname = op.join(eventlabel_path, eventlabel_file)
    event_vals = np.genfromtxt(eventlabel_fname,delimiter=',')

    for count, value in enumerate(events):
        events[count][2] = int(event_vals[count])
    
    events_id = {'Body': 1, 'Face': 2, 'Object': 3, 'Scene': 4, 'Word': 5, 'Control': 6, 'Repeated': 999}
    
    # Save the events in a dedicted FIF-file: 
    mne.write_events(filename_events, events, overwrite=True)
    
    
    fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_id
    )

    # Bids preparation
    bids_path = BIDSPath(subject=subject, session=session,
                          task=task, run=run, root=bids_root)
    

    write_raw_bids(raw, bids_path, events=events, 
                       event_id=events_id, overwrite=True)
    
    # Write in Maxfilter files
    write_meg_calibration(calibration_file, bids_path=bids_path, verbose=False)
    write_meg_crosstalk(crosstalk_file, bids_path=bids_path, verbose=False)
    
#%%
task  = 'sog'  # name of the task

for run in [1,2]:
   file_name = f's{subject}_{task}_r{run}_raw'
   raw_fname = op.join(MEG_data_root, file_name + file_extension) 
   filename_events = op.join(MEG_data_root, file_name + '-eve' + file_extension)

   # read raw and define the stim channel
   raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True, 
                             verbose=True, preload=False)
   stim_channel = 'STI101'

   events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.001001,
                            consecutive=False, mask=65280,
                            mask_type='not_and')  #' mask removes triggers associated
                                                  # with response box channel 
                                                  # (not the response triggers)'
   events_id = {'run_start': 1, 'blockstart': 2, 'crossOn': 4, 'run_end': 8, 'endblock': 16, 'endtask': 32}

   # Save the events in a dedicted FIF-file: 
   mne.write_events(filename_events, events, overwrite=True)


   fig = mne.viz.plot_events(
       events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_id
   )

   # Bids preparation
   bids_path = BIDSPath(subject=subject, session=session,
                         task=task, run=run, root=bids_root)


   write_raw_bids(raw, bids_path, events=events, 
                      event_id=events_id, overwrite=True)

   # Write in Maxfilter files
   write_meg_calibration(calibration_file, bids_path=bids_path, verbose=False)
   write_meg_crosstalk(crosstalk_file, bids_path=bids_path, verbose=False)
   
#%%
task  = 'mem'  # name of the task

files = os.listdir(MEG_data_root)
mem_files = [file for file in files if file.startswith(f's{subject}_{task}_r') & file.endswith('raw.fif')]

for run_idx in [1,2]:#range(len(mem_files)):
    run = run_idx+1
    print(run)
    file_name = f's{subject}_{task}_r{run}_raw'
    raw_fname = op.join(MEG_data_root, file_name + file_extension) 
    filename_events = op.join(MEG_data_root, file_name + '-eve' + file_extension)

    # read raw and define the stim channel
    raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True, 
                              verbose=True, preload=False)
    stim_channel = 'STI101'

    events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.001001,
                             consecutive=False, mask=65280,
                             mask_type='not_and')  #' mask removes triggers associated
                                                   # with response box channel 
                                                   # (not the response triggers)'
                                                   
    if 64 not in events[:,2]:
        events = np.append(events,np.array([[events[-1][0]+1000, 0, 64 ]]), axis=0)
    
    events_id = {'run_start': 1, 'run_start_burst': 10, 'learning_round': 2, 'fixation_on': 4, 'search_on': 8, 'image_on': 16, 'image_off': 18, 'testStart': 32, 'image_selected': 33, 'image_dropped': 34, 'run_end': 64}

    # Save the events in a dedicted FIF-file: 
    mne.write_events(filename_events, events, overwrite=True)

    fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_id
    )

    # Bids preparation
    bids_path = BIDSPath(subject=subject, session=session,
                          task=task, run=run, root=bids_root)


    write_raw_bids(raw, bids_path, events=events, 
                       event_id=events_id, overwrite=True)

    # Write in Maxfilter files
    write_meg_calibration(calibration_file, bids_path=bids_path, verbose=False)
    write_meg_crosstalk(crosstalk_file, bids_path=bids_path, verbose=False)

#%%
task    = 'memtest'  # name of the task

run = 1
file_name = f's{subject}_{task}_r{run}_raw'
raw_fname = op.join(MEG_data_root, file_name + file_extension) 
filename_events = op.join(MEG_data_root, file_name + '-eve' + file_extension)

# read raw and define the stim channel
raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True, 
                          verbose=True, preload=False)
stim_channel = 'STI101'

events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.001001,
                         consecutive=False, mask=65280,
                         mask_type='not_and')  #' mask removes triggers associated
                                               # with response box channel 
                                               # (not the response triggers)'
events_id = {'run_start': 1, 'run_start_burst': 10, 'testStart': 32, 'image_selected': 33, 'image_dropped': 34, 'run_end': 64}

# Save the events in a dedicted FIF-file: 
mne.write_events(filename_events, events, overwrite=True)


fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_id
)

# Bids preparation
bids_path = BIDSPath(subject=subject, session=session,
                      task=task, run=run, root=bids_root)


write_raw_bids(raw, bids_path, events=events, 
                   event_id=events_id, overwrite=True)

# Write in Maxfilter files
write_meg_calibration(calibration_file, bids_path=bids_path, verbose=False)
write_meg_crosstalk(crosstalk_file, bids_path=bids_path, verbose=False)

#%% add sleep staging as events to raw file
task    = 'sleep'  # name of the task
import scipy.io

files = os.listdir(MEG_data_root)
sleep_files = [file for file in files if file.startswith(f's{subject}_{task}_r')& file.endswith('raw.fif')]

for run_idx in range(len(sleep_files)):
    run = run_idx+1
    print(run)
    file_name = f's{subject}_{task}_r{run}_raw'
    raw_fname = op.join(MEG_data_root, file_name + file_extension) 
    stage_fname = f'/Volumes/MEGMORI/sleep_edf/s{subject}/auto_stage_s{subject}_r{run}.mat' 
    stage_data = scipy.io.loadmat(stage_fname)
    stage_data = stage_data['stageData']['stages'][0][0]
    
    # read raw and define the stim channel
    raw = mne.io.read_raw_fif(raw_fname, allow_maxshield=True, 
                              verbose=True, preload=False)
    
    sfreq = raw.info['sfreq']  # should be 1000 Hz in your case
    # epoch_duration = 30  # seconds
    
    # # Create event markers
    # events = []
    # for i, stage in enumerate(stage_data):
    #     start_sample = raw.first_samp + int(i * epoch_duration * sfreq)
    #     events.append([start_sample, 0, stage[0]])
    
    # events = np.array(events)
    
    # # Get unique sleep stages
    # unique_stages = np.unique(stage_data)
    
    # # Define event IDs based on unique sleep stages
    # events_id = {}
    # for stage_id in unique_stages:
    #     events_id[f'Stage_{stage_id}'] = stage_id


    # orig_time   = raw.info['meas_date']
    # annotations = Annotations(onset=events[:, 0] / sfreq,
    #                           duration=epoch_duration,
    #                           description=[str(stage) for stage in events[:, 2]]) #, orig_time=orig_time
   
    # # Add annotations to raw data
    # raw.set_annotations(annotations)
    # raw.plot(start=0, duration=30.0,block=True) # check sleep stages
           
    # fig = mne.viz.plot_events(
    #     events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_id
    # )

    # Bids preparation
    bids_path = BIDSPath(subject=subject, session=session,
                          task=task, run=run, root=bids_root)

    write_raw_bids(raw, bids_path, overwrite=True)

    # Write in Maxfilter files
    write_meg_calibration(calibration_file, bids_path=bids_path, verbose=False)
    write_meg_crosstalk(crosstalk_file, bids_path=bids_path, verbose=False)
    
#%% MRI bids conversion
## DO THIS AFTER HEADMODEL IS COREGISTERED
deriv_root   = op.join(bids_root, 'derivatives/analysis')
trans_bids_path = BIDSPath(subject=subject, session=session,
                           task='loc', run=12, suffix='trans', datatype='meg',
                           root=deriv_root, extension='.fif', check=False)

fs_sub_dir = '/Volumes/MEGMORI/BIDS-data/derivatives/FreeSurfer'
fs_subject = 'sub-' + subject  # name of subject from freesurfer
t1_fname = op.join(fs_sub_dir, fs_subject, 'mri', 'T1.mgz')

# Create the BIDSpath object
""" creat MRI specific bidspath object and then use trans file to transform 
landmarks from the raw file to the voxel space of the image"""

t1w_bids_path = BIDSPath(subject=subject, session=session, 
                         root=bids_root, suffix='T1w')

meg_bids_path = BIDSPath(subject=subject, session=session,
                      task='loc', run=1, root=bids_root)
info = read_raw_bids(bids_path=meg_bids_path, verbose=False).info

trans = mne.read_trans(trans_bids_path)  
landmarks = get_anat_landmarks(
    image=t1_fname,  # path to the nifti file
    info=info,       # MEG data file info from the subject
    trans=trans,
    fs_subject=fs_subject,
    fs_subjects_dir=fs_sub_dir)

t1w_bids_path = write_anat(
    image=t1_fname, bids_path=t1w_bids_path,
    landmarks=landmarks, deface=False,
    overwrite=True, verbose=True)