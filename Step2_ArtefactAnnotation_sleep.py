## IN PROGRESS
## loop through all sleep run of the same participant
import os.path as op
import os
import sys
import numpy as np

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import annotate_muscle_zscore
import matplotlib
import matplotlib.pyplot as plt
from osl import preprocessing

# import settings
from FLUXSettings import bids_root,subject,task_list,session

matplotlib.use('Qt5Agg')

## Set the path and load the FIF-files:
meg_suffix = 'meg'
max_suffix = 'raw_sss'
ann_suffix = 'ann'

deriv_root = op.join(bids_root, 'derivatives/preprocessing')  # output path
#%%
task = 'sleep'
files = os.listdir(op.join(bids_root,f'sub-{subject}',f'ses-{session}','meg'))
task_files = [file for file in files if file.endswith('.fif') & file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]
#%%
for run_idx in range(len(task_files)):
    run = run_idx+1
    print(run)
    
    bids_path = BIDSPath(subject=subject, session=session, datatype='meg',
                task=task, run=run, suffix=max_suffix,
                root=deriv_root, extension='.fif', check=False)

    # The annotations will be stored in these files
    deriv_fname = bids_path.basename.replace(max_suffix, ann_suffix) # fif output filename

    deriv_fname_fif = op.join(bids_path.directory, deriv_fname)    
    deriv_fname_csv = deriv_fname_fif.replace('fif', 'csv') # csv output filename
    
    print(bids_path)
    print(deriv_fname_fif)
    print(deriv_fname_csv)

    # read data
    raw = read_raw_bids(bids_path=bids_path, 
                        extra_params={'preload':True},
                        verbose=True)

    ## Identifying eye blinks # also detect slow eye movement
    # not very usefule for sleep data but still save the annotation for wake epochs
    # Here we show how the artefacts associated with eye blinks can be marked automatically on the basis of the vertical EOG channel (EOG001 in our case). 
    eog_events        = mne.preprocessing.find_eog_events(raw, ch_name='EOG001')
    n_blinks          = len(eog_events)
    onset             = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration          = np.repeat(0.5, n_blinks)
    description       = ['blink'] * n_blinks
    orig_time         = raw.info['meas_date']
    annotations_blink = mne.Annotations(onset, duration, description, orig_time)

    threshold_muscle = 10  
    annotations_muscle, scores_muscle = annotate_muscle_zscore(
        raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.5,
        filter_freq=[110, 140])
        
    ## Finding muscle artefacts and remove 1sec window
    # not very useful for sleeo; too sensitive when muscle tone is low
    # emg_events        = mne.preprocessing.find_eog_events(raw, ch_name='EMG004') 
    # n_emg             = len(emg_events)
    # onset             = emg_events[:, 0] / raw.info['sfreq'] - 0.5
    # duration          = np.repeat(1, n_emg)
    # description       = ['BAD_muscle'] * n_emg
    # orig_time         = raw.info['meas_date']
    # annotations_emg   = mne.Annotations(onset, duration, description, orig_time)

    # preprocessing.osl_wrappers.detect_badsegments(raw,picks='meg',segment_len=500) # this detect gradiant
    # annotations_event = raw.annotations
    # will create annotation named bad_segment_meg
    # preprocessing.osl_wrappers.detect_badsegments(raw,picks='meg',segment_len=500, mode = 'maxfilter')
    # When ``mode='diff'`` will calculate a difference time series before
    # detecting bad segments. When ``mode='maxfilter'`` we only mark the
    # segments with zeros from MaxFiltering as bad.
    # the setting maxfilter only detect maxfilter rejected timestamps # need to check how it reads in the maxfilter bad data
    
    ## Include annotations in dataset and inspect
    # Now mark all the annotations in the data set.
    # Calling set_annotations() replaces any annotations currently stored in the Raw object. 
    # To prevent that, we first extract their annotations and then combine them together with the blink and muscle annotations.
    
    # if all hpi were inactive n_active is a zero-array
    n_active = mne.chpi.get_active_chpi(raw)
    # Find the indices where n_active is less than 3
    
    indices           = np.where(n_active < 3)[0] + raw.first_samp
    continuous_segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    segment_bounds = [(segment[0], segment[-1]) for segment in continuous_segments if len(segment) > 0]
    segment_durations = []
    onsets = []
    for start, end in segment_bounds:
        segment_duration = (end - start + 1)/ raw.info['sfreq'] + 0.5  # +1 because end index is inclusive
        segment_durations.append(segment_duration)
    
        onset = start / raw.info['sfreq'] - 0.25  # Calculate onset in seconds
        onsets.append(onset)

    n_bad_hp          = len(continuous_segments)
    annotations_bad_hpi = mne.Annotations(onset = onsets, duration=segment_durations, description = ['BAD_HPI'] * n_bad_hp, orig_time=raw.info['meas_date'])

    raw.set_annotations(annotations_bad_hpi + annotations_blink + annotations_muscle) #annotations_emg
    raw.plot(start=0,block=True)

    ## Save the annotations in a file to be used in the subsequent sections:
    raw.save(deriv_fname_fif, overwrite=True)
    raw.annotations.save(deriv_fname_csv, overwrite=True)
