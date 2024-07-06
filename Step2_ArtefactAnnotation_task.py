## loop through all tasks of the same participant
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

for task in task_list[:4]: # loop through all tasks and merge bad sensors for different runs of the same task
    files = os.listdir(op.join(bids_root,f'sub-{subject}',f'ses-{session}','meg'))
    task_files = [file for file in files if file.endswith('.fif') & file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]

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
    
        ## Identifying eye blinks
        # Here we show how the artefacts associated with eye blinks can be marked automatically on the basis of the vertical EOG channel (EOG001 in our case). 
        # After bandpass filtering the EOG signal between 1 - 10 Hz, the threshold for blink detection is determined 
        eog_events        = mne.preprocessing.find_eog_events(raw, ch_name='EOG001') 
        n_blinks          = len(eog_events)
        onset             = eog_events[:, 0] / raw.info['sfreq'] - 0.25
        duration          = np.repeat(0.5, n_blinks)
        description       = ['blink'] * n_blinks
        orig_time         = raw.info['meas_date']
        annotations_blink = mne.Annotations(onset, duration, description, orig_time)
    
        ## Finding muscle artefacts
        #  Muscle artefacts are identified from the magnetometer data filtered in the 110 - 140 Hz range. 
        # The data are subsequently z-scored. If they exceed the value threshold_muscle, 
        # the corresponding section of data is annotated as muscle artifact. 
        # The shortest allowed duration of non-annotated data is 0.2 s; shorter segments will be incorporated into the surrounding annotations.
        threshold_muscle = 10  
        annotations_muscle, scores_muscle = annotate_muscle_zscore(
            raw, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
            filter_freq=[110, 140])
    
        preprocessing.osl_wrappers.detect_badsegments(raw,picks='meg',segment_len=500) # bad_segment_meg
        ## Include annotations in dataset and inspect
        # Now mark all the annotations in the data set.
        # Calling set_annotations() replaces any annotations currently stored in the Raw object. 
        # To prevent that, we first extract their annotations and then combine them together with the blink and muscle annotations.
        annotations_event = raw.annotations
        raw.set_annotations(annotations_event + annotations_blink + annotations_muscle)
            
        raw.plot(start=50,block=True)

        ## Save the annotations in a file to be used in the subsequent sections:
        raw.save(deriv_fname_fif, overwrite=True)
        raw.annotations.save(deriv_fname_csv, overwrite=True)
