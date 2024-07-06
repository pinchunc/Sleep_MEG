import os.path as op
import os
import sys
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids
from autoreject import get_rejection_threshold
import numpy as np
# import settings
from FLUXSettings import bids_root,subject,task_list,session
get_ipython().run_line_magic("matplotlib", "qt")


resample_sfreq = 500;
l_freq = 0.35
h_freq = 200 # for gamma
iir_params = dict(order=5, ftype='butter')
notch_filter_freqs = (50) # must be lowe than lowpass
task = 'sleep'
meg_suffix = 'meg'
ica_suffix = 'ica'
epo_suffix = 'epo'
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

#%% load sleep events timing
# megeeg_slythms_nrem_Cz.mat
import scipy.io
slythm_fname = op.join(deriv_root, f'sub-{subject}',f'ses-{session}','megeeg','megeeg_slythms_nrem_Cz.mat')
slythm_data = scipy.io.loadmat(slythm_fname)
spindles_minTime_timestamps = slythm_data['spindles_minTime_timestamps'][0]
spindles_minTime_run_idx = slythm_data['spindles_minTime_run_idx'][0]
spindles_stage = slythm_data['spindles_stage'][0]
spindles_surrogate_minTime_timestamps = slythm_data['spindles_surrogate_minTime_timestamps'][0]
spindles_surrogate_minTime_run_idx = slythm_data['spindles_surrogate_minTime_run_idx'][0]

SO_minTime_timestamps = slythm_data['SO_minTime_timestamps'][0]
SO_minTime_run_idx = slythm_data['SO_minTime_run_idx'][0]
SO_stage = slythm_data['SO_stage'][0]
SO_surrogate_minTime_timestamps = slythm_data['SO_surrogate_minTime_timestamps'][0]
SO_surrogate_minTime_run_idx = slythm_data['SO_surrogate_minTime_run_idx'][0]

#%%
bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=1, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False).mkdir()

# deriv_file_slythms = bids_path.basename.replace('run-01', 'run-all-slythms')  # run 12 -> run 01 concatenated with run 02
# deriv_fname_slythms = op.join(bids_path.directory, deriv_file_slythms)
deriv_file_spindles = bids_path.basename.replace('run-01', 'run-all-spindles')  # run 12 -> run 01 concatenated with run 02
deriv_fname_spindles = op.join(bids_path.directory, deriv_file_spindles)
deriv_file_SO = bids_path.basename.replace('run-01', 'run-all-SO')  # run 12 -> run 01 concatenated with run 02
deriv_fname_SO = op.join(bids_path.directory, deriv_file_SO)

#%% loop through runs
raw_list = []
events_list = []

task = 'sleep'
files = os.listdir(os.path.join(bids_root, f'sub-{subject}', f'ses-{session}', 'meg'))
task_files = [file for file in files if file.endswith('.fif') and file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]

for run_idx in range(len(task_files)):
    run = run_idx + 1
    bids_path_preproc = BIDSPath(subject=subject, session=session,
                                 task=task, run=run, suffix=meg_suffix, datatype='meg',
                                 root=preproc_root, extension='.fif', check=False)

    bids_path = BIDSPath(subject=subject, session=session,
                         task=task, run=run, suffix=epo_suffix, datatype='meg',
                         root=deriv_root, extension='.fif', check=False).mkdir()

    print(bids_path_preproc.fpath)

    # Create event markers
    ica_path = BIDSPath(subject=subject, session=session,
                        task=task, run=run, suffix=ica_suffix, datatype='meg',
                        root=preproc_root, extension='.fif', check=False).mkdir()
    raw = mne.io.read_raw_fif(ica_path, allow_maxshield=True, verbose=True, preload=False)
    sfreq = raw.info['sfreq']  # should be 1000 Hz in your case

    stage_fname = f'/Volumes/MEGMORI/sleep_edf/s{subject}/auto_stage_s{subject}_r{run}.mat'
    stage_data = scipy.io.loadmat(stage_fname)
    stage_data = stage_data['stageData']['stages'][0][0]

    # Check if stage_data contains stages 2 or 3
    if not (2 in stage_data or 3 in stage_data):
        print(f"Stages 2 or 3 not found in stage data for run {run}. Skipping...")
        continue

    # Create event markers
    epoch_duration = 30  # seconds
    stages_events = []
    for i, stage in enumerate(stage_data):
        start_sample = int(i * epoch_duration * sfreq) + raw.first_samp
        stages_events.append([start_sample, 0, stage[0]])

    stages_events = np.array(stages_events)
    onset = stages_events[:, 0] / raw.info['sfreq']
    n_stages = len(stages_events)
    duration = np.repeat(epoch_duration, n_stages)
    description = [str(stage) for stage in stages_events[:, 2]]
    orig_time = raw.info['meas_date']
    annotations_stages = mne.Annotations(onset, duration, description, orig_time)

    raw.set_annotations(annotations_stages)
    events, event_id = mne.events_from_annotations(raw)

    # Create 5-second epochs within the same sleep stage
    epochs_list = []
    for stage in [2, 3]:
        stage_events = events[events[:, 2] == stage]
        for start_sample in stage_events[:, 0]:
            tmin = start_sample / sfreq
            tmax = tmin + 5.0
            if raw.annotations.onset[-1] < tmax:
                continue  # Skip if the 5 seconds interval extends beyond the recording
            if len(raw.annotations[(tmin <= raw.annotations.onset) & (raw.annotations.onset < tmax) & (raw.annotations.description == str(stage))]) == 0:
                # Skip if there are annotations that do not match the desired stage within this interval
                continue
            epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True)
            epochs_list.append(epochs)

    # Combine all epochs and append to list
    all_epochs = mne.concatenate_epochs(epochs_list)
    raw_list.append(raw)
    events_list.append(events)

    # Save the epochs
    epochs_fname = os.path.join(bids_path.directory, f's{subject}_{task}_r{run}_5sec-epo.fif')
    all_epochs.save(epochs_fname, overwrite=True)
    
    

raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
del raw_list
raw.plot(start=0, duration = 30,block=True)

#%% downsample
# current_sfreq = raw.info["sfreq"]
# decim = np.round(current_sfreq / resample_sfreq).astype(int)
# obtained_sfreq = current_sfreq / decim
# lowpass_freq = obtained_sfreq / 3.0

# raw_downsampled = raw.copy().resample(sfreq = resample_sfreq)
# raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=lowpass_freq)
# events = mne.find_events(raw_filtered)
# epochs = mne.Epochs(raw_filtered, events, decim=decim)
# raw.plot(start=0, duration = 30,block=True)


#%% spindles_events
#events_dict = {'Body': 1, 'Face': 2, 'Object': 3, 'Scene': 4, 'Word': 5, 'Control': 6, 'Repeated': 999}
events, events_id = mne.events_from_annotations(raw) #, event_id=events_dict
events_picks_id = {k:v for k, v in events_id.items() if k.startswith('spindles_events') or k.startswith('spindles_surr_events')}

# Make epochs 
epochs = mne.Epochs(raw,
    events, events_picks_id,
    tmin=-2 , tmax=2,
    baseline=None,
    proj=False,
    picks = 'all',
    detrend = 0,
    reject=None,
    reject_by_annotation=True, ## need to check if HPI bad epochs were save
    preload=True,
    verbose=True)

# visually inspect
evoked = epochs['spindles_events'].average()
evoked.plot()
plt.show()


#%% use autoreject to reject trials 
#ar = AutoReject()
#epochs_clean = ar.fit_transform(epochs)
subset_size = 100  # Number of epochs to use for threshold estimation
random_indices = np.random.choice(len(epochs), size=subset_size, replace=False)
epochs_subset = epochs[random_indices]

reject    = get_rejection_threshold(epochs_subset, ch_types=['mag', 'grad'])
MEGreject = reject#{k: reject[k] for k in list(reject)[:2]}
del epochs 
del epochs_subset
print(MEGreject)

#%%
epochs = mne.Epochs(raw,
        events, events_picks_id,
        tmin=-2 , tmax=2,
        baseline=None,
        proj=False,
        picks = 'all',
        detrend = 0,
        reject=MEGreject,
        reject_by_annotation=False,
        preload=True,
        verbose=True)

epochs.plot_drop_log()
print(deriv_file_spindles)
epochs.save(deriv_fname_spindles, overwrite=True)


