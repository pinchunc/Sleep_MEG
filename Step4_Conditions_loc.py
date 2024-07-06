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

# import settings
from FLUXSettings import bids_root,subject,task,session,run
matplotlib.use('Qt5Agg')


def main():
    resample_sfreq = 200;
    l_freq = 1
    h_freq = 90 # for gamma
    iir_params = dict(order=5, ftype='butter')
    notch_filter_freqs = (50) # must be lowe than lowpass
    task = 'loc'
    meg_suffix = 'meg'
    ica_suffix = 'ica'
    epo_suffix = 'epo'
    preproc_root = op.join(bids_root, 'derivatives/preprocessing')
    deriv_root   = op.join(bids_root, 'derivatives/analysis')

    run = 1
    bids_path_preproc = BIDSPath(subject=subject, session=session,
                task=task, run=run, suffix=ica_suffix, datatype='meg',
                root=preproc_root, extension='.fif', check=False)

    bids_path = BIDSPath(subject=subject, session=session,
                task=task, run=run, suffix=epo_suffix, datatype='meg',
                root=deriv_root, extension='.fif', check=False).mkdir()

    deriv_file = bids_path.basename.replace('run-01', 'run-12')  # run 12 -> run 01 concatenated with run 02
    deriv_fname = op.join(bids_path.directory, deriv_file)

    print(bids_path_preproc.fpath)
    print(deriv_fname)

    events_dict = {'Body': 1, 'Face': 2, 'Object': 3, 'Scene': 4, 'Word': 5, 'Control': 6, 'Repeated': 999}

    # combine two runs
    raw_list    = list()
    events_list = list()
    for subfile in range(2):
        print(subfile)
        if subfile == 0:
            bids_path_preproc.update(run='01')
        elif subfile == 1:
            bids_path_preproc.update(run='02')
        print(bids_path_preproc)
        raw = read_raw_bids(bids_path=bids_path_preproc, 
                extra_params={'preload':True},
                verbose=True)
        
        events, events_id = mne.events_from_annotations(raw, event_id=events_dict)
        print(events)                        
        raw_list.append(raw)
        events_list.append(events)

    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    del raw_list
    raw.plot(start=50,block=True)

    events_picks_id = {k:v for k, v in events_id.items() if not k.startswith('Repeated')}
    events_picks_id = {k:v for k, v in events_picks_id.items() if not k.startswith('blink')}
    events_picks_id = {k:v for k, v in events_picks_id.items() if not k.startswith('bad_segment_meg')}
    
    # Make epochs 
    epochs = mne.Epochs(raw,
        events, events_picks_id,
        tmin=-2 , tmax=2,
        baseline=None,
        proj=False,
        picks = 'all',
        detrend = 1,
        reject=None,
        reject_by_annotation=True,
        preload=True,
        verbose=True)

    # use autoreject to reject trials 
    #ar = AutoReject()
    #epochs_clean = ar.fit_transform(epochs)
    reject    = get_rejection_threshold(epochs)  
    MEGreject = {k: reject[k] for k in list(reject)[:2]}
    del epochs
    print(MEGreject)

    epochs = mne.Epochs(raw,
            events, events_picks_id,
            tmin=-2 , tmax=2,
            baseline=None,
            proj=False,
            picks = 'all',
            detrend = 1,
            reject=MEGreject,
            reject_by_annotation=False,
            preload=True,
            verbose=True)
    
    epochs.plot_drop_log()
    print(deriv_fname)
    epochs.save(deriv_fname, overwrite=True)


# MAIN
if __name__ == "__main__":
    main()