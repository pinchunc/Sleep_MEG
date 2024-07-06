import os.path as op
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import mne
from mne_bids import BIDSPath, read_raw_bids

# import settings
from FLUXSettings import bids_root,subject,task_list,session
#matplotlib.use('Qt5Agg')
get_ipython().run_line_magic("matplotlib", "qt")

task = 'sleep'
meg_suffix = 'meg'
epo_suffix = 'epo'

preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=1, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)
print(bids_path.basename,bids_path.fpath)
#%%
deriv_file_spindles = bids_path.basename.replace('run-01', 'run-all-spindles')  # run 12 -> run 01 concatenated with run 02
deriv_fname_spindles = op.join(bids_path.directory, deriv_file_spindles)

epochs = mne.read_epochs(deriv_fname_spindles,
                         proj = False,
                         preload=True,
                         verbose=True)

#epochs.copy().filter(0.0,30).crop(-0.5,0.5).picks=['mag', 'grad']

#%%
freqs = np.arange(0.3, 30.3, 1)
n_cycles = freqs / 2 
time_bandwidth = 2.0

tfr_spindles =  mne.time_frequency.tfr_multitaper(
    epochs['spindles_events'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

tfr_spindles_surr =  mne.time_frequency.tfr_multitaper(
    epochs['spindles_surr_events'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

#%% Comparing real vs surrogate
tfr_spindles_diff = tfr_spindles.copy()
tfr_spindles_diff.data = (tfr_spindles.data - tfr_spindles_surr.data)/(tfr_spindles.data + tfr_spindles_surr.data)
tfr_spindles_diff.plot_topo(
    tmin=-0.5, tmax=0.5, 
    fig_facecolor='w',
    font_color='k',
    title='spindles - surr');

#%%
tfr_spindles_diff.plot(
    picks=['MEG1812'], 
    mode="percent", 
    tmin=-1, tmax=1,
    title='MEG1812', 
    vmin=-0.75, vmax=0.75)  

#%%
tfr_spindles.plot(
    picks=['MEG1812'], 
    baseline=[-2,-1.5], 
    mode="percent", 
    tmin=-1, tmax=1,
    title='MEG1812', 
    vmin=-0.75, vmax=0.75)  

#%%
plt = tfr_spindles.plot_topo(
    tmin=-0.5, tmax=0.5, 
    baseline=[-1,-0.5], 
    mode="percent", 
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title='TFR of spindles')

#%%
tfr_spindles.plot_topomap(
    tmin=-0.5, tmax=0.5, 
    fmin=12, fmax=16,
    baseline=[-1,-0.5], 
    mode="percent");
#plot_compare_evokeds

#%% spindle-locked ripples
freqs = np.arange(80, 120, 5)
n_cycles = freqs / 2 
time_bandwidth = 2.0

tfr_spindles_ripples =  mne.time_frequency.tfr_multitaper(
    epochs['spindles_events'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

tfr_spindles_surr_ripples =  mne.time_frequency.tfr_multitaper(
    epochs['spindles_surr_events'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

#%% Comparing real vs surrogate
tfr_spindles_locked_ripples_diff = tfr_spindles_ripples.copy()
tfr_spindles_locked_ripples_diff.data = (tfr_spindles_ripples.data - tfr_spindles_surr_ripples.data)/(tfr_spindles_ripples.data + tfr_spindles_surr_ripples.data)
tfr_spindles_locked_ripples_diff.plot_topo(
    tmin=-0.5, tmax=0.5, 
    fig_facecolor='w',
    font_color='k',
    title='spindles - surr');

#%% SO
deriv_file_SO = bids_path.basename.replace('run-01', 'run-all-SO')  # run 12 -> run 01 concatenated with run 02
deriv_fname_SO = op.join(bids_path.directory, deriv_file_SO)

epochs = mne.read_epochs(deriv_fname_SO,
                         proj = False,
                         preload=True,
                         verbose=True)

epochs_filt = epochs.copy().filter(0,30).crop(-0.5,0.5).picks=['mag', 'grad']

#%%
freqs = np.arange(0.3, 30.3, 1)
n_cycles = freqs / 2 
time_bandwidth = 2.0

tfr_SO =  mne.time_frequency.tfr_multitaper(
    epochs['SO_events'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

tfr_SO_surr =  mne.time_frequency.tfr_multitaper(
    epochs['SO_surr_events'], 
    freqs=freqs, 
    n_cycles=n_cycles,
    time_bandwidth=time_bandwidth, 
    picks = 'grad',
    use_fft=True, 
    return_itc=False,
    average=True, 
    decim=2,
    n_jobs = -1,
    verbose=True)

#%% Comparing real vs surrogate
tfr_SO_diff = tfr_SO.copy()
tfr_SO_diff.data = (tfr_SO.data - tfr_SO_surr.data)/(tfr_SO.data + tfr_SO_surr.data)
tfr_SO_diff.plot_topo(
    tmin=-0.5, tmax=1.0, 
    fig_facecolor='w',
    font_color='k',
    title='SO - surr');

tfr_SO_diff.plot_topomap(tmin=-1, tmax=1, fmin=0.3, fmax=2);
tfr_SO_diff.plot_topomap(tmin=0.2, tmax=0.8, fmin=12, fmax=16); # fast spindle
tfr_SO_diff.plot_topomap(tmin=-0.4, tmax=0, fmin=8, fmax=12); # slow spindle

tfr_SO_diff.plot(
    picks=['MEG1812'], 
    mode="percent", 
    tmin=-1, tmax=1,
    title='MEG1812', 
    vmin=-0.75, vmax=0.75)  


#%%
tfr_SO.plot(
    picks=['MEG1812'], 
    baseline=[-2,-1.5], 
    mode="percent", 
    tmin=-1, tmax=1,
    title='MEG1812', 
    vmin=-0.75, vmax=0.75)  
#%%
plt = tfr_SO.plot_topo(
    tmin=-1, tmax=1.0, 
    baseline=[-2,-1.5], 
    mode="percent", 
    fig_facecolor='w',
    font_color='k',
    vmin=-1, vmax=1,
    title='TFR of SO')

#%%
tfr_SO.plot_topomap(
    tmin=-1, tmax=1, 
    fmin=0.5, fmax=2,
    baseline=[-2,-1.5], 
    mode="percent");

