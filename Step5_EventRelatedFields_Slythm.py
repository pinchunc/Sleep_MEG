import os.path as op
import os
import matplotlib
import numpy as np
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
deriv_file_spindles = bids_path.basename.replace('run-01', 'run-all-spindles')
deriv_fname_spindles = op.join(bids_path.directory, deriv_file_spindles)

epochs = mne.read_epochs(deriv_fname_spindles,
                         proj = False,
                         preload=True,
                         verbose=True)

#%% Averaging the trial data
evoked_spindles= epochs['spindles_events'].copy().average(method='mean').filter(0.0, 30).crop(-0.5,0.5)

#%% Plotting event-related fields
epochs['spindles_events'].copy().filter(0.0,30).crop(-0.5,0.5).plot_image(picks=['MEG1911'], vmin=-500, vmax=500)

#%% Plot Magnetometers
evoked_spindles.copy().pick('mag').plot_topo(title = 'Magnetometers')

#%% Plot Gradiometers
evoked_spindles.copy().pick('grad').plot_topo(title='Gradiometers', merge_grads=True)

#%%
evoked_spindles.copy().pick('eeg').plot_topo(title='EEG')

#%% To plot a topographic map of the response at 110 ms 
times = np.arange(-0.5,0.5,0.1)
evoked_spindles.plot_topomap(times, ch_type='mag', time_unit='s')
evoked_spindles.plot_topomap(times, ch_type='grad', time_unit='s')

#%%

deriv_file_SO = bids_path.basename.replace('run-01', 'run-all-SO')  # run 12 -> run 01 concatenated with run 02
deriv_fname_SO = op.join(bids_path.directory, deriv_file_SO)

epochs = mne.read_epochs(deriv_fname_SO,
                         proj = False,
                         preload=True,
                         verbose=True)

#%% Averaging the trial data
evoked_SO = epochs['SO_events'].copy().average(method='mean').filter(0.0, 10).crop(-1,1)

#%% Plotting event-related fields
epochs['SO_events'].copy().filter(0.0,10).crop(-1,1).plot_image(picks=['MEG1911'], vmin=-500, vmax=500)

#%% Plot Magnetometers
evoked_SO.copy().pick('mag').plot_topo(title = 'Magnetometers')

#%% Plot Gradiometers
evoked_SO.copy().pick('grad').plot_topo(title='Gradiometers', merge_grads=True)

#%%
evoked_SO.copy().pick('eeg').plot_topo(title='EEG')

#%% To plot a topographic map of the response at 110 ms 
times = np.arange(-0.5,0.5,0.1)
evoked_SO.plot_topomap(times, ch_type='mag', time_unit='s')
evoked_SO.plot_topomap(times, ch_type='grad', time_unit='s')


