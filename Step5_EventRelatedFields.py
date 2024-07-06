import os.path as op
import os
import matplotlib
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids 

# import settings
from FLUXSettings import bids_root,subject,task,session
#matplotlib.use('Qt5Agg')
get_ipython().run_line_magic("matplotlib", "qt")

run     = '12'  # both runs compbined
meg_suffix = 'meg'
epo_suffix = 'epo'

preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)
print(bids_path.basename,bids_path.fpath)

epochs = mne.read_epochs(bids_path.fpath,
                         proj = False,
                         preload=True,
                         verbose=True)

#%%
# Averaging the trial data
evoked_word= epochs['Word'].copy().average(method='mean').filter(0.0, 30).crop(-0.1,0.4)

#%%
# Plotting event-related fields
epochs['Word'].copy().filter(0.0,30).crop(-0.1,0.4).plot_image(picks=['MEG1911'], vmin=-500, vmax=500)
#%%
# Plot Magnetometers
evoked_word.copy().apply_baseline(baseline=(-0.1, 0))
evoked_word.copy().pick('mag').plot_topo(title = 'Magnetometers')
#%%
# Plot Gradiometers
evoked_word.copy().apply_baseline(baseline=(-0.1, 0))
evoked_word.copy().pick('grad').plot_topo(title='Gradiometers', merge_grads=True)

#%%
# To plot a topographic map of the response at 110 ms 
times = np.arange(0.05, 0.2, 0.05)
evoked_word.plot_topomap(times, ch_type='mag', time_unit='s')
evoked_word.plot_topomap(times, ch_type='grad', time_unit='s')