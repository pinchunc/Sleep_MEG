import os.path as op
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne
from mne_bids import BIDSPath
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from FLUXSettings import bids_root,subject,task_list,session
matplotlib.use('Qt5Agg')

task = 'loc'
run = '12'  
meg_suffix = 'meg'
epo_suffix = 'epo'
fwd_suffix = 'fwd'
mri_suffix = 'T1w'

preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

#%%
## Reading and preparing the trial based data
bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)

epochs = mne.read_epochs(bids_path.fpath,
                         preload=True,
                         verbose=True).pick(['grad'])

epochs


## Filter and downsample the data
epochs_rs = epochs.copy().filter(0,10)
epochs_rs.resample(100)
epochs_rs.crop(tmin=-0.1, tmax=1)

#%%
### pick conditions
XX = epochs_rs['Face', 'Scene'] #'Body', 'Object', , 'Word'
X = XX.get_data(picks='meg')  # n_trial by n_chan by time
y = XX.events[:, 2] # n_trial
print(y)

#%% Decoding
# Projecting sensor-space patterns to source space
clf = make_pipeline(
    StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))
)
time_decod = SlidingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)
time_decod.fit(X, y)
#%%
coef = get_coef(time_decod, "patterns_", inverse_transform=True)
evoked_time_gen = mne.EvokedArray(coef, epochs_rs.info, tmin=epochs_rs.times[0])
joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
evoked_time_gen.plot_joint(
    times=np.arange(0.0, 0.500, 0.100), title="Face vs Scene", **joint_kwargs
)
#%%
cov = mne.compute_covariance(epochs_rs, tmax=0.0)
fwd_fname = bids_path.basename.replace(epo_suffix, fwd_suffix)
fwd_file = op.join(bids_path.directory, fwd_fname)
print("* Read Forward model: ",fwd_file)
fwd = mne.read_forward_solution(fwd_file)
#%%
inv = mne.minimum_norm.make_inverse_operator(evoked_time_gen.info, fwd, cov, loose="auto")#0.0)
stc = mne.minimum_norm.apply_inverse(evoked_time_gen, inv, 1.0 / 9.0, "dSPM")
src = inv["src"]
del fwd, inv
#%%
fs_sub_dir = '/Volumes/MEGMORI/BIDS-data/derivatives/FreeSurfer'
brain = stc.plot(src, initial_time=0.1, subjects_dir=fs_sub_dir) #hemi="split", views=("lat", "med"), 