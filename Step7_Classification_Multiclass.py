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
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)
print(bids_path.basename,bids_path.fpath)

#%%
## Reading and preparing the trial based data
epochs = mne.read_epochs(bids_path.fpath,
                         preload=True,
                         verbose=True).pick(['meg'])

epochs


## Filter and downsample the data
epochs_rs = epochs.copy().filter(0,10)
epochs_rs.resample(100)
epochs_rs.crop(tmin=-0.1, tmax=1)

#%%
### pick conditions
XX = epochs_rs['Body', 'Face', 'Object', 'Scene', 'Word']
X = XX.get_data(picks='meg')  # n_trial by n_chan by time
y = XX.events[:, 2] # n_trial
print(y)

#%% LDA
### initialize pipeline
clf = LinearDiscriminantAnalysis()
#clf = make_pipeline(Vectorizer(),StandardScaler(),  
#                   LinearModel(sklearn.svm.SVC(kernel = 'linear')))                          
time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='accuracy', verbose=True)
#%%
## Performing the classification ##
# The classification will be performed timepoint by timepoint using a SVM by training on 80% of the trials on test on 20% in 5 runs. 
# This  results in a 5-fold cross-validation (*cv=5*). The output will be reported as Area Under the Curve (AUC). 

scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)

# plot results
scores = np.mean(scores, axis=0)

fig, ax = plt.subplots()
#plt.ylim([0.35, 1])
ax.plot(epochs_rs.times, scores, label='score')
ax.axhline(1/5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('accuracy')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

#%% Temporal generalization
# define the Temporal generalization object
clf = LinearDiscriminantAnalysis()
time_gen = GeneralizingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=None)
print(scores.shape)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(epochs_rs.times, np.diag(scores), label="score")
ax.axhline(1/5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("accuracy")
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Decoding MEG sensors over time")

fig, ax = plt.subplots(1, 1)
im = ax.imshow(
    scores,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=epochs_rs.times[[0,-1, 0,-1]],
    vmin=0.0,
    vmax=0.5,
)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("accuracy")

#%% Classify across channels
### pick conditions
XX = epochs_rs['Body', 'Face', 'Object', 'Scene', 'Word']
X = XX.get_data(picks='meg')  # n_trial by n_chan by time
y = XX.events[:, 2] # n_trial
print(y)
X = X.transpose(0, 2, 1)
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
scores = np.mean(scores, axis=0)

# Pick megnotometer channels
magnotometer_picks = mne.pick_types(epochs_rs.info, meg='mag')
# Plot the statistics on the sensor layout
pos = mne.find_layout(epochs.info, ch_type='mag').pos
mask = scores[magnotometer_picks] > 0.4
mask_params = dict(markersize=10, markerfacecolor="y")
fig,ax = plt.subplots(ncols=1)
im,cm = mne.viz.plot_topomap(scores[magnotometer_picks], pos, axes=ax, ch_type='mag',mask=mask, mask_params=mask_params)
# manually fiddle the position of colorbar
ax_x_start = 0.8
ax_x_width = 0.04
ax_y_start = 0.2
ax_y_height = 0.7
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
plt.title('megnotometer')
plt.colorbar()
plt.show()

        
# Pick gradiometer channelsclb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar
gradiometer_picks = mne.pick_types(epochs_rs.info, meg='grad')
gradiometer_labels = [epochs_rs.ch_names[pick] for pick in gradiometer_picks]
pos = mne.find_layout(epochs.info, ch_type='grad').pos
mask = scores[gradiometer_picks] > 0.4
mask_params = dict(markersize=10, markerfacecolor="y")
# Plot the statistics on the sensor layout
fig,ax = plt.subplots(ncols=1)
im,cm = mne.viz.plot_topomap(scores[gradiometer_picks], pos, axes=ax, ch_type='grad',mask=mask, mask_params=mask_params)
# manually fiddle the position of colorbar
ax_x_start = 0.8
ax_x_width = 0.04
ax_y_start = 0.2
ax_y_height = 0.7
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
plt.title('gradiometer')
plt.colorbar()
plt.show()
#%% RSA
clf = make_pipeline(StandardScaler(),  
                   LinearModel(sklearn.svm.SVC(kernel = 'linear')))             

#%% Common spatial pattern
# cannot converge
csp = CSP(n_components=3, norm_trace=False)
clf_csp = make_pipeline(csp, LinearModel(LogisticRegression(solver="liblinear")))
scores = cross_val_multiscore(clf_csp, X, y, cv=5, n_jobs=None)
print(f"CSP: {100 * scores.mean():0.1f}%")


csp.fit(X, y)
csp.plot_patterns(epochs.info)
csp.plot_filters(epochs.info, scalings=1e-9)
