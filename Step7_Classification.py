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

from FLUXSettings import bids_root,subject,task,session
matplotlib.use('Qt5Agg')


run = '12'  
meg_suffix = 'meg'
epo_suffix = 'epo'
preproc_root = op.join(bids_root, 'derivatives/preprocessing')
deriv_root   = op.join(bids_root, 'derivatives/analysis')

bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)
print(bids_path.basename,bids_path.fpath)

## Reading and preparing the trial based data
epochs = mne.read_epochs(bids_path.fpath,
                         preload=True,
                         verbose=True).pick(['meg'])

epochs

## Filter and downsample the data
epochs_rs = epochs.copy().filter(0,10)
epochs_rs.resample(100)
epochs_rs.crop(tmin=-0.1, tmax=1)

### pick conditions
XX = epochs_rs['Object','Face']
X = XX.get_data(picks='meg') 
y = XX.events[:, 2]
print(y)


### initialize pipeline
clf = make_pipeline(Vectorizer(),StandardScaler(),  
                   LinearModel(sklearn.svm.SVC(kernel = 'linear')))                          
time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)


## Performing the classification ##
# The classification will be performed timepoint by timepoint using a SVM by training on 80% of the trials on test on 20% in 5 runs. 
# This  results in a 5-fold cross-validation (*cv=5*). The output will be reported as Area Under the Curve (AUC). 

scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)

# plot results
scores = np.mean(scores, axis=0)

fig, ax = plt.subplots()
plt.ylim([0.35, 1])
ax.plot(epochs_rs.times, scores, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()