import os
import os.path as op
import numpy as np

import mne
from mne_bids import BIDSPath
from mne.cov import compute_covariance
from mne.beamformer import make_lcmv, apply_lcmv_cov, make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet
from mne.datasets import fetch_fsaverage

# import settings
from FLUXSettings import bids_root,subject,task_list,session
#matplotlib.use('Qt5Agg')
get_ipython().run_line_magic("matplotlib", "qt")

#%%
task    = 'loc'
run     = '12'  
meg_suffix = 'meg'
epo_suffix = 'epo'
mri_suffix = 'T1w'
bem_suffix = 'bem-sol'
src_suffix = 'src'
fwd_suffix = 'fwd'
deriv_root   = op.join(bids_root, 'derivatives/analysis')

bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)
print(bids_path.basename,bids_path.fpath)
#%%
fwd_fname = bids_path.basename.replace(epo_suffix, fwd_suffix)
fwd_file = op.join(bids_path.directory, fwd_fname)
src_file = fwd_file.replace(fwd_suffix, src_suffix)
print(src_file)

fs_subject = f'sub-{subject}'
fs_root = op.join(bids_root, 'derivatives/FreeSurfer')
print(fs_root) 
#%%
# load forward model
fwd = mne.read_forward_solution(fwd_file)
# load epochs
epochs = mne.read_epochs(bids_path.fpath, proj = False)
epochs = epochs.pick_types(meg='grad')
## need to think about how to epoch sleep data
epochs_face  = epochs[['Face']].filter(8, 12).copy()
epochs_scene = epochs[['Scene']].filter(8, 12).copy()

#%% Calculating the covariance matrices 
rank = mne.compute_rank(epochs, tol=1e-6, tol_kind='relative')

baseline_face_cov = compute_covariance(epochs_face,tmin=-0.75,tmax=-0.1,method='shrunk',rank=rank,n_jobs = 4,verbose=True)
active_face_cov = compute_covariance(epochs_face,tmin=0,tmax=1,method='shrunk',rank=rank,n_jobs = 4,verbose=True)

baseline_scene_cov = compute_covariance(epochs_scene,tmin=-0.75,tmax=-0.1,method='shrunk',rank=rank,n_jobs = 4,verbose=True)
active_scene_cov = compute_covariance(epochs_scene,tmin=0,tmax=1,method='shrunk',rank=rank,n_jobs = 4,verbose=True)

common_cov = baseline_face_cov + active_face_cov + baseline_scene_cov + active_scene_cov 

common_cov.plot(epochs_face.info)

#%% Note
# Source estimates using beamforming approaches (LCMV and DICS) will have an increase in noise bias towards the centre of the head (Gross et al., 2001; Van Veen et al., 1997). 
# This bias is best ‘subtracted out’ by comparing conditions relatively. 

## Handling depth bias
# mne.beamformer.make_lcmv() has a depth parameter that normalizes the forward model prior to computing the spatial filters. 
# If float (default 0.8), it acts as the depth weighting exponent (exp) to use None is equivalent to 0, meaning no depth weighting is performed.
# Unit-noise gain beamformers handle depth bias by normalizing the weights of the spatial filter. Choose this by setting weight_norm='unit-noise-gain'.
#%% Make LCMV
# Source reconstruction with several sensor types requires a noise covariance matrix to be able to apply whitening.
filters = make_lcmv(epochs.info, 
                            fwd, 
                            common_cov, 
                            reg=0.05,
                            noise_cov=None, 
                            pick_ori='max-power', # pick_ori="vector"
                            weight_norm='unit-noise-gain')
#%%
#stc = apply_lcmv(evoked, filters)
#stc_vec = apply_lcmv(evoked, filters_vec)
#del filters, filters_vec
#%%
stc_face_base = apply_lcmv_cov(baseline_face_cov, filters)
stc_face_act = apply_lcmv_cov(active_face_cov, filters)
stc_scene_base = apply_lcmv_cov(baseline_scene_cov, filters)
stc_scene_act = apply_lcmv_cov(active_scene_cov, filters)
stc_rel = ((stc_face_act + stc_scene_act) - (stc_face_base + stc_scene_base)) / (stc_face_base + stc_scene_base)
stc_RvsL = (stc_face_act - stc_scene_act) / (stc_face_act + stc_scene_act)

#%% Plot
src_fs = mne.read_source_spaces(src_file)
#%%
stc_rel.plot(src=fwd['src'],subject=fs_subject, # hemi = 'both' , views = 'parietal', surface = 'inflated',  
                        subjects_dir=fs_root);
#%%
stc_RvsL.plot(src=fwd['src'],subject=fs_subject, subjects_dir=fs_root);
#stc_RvsL.plot(src=fwd['src'], hemi = 'both' , views = 'parietal', surface = 'inflated',  subject=fs_subject,
#                        subjects_dir=fs_root);