import os
import os.path as op
import numpy as np
import matplotlib
import mne
from mne_bids import BIDSPath
from mne.cov import compute_covariance
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_multitaper

# import settings
from FLUXSettings import bids_root,subject,task_list,session
#matplotlib.use('Qt5Agg')
get_ipython().run_line_magic("matplotlib", "qt")
#%%
run        = '12'  
task       = 'sleep'
meg_suffix = 'meg'
epo_suffix = 'epo'
mri_suffix = 'T1w'
bem_suffix = 'bem-sol'
src_suffix = 'src'
fwd_suffix = 'fwd'
deriv_root   = op.join(bids_root, 'derivatives/analysis')

bids_path = BIDSPath(subject=subject, session=session,
            task='loc', run=12, suffix=epo_suffix, datatype='meg', # use loc task's forward model
            root=deriv_root, extension='.fif', check=False)

bids_sleep = BIDSPath(subject=subject, session=session,
            task='sleep', run=1, suffix=epo_suffix, datatype='meg', # use loc task's forward model
            root=deriv_root, extension='.fif', check=False)

deriv_file_spindles = bids_sleep.basename.replace('run-01', 'run-all-spindles')
deriv_fname_spindles = op.join(bids_sleep.directory, deriv_file_spindles)
deriv_file_SO = bids_sleep.basename.replace('run-01', 'run-all-SO')
deriv_fname_SO = op.join(bids_sleep.directory, deriv_file_SO)

fwd_fname = bids_path.basename.replace(epo_suffix, fwd_suffix)
fwd_file = op.join(bids_path.directory, fwd_fname)
src_file = fwd_file.replace(fwd_suffix, src_suffix)
bem_file = fwd_file.replace(fwd_suffix, bem_suffix)


fs_subject = 'sub-'+subject
fs_root = op.join(bids_root, 'derivatives', 'FreeSurfer')

#%%
# load forward model
fwd = mne.read_forward_solution(fwd_file)
# load epochs
epochs_spindles = mne.read_epochs(deriv_fname_spindles,
                                  proj = False,
                                  preload=True,
                                  verbose=True)
#%% Source modeling of modulations of spindles band activity 
# cross-spectral density (CSD)
csd_spindles_base   = csd_multitaper(epochs_spindles['spindles_events'],  fmin=12, fmax=16, tmin=-1.5, tmax=-0.5, bandwidth = 3, low_bias = True, verbose = False, n_jobs=-1)
csd_spindles        = csd_multitaper(epochs_spindles['spindles_events'],  fmin=12, fmax=16, tmin=-0.5, tmax= 0.5, bandwidth = 3,  low_bias = True, verbose = False, n_jobs=-1)

csd_spindles_surr_base   = csd_multitaper(epochs_spindles['spindles_surr_events'],  fmin=12, fmax=16, tmin=-1.5, tmax=-0.5, bandwidth = 3, low_bias = True, verbose = False, n_jobs=-1)
csd_spindles_surr        = csd_multitaper(epochs_spindles['spindles_surr_events'],  fmin=12, fmax=16, tmin=-0.5, tmax= 0.5, bandwidth = 3,  low_bias = True, verbose = False, n_jobs=-1)

#%%
csd_common = csd_spindles
csd_noise  = csd_spindles
csd_common._data = ( csd_spindles._data + csd_spindles_surr._data ) / 2
csd_noise._data = ( csd_spindles_base._data + csd_spindles_surr_base._data) / 2

#%%
## For calculating the spatial filters we also need to derive the rank of the CSD. It will be similar to the the rank of the coveriance matrix: 
rank = mne.compute_rank(epochs_spindles, tol=1e-6, tol_kind='relative', proj = False)
print(rank)
MEGrank = {k: rank[k] for k in list(rank)[:1]}

#%%
filters = make_dics(epochs_spindles.info, fwd, csd_common.mean() , noise_csd=csd_noise.mean(), 
                    reg=0, pick_ori='max-power', reduce_rank=True, real_filter=True, rank=MEGrank, depth = 0)
#%%
stc_spindles, freqs = apply_dics_csd(csd_spindles.mean(), filters)
stc_spindles_base, freqs = apply_dics_csd(csd_spindles_base.mean(), filters)
stc_spindles_surr, freqs = apply_dics_csd(csd_spindles_surr.mean(), filters)

stc_spindles_rel = (stc_spindles - stc_spindles_base) / (stc_spindles + stc_spindles_base)
stc_spindles_surrl = (stc_spindles - stc_spindles_surr) / (stc_spindles_surr)

#%% ploting
src = fwd['src']
stc_spindles_rel.plot(src=src, subject=fs_subject, subjects_dir=fs_root, mode='stat_map', 
              verbose = True)
stc_spindles_surrl.plot(src=src, subject=fs_subject, subjects_dir=fs_root, mode='stat_map', 
              verbose = True)

##%

#%% Source modeling of modulations of SO band activity 
epochs_SO = mne.read_epochs(deriv_fname_SO,
                                  proj = False,
                                  preload=True,
                                  verbose=True)
# later add surrogate data as a different condition
# time windows have to be the same
csd_SO_base   = csd_multitaper(epochs_SO['SO_events'],  fmin=0.5, fmax=2, tmin=-2, tmax=-1, bandwidth = 3, low_bias = True, verbose = False, n_jobs=-1)
csd_SO        = csd_multitaper(epochs_SO['SO_events'],  fmin=0.5, fmax=2, tmin=-1, tmax= 1, bandwidth = 3,  low_bias = True, verbose = False, n_jobs=-1)
csd_SO_surr_base   = csd_multitaper(epochs_SO['SO_surr_events'],  fmin=0.5, fmax=2, tmin=-2, tmax=-1, bandwidth = 3, low_bias = True, verbose = False, n_jobs=-1)
csd_SO_surr        = csd_multitaper(epochs_SO['SO_surr_events'],  fmin=0.5, fmax=2, tmin=-1, tmax= 1, bandwidth = 3,  low_bias = True, verbose = False, n_jobs=-1)

#%%
csd_common = csd_SO
csd_noise  = csd_SO
csd_common._data = csd_SO._data
csd_noise._data = csd_SO_base._data

#%%
## For calculating the spatial filters we also need to derive the rank of the CSD. It will be similar to the the rank of the coveriance matrix: 
rank = mne.compute_rank(epochs_SO, tol=1e-6, tol_kind='relative', proj = False)
print(rank)

MEGrank = {k: rank[k] for k in list(rank)[:1]}

#%%
filters = make_dics(epochs_SO.info, fwd, csd_common.mean() , noise_csd=csd_noise.mean(), 
                    reg=0, pick_ori='max-power', reduce_rank=True, real_filter=True, rank=MEGrank, depth = 0)
#%%
stc_SO, freqs = apply_dics_csd(csd_SO.mean(), filters)
stc_SO_base, freqs = apply_dics_csd(csd_SO_base.mean(), filters)

stc_SO_rel = (stc_SO - stc_SO_base) / (stc_SO + stc_SO_base)

#%%
## ploting
src = fwd['src']
stc_SO_rel.plot(src=src, subject=fs_subject, subjects_dir=fs_root, mode='stat_map', 
              verbose = True)