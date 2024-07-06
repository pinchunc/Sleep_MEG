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
from FLUXSettings import bids_root,subject,task,session
#matplotlib.use('Qt5Agg')
get_ipython().run_line_magic("matplotlib", "qt")
#%%

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


fwd_fname = bids_path.basename.replace(epo_suffix, fwd_suffix)
fwd_file = op.join(bids_path.directory, fwd_fname)
src_file = fwd_file.replace(fwd_suffix, src_suffix)
bem_file = fwd_file.replace(fwd_suffix, bem_suffix)


fs_subject = 'sub-'+subject
fs_root = op.join(bids_root, 'derivatives', 'FreeSurfer')

print(bids_path.basename,bids_path.fpath)
print(src_file)
print(fs_root) 
print(fwd_file)
#%%
# load forward model
fwd = mne.read_forward_solution(fwd_file)
# load epochs
epochs = mne.read_epochs(bids_path.fpath, proj = False)

## Source modeling of modulations of alpha band activity 
csd_aface_base   = csd_multitaper(epochs['Face'],  fmin=8, fmax=12, tmin=-0.75, tmax=-0.4, bandwidth = 3, low_bias = True, verbose = False, n_jobs=-1)
csd_ascene_base  = csd_multitaper(epochs['Scene'], fmin=8, fmax=12, tmin=-0.75, tmax=-0.4, bandwidth = 3, low_bias = True,verbose = False, n_jobs=-1)

csd_aface_pre   = csd_multitaper(epochs['Face'],  fmin=8, fmax=12, tmin=-0.4, tmax=0, bandwidth = 3, low_bias = True, verbose = False, n_jobs=-1)
csd_ascene_pre  = csd_multitaper(epochs['Scene'], fmin=8, fmax=12, tmin=-0.4, tmax=0, bandwidth = 3, low_bias = True,verbose = False, n_jobs=-1)

csd_aface_post =  csd_multitaper(epochs['Face'],  fmin=8, fmax=12, tmin=0, tmax= 1, bandwidth = 3,  low_bias = True, verbose = False, n_jobs=-1)
csd_ascene_post = csd_multitaper(epochs['Scene'], fmin=8, fmax=12, tmin=0, tmax= 1, bandwidth = 3,  low_bias = True, verbose = False, n_jobs=-1)

#%%
csd_acommon = csd_aface_pre
csd_anoise =  csd_aface_pre
csd_acommon._data = (csd_aface_pre._data + csd_ascene_pre._data + 
                         csd_aface_post._data + csd_ascene_post._data)/4

csd_anoise._data = (csd_aface_base._data + csd_ascene_base._data)/2

## For calculating the spatial filters we also need to derive the rank of the CSD. It will be similar to the the rank of the coveriance matrix: 
rank = mne.compute_rank(epochs, tol=1e-6, tol_kind='relative', proj = False)
print(rank)


MEGrank = {k: rank[k] for k in list(rank)[:1]}

afilters = make_dics(epochs.info, fwd, csd_acommon.mean() , noise_csd=csd_anoise.mean(), 
                    reg=0, pick_ori='max-power', reduce_rank=True, real_filter=True, rank=MEGrank, depth = 0)

stc_aface_post, freqs = apply_dics_csd(csd_aface_post.mean(), afilters)
stc_ascene_post, freqs  = apply_dics_csd(csd_ascene_post.mean(), afilters)
stc_aface_pre, freqs = apply_dics_csd(csd_aface_pre.mean(), afilters)
stc_ascene_pre, freqs  = apply_dics_csd(csd_ascene_pre.mean(), afilters)

stc_arel = ((stc_aface_post + stc_ascene_post) - (stc_aface_pre + stc_ascene_pre)) / (stc_aface_pre + stc_ascene_pre)
stc_aFvsS = (stc_aface_post - stc_ascene_post) / (stc_aface_post + stc_ascene_post)

## ploting
src = fwd['src']
stc_arel.plot(src=src, subject=fs_subject, subjects_dir=fs_root, mode='stat_map', 
              verbose = True)

stc_aFvsS.plot(src=src, subject=fs_subject, subjects_dir=fs_root, mode='stat_map')
#%%

# Source modeling of modulations of gamma band activity
print('Pre')
csd_gface_base   = csd_multitaper(epochs['Face'],  fmin=60, fmax=90, tmin=-0.75, tmax=0, bandwidth = 8, adaptive=True, low_bias = True,verbose = False, n_jobs=4)
csd_gscene_base  = csd_multitaper(epochs['Scene'], fmin=60, fmax=90, tmin=-0.75, tmax=0, bandwidth = 8, adaptive=True, low_bias = True,verbose = False, n_jobs=4)

print('Stim')
csd_gface_post =  csd_multitaper(epochs['Face'],  fmin=60, fmax=90, tmin=0, tmax= 0.75, bandwidth = 8, adaptive=True, low_bias = True, verbose = False, n_jobs=4)
csd_gscene_post = csd_multitaper(epochs['Scene'], fmin=60, fmax=90, tmin=0, tmax= 0.75, bandwidth = 8, adaptive=True, low_bias = True, verbose = False, n_jobs=4)

csd_gcommon = csd_gface_post
csd_gnoise =  csd_gface_base

csd_gcommon._data = (csd_gface_post._data + csd_gscene_post._data)/2
csd_gnoise._data  = (csd_gface_base._data + csd_gscene_base._data)/2

gfilters = make_dics(epochs.info, fwd, csd_gcommon.mean() , noise_csd=csd_gnoise.mean(), reg=0, pick_ori='max-power',
                        reduce_rank=True, real_filter=True, rank=MEGrank, depth = 0)

stc_gface_post, freqs = apply_dics_csd(csd_gface_post.mean(), gfilters)
stc_gscene_post, freqs  = apply_dics_csd(csd_gscene_post.mean(), gfilters)

stc_aFvsS = (stc_gface_post - stc_gscene_post) / (stc_gface_post + stc_gscene_post)

src = fwd['src']
stc_aFvsS.plot(src=src, subject=fs_subject, subjects_dir=fs_root, mode='stat_map')
stc_aFvsS.plot(src=src, subject=fs_subject, subjects_dir=fs_root, 
              mode='stat_map',clim=dict(kind='value', pos_lims=[0.,0.025, 0.05],
                                        neg_lims=[-0.05, -0.025, 0]));

