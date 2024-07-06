import os.path as op
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from osl import preprocessing


# import settings
from FLUXSettings import bids_root,subject,task_list,session

#get_ipython().run_line_magic("matplotlib", "qt")
matplotlib.use('Qt5Agg')

## these are just for ICA not actually saving the data
notch_filter_freqs = (50,100) # must be lowe than lowpass
#l_freq = 1
#h_freq = 90 # for gamma
resample_sfreq = 200
#iir_params = dict(order=5, ftype='butter')
# downsample data and filtering
#raw_resmpl = raw.copy()
#raw_resmpl.resample(resample_sfreq)  # dowsample 
#raw_resmpl.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params = iir_params, verbose=True)
#raw_resmpl.notch_filter(notch_filter_freqs)


bad_segments = dict(segment_len=500, picks='meg')

# Set the local paths to the data:
meg_suffix = 'meg'
ann_suffix = 'ann'
ica_suffix = 'ica'
ann_suffix = 'ann'

deriv_root   = op.join(bids_root, 'derivatives/preprocessing')  # output path

def check_ica(raw, ica, save_dir, subject, task, run):
    os.makedirs(save_dir, exist_ok=True)

    # Find EOG and ECG correlations
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    ica_scores = ecg_scores + eog_scores

    # Barplot of ICA component "EOG match" and "ECG match" scores
    ica.plot_scores(ica_scores)
    plt.savefig(f"{save_dir}/sub-{subject}_task-{task}_run-0{run}_correl_plot.png", bbox_inches="tight")
    plt.close()

    # Plot bad components
    ica.plot_components(ica.exclude)
    plt.savefig(f"{save_dir}/sub-{subject}_task-{task}_run-0{run}_bad_components.png", bbox_inches="tight")
    plt.close()
    
def plot_psd(raw, save_dir):
    raw.compute_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4)).plot()
    plt.savefig(f"{save_dir}/sub-{subject}_task-{task}_run-0{run}_powspec.png", bbox_inches="tight")
    plt.close()


#%% Create figures to check ICA with
task = 'sleep'
files = os.listdir(op.join(bids_root,f'sub-{subject}',f'ses-{session}','meg'))
task_files = [file for file in files if file.endswith('.fif') & file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]

for run_idx in range(4,len(task_files)):
    run = run_idx+1
    print(run)
    bids_path = BIDSPath(subject=subject, session=session, datatype='meg',
            task=task, run=run, suffix=ann_suffix, 
            root=deriv_root, extension='.fif', check=False)
    
    deriv_fname = bids_path.basename.replace(ann_suffix, ica_suffix)
    deriv_file  = op.join(bids_path.directory, deriv_fname)
    
    print(bids_path)
    print(deriv_file)
    # Resampling and filtering of the raw data
    raw = read_raw_bids(bids_path=bids_path, 
                    extra_params={'preload':True},
                    verbose=True)
    
    raw_resmpl = raw.copy()
    raw_resmpl.resample(resample_sfreq)  # dowsample 
    raw_resmpl.filter(1, 40)
    #raw_resmpl.pick('meg').filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params = iir_params, verbose=True)
    
    # Applying the ICA 
    ica = ICA(method='fastica',
        random_state=97,
        n_components=30,
        verbose=True)
    
    ica.fit(raw_resmpl.pick(['meg','eog','ecg','emg']),verbose=True) #'emg' for sleep
    
    # Mark bad ICA components interactively
    preprocessing.plot_ica(ica, raw_resmpl, block=True)
    #ica.plot_sources(raw_resmpl, title='ICA',block=True); # left click on the component to be removed
    
    check_ica(raw_resmpl, ica, bids_path.directory, subject, task, run)
    
    print(ica.exclude)
    
    raw_ica = read_raw_bids(bids_path=bids_path,
                    extra_params={'preload':True},
                    verbose=True)
    ica.apply(raw_ica)
    raw_ica.save(deriv_file, overwrite=True)
            
    
    
