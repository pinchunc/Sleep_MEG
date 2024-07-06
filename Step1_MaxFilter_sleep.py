## This code detects bad sensors automatically for each sleep block and apply all bad sensors to all blocks when doing maxfilter
import os.path as op
import os
import sys
import numpy as np

import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import scipy

# import settings
from FLUXSettings import bids_root,subject,task_list,session
task = 'sleep'

# General advice (from CBU):   
#* don't use ``'trans'`` with ``'movecomp'``
#* don't use ``'autobad'`` with ``'headpos'`` or ``'movecomp'``
#* don't use ``'autobad'`` with ``'st'``

# https://imaging.mrc-cbu.cam.ac.uk/meg/Maxfilter
# https://imaging.mrc-cbu.cam.ac.uk/meg/maxbugs
    
# check how to add headpos movecomp multistage
#%%
def main():
    
    # Read the raw1 data for destination file:
    raw1 = read_raw_bids(bids_path=BIDSPath(subject=subject, session=session,
                task='loc', run=1, suffix='meg', extension='.fif', root=bids_root), 
                     extra_params={'preload':True},
                     verbose=True)
    destination = raw1.info["dev_head_t"]
    
    files = os.listdir(op.join(bids_root,f'sub-{subject}',f'ses-{session}','meg'))
    sleep_files = [file for file in files if file.endswith('.fif') & file.startswith(f'sub-{subject}_ses-{session}_task-{task}_r')]

    auto_bad_list = [None] * len(sleep_files)

    for run_idx in range(len(sleep_files)):
        run = run_idx+1
    
        ## Define path to the local data and then define the file names:
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, suffix='meg', extension='.fif', root=bids_root)       
    
        crosstalk_file   = bids_path.meg_crosstalk_fpath
        calibration_file = bids_path.meg_calibration_fpath
    
       
        ### Idenfity the faulty sensors
        # The following scripts automatically identify the faulty sensors:
        raw = read_raw_bids(bids_path=bids_path, 
                      extra_params={'preload':True},
                      verbose=True)
        
        raw.info['bads'] = []
        raw_copy = raw.copy()  # copy to make sure we are not overwriting the raw data
        auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
            raw_copy, 
            cross_talk=crosstalk_file,
            calibration=calibration_file,
            return_scores=True, 
            verbose=True)
        
        print('noisy =', auto_noisy_chs)
        print('flat =', auto_flat_chs)
    
        auto_bad_list[run-1] = auto_noisy_chs + auto_flat_chs # for python index
    
    # save bad sensors info to a txt file
    deriv_root = op.join(bids_root, 'derivatives/preprocessing')  # output path
    auto_bad_path = op.join(deriv_root,f'sub-{subject}',f'ses-{session}','meg')
    
    with open(f'{auto_bad_path}/{task}_auto_bad_list.txt', 'w') as file:
        for line in auto_bad_list:
            file.write(str(line) + '\n')

##%        
    for run_idx in range(len(sleep_files)):
        run = run_idx+1
        ## Define path to the local data and then define the file names:
        bids_path = BIDSPath(subject=subject, session=session,
                    task=task, run=run, suffix='meg', extension='.fif', root=bids_root)

        deriv_path = BIDSPath(subject=subject, session=session, datatype='meg',
                    task=task, run=run, suffix='meg', root=deriv_root).mkdir()
    
        deriv_fname = bids_path.basename.replace('meg', 'raw_sss') # output filename
        deriv_file  = op.join(deriv_path.directory, deriv_fname)
    
        ### load raw file
        raw = read_raw_bids(bids_path=bids_path, 
                      extra_params={'preload':True},
                      verbose=True)
        
        # time-resolved information on active HPI coils
        # chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        # chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        # head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)

        ## add the bad sensor labels
        with open(f'{auto_bad_path}/{task}_auto_bad_list.txt', 'r') as file:
            lines = file.readlines()
            
        flattened_list = []
        for line in lines:
            # Ignore 'None' lines
            if 'None' in line:
                continue
            # Convert string representation of list to an actual list
            elements = eval(line.strip())
            flattened_list.extend(elements)
        
        raw.info['bads'].extend(set(flattened_list))
        print('bads =', raw.info['bads'])  
    
        # Change MEGIN magnetometer coil types (type 3022 and 3023 to 3024) to ensure compatibility across systems.
        raw.fix_mag_coil_types()
        
        ## Apply the Maxfilter and calibration
        # Apply the algorithm performing the Maxfiltering, SSS, calibration and cross-talk reduction:
        raw_sss = mne.preprocessing.maxwell_filter(raw,
                                                destination=destination,
                                                st_duration=10,
                                                cross_talk=crosstalk_file,                                          
                                                calibration=calibration_file,
                                                #head_pos = head_pos,
                                                verbose=True)
        # Plot the power spectra of raw data:
        fig_raw = raw.compute_psd(fmax=60, n_fft=1000).plot();
        fig_raw.savefig(f'{deriv_root}/sub-{subject}/ses-{session}/meg/sub-{subject}_task-{task}_run-0{run}_power_raw.png', dpi=300)
        
        # Compared them to the power spectra after the application of the noise reduction algorithms:
        fig_sss = raw_sss.compute_psd(fmax=60, n_fft=1000).plot();
        fig_sss.savefig(f'{deriv_root}/sub-{subject}/ses-{session}/meg/sub-{subject}_task-{task}_run-0{run}_power_sss.png', dpi=300)
        
        # Save the result in a FIF-file:
        raw_sss.save(deriv_file, overwrite=True)

# MAIN

if __name__ == "__main__":
    main()
