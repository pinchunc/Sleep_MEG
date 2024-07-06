import os.path as op
import os
import sys
import numpy as np

import mne
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
import scipy

# import settings
from FLUXSettings import bids_root,subject,task,session
task = task[0]
run  = 1
#%%
def main():

    ## Define path to the local data and then define the file names:
    bids_path = BIDSPath(subject=subject, session=session,
                task=task, run=run, suffix='meg', extension='.fif', root=bids_root)

    deriv_root = op.join(bids_root, 'derivatives/preprocessing')  # output path

    deriv_path = BIDSPath(subject=subject, session=session, datatype='meg',
                task=task, run=run, suffix='meg', root=deriv_root).mkdir()

    deriv_fname  = bids_path.basename.replace('meg', 'raw_sss') # output filename
    deriv_file_1 = op.join(deriv_path.directory, deriv_fname)
    deriv_file_2 = deriv_file_1.replace('run-01', 'run-02')

    crosstalk_file   = bids_path.meg_crosstalk_fpath
    calibration_file = bids_path.meg_calibration_fpath
    print(crosstalk_file)
    print(calibration_file)

    # Read the raw data:
    raw1 = read_raw_bids(bids_path=bids_path, 
                     extra_params={'preload':True},
                     verbose=True)
    
    ### Idenfity the faulty sensors
    # The following scripts automatically identify the faulty sensors:
    
    raw1.info['bads'] = []
    raw1_copy = raw1.copy()  # copy to make sure we are not overwriting the raw data
    auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
        raw1_copy, 
        cross_talk=crosstalk_file, 
        calibration=calibration_file,
        return_scores=True, 
        verbose=True)
    
    print('noisy =', auto_noisy_chs)
    print('flat =', auto_flat_chs)

    # Set the noisy and flat sensors as 'bad' in the data set:
    raw1.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
    print('bads =', raw1.info['bads'])  

    # Change MEGIN magnetometer coil types (type 3022 and 3023 to 3024) to ensure compatibility across systems.
    raw1.fix_mag_coil_types()

    ## Apply the Maxfilter and calibration
    # Apply the algorithm performing the Maxfiltering, SSS, calibration and cross-talk reduction:
    destination = raw1.info["dev_head_t"]
    raw1_sss = mne.preprocessing.maxwell_filter(
        raw1,
        destination=destination,
        st_duration=10,
        cross_talk=crosstalk_file,
        calibration=calibration_file,
        verbose=True)
    
    # Plot the power spectra of raw data:
    raw1.compute_psd(fmax=60, n_fft=1000).plot();
    # Compared them to the power spectra after the application of the noise reduction algorithms:
    raw1_sss.compute_psd(fmax=60, n_fft=1000).plot();


    ## Repeat for 2. run
    bids_path.update(run='02')
    print(bids_path.basename)
    raw2 = read_raw_bids(bids_path=bids_path, 
                      extra_params={'preload':True},
                      verbose=True)
    
    # detect faulty sensors
    raw2.info['bads'] = []
    raw2_copy = raw2.copy()  # copy to make sure we are not overwriting the raw data
    auto2_noisy_chs, auto2_flat_chs, auto2_scores = mne.preprocessing.find_bad_channels_maxwell(
        raw2_copy, 
        cross_talk=crosstalk_file, 
        calibration=calibration_file,
        return_scores=True, 
        verbose=True)

    raw2.info['bads'].extend(auto_noisy_chs + auto_flat_chs + auto2_noisy_chs + auto2_flat_chs)
    raw1.info['bads'] = []
    raw1.info['bads'].extend(auto_noisy_chs + auto_flat_chs + auto2_noisy_chs + auto2_flat_chs)
    raw2.fix_mag_coil_types()

    destination = raw1.info["dev_head_t"]
    raw2_sss = mne.preprocessing.maxwell_filter(raw2,
                                             destination=destination,
                                             st_duration=10,
                                             cross_talk=crosstalk_file,                                          
                                             calibration=calibration_file,
                                             verbose=True)

    # Save the result in a FIF-file:
    raw2_sss.save(deriv_file_2, overwrite=True)
    raw1_sss.save(deriv_file_1, overwrite=True)


# MAIN

if __name__ == "__main__":
    main()
