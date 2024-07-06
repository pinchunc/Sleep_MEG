import os.path as op
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA

# import settings
from FLUXSettings import bids_root,subject,task,session

#get_ipython().run_line_magic("matplotlib", "qt")
matplotlib.use('Qt5Agg')


def main():

    # Set the local paths to the data:
    meg_suffix = 'meg'
    ann_suffix = 'ann'
    ica_suffix = 'ica'
    ann_suffix = 'ann'

    deriv_root   = op.join(bids_root, 'derivatives/preprocessing')  # output path

    for run in ['01','02']:
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
        raw_resmpl = raw.copy().pick('meg')
        raw_resmpl.resample(200) # dowsample to 200 Hz
        raw_resmpl.filter(1, 40) # band-pass filtert from 1 to 40 Hz

        # Applying the ICA 
        ica = ICA(method='fastica',
            random_state=97,
            n_components=30,
            verbose=True)

        ica.fit(raw_resmpl,verbose=True)

        # Identifying ICA components reflecting artefacts
        ica.plot_sources(raw_resmpl, title='ICA',block=True);
        print( ica.exclude)
        
        raw_ica = read_raw_bids(bids_path=bids_path,
                        extra_params={'preload':True},
                        verbose=True)
        ica.apply(raw_ica)

        raw_ica.save(deriv_file, overwrite=True)

# MAIN
if __name__ == "__main__":
    main()
