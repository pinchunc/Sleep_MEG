from mne import bem
import os.path as op

subject_dir= '/Applications/freesurfer/7.3.2/subjects'
subject = 's107'

bem.make_scalp_surfaces(subject, subject_dir, overwrite=True)
bem.make_watershed_bem(subject, subjects_dir=subject_dir, overwrite=True)