import os.path as op
import os
import mne
import matplotlib
import nibabel

from mne_bids import BIDSPath, read_raw_bids
# import settings
from FLUXSettings import bids_root,subject,task_list,session

#get_ipython().run_line_magic("matplotlib", "qt")
matplotlib.use('Qt5Agg')

task = 'loc'
run        = '12'  
meg_suffix = 'meg'
mri_suffix = 'T1w'
epo_suffix = 'epo'
bem_suffix = 'bem-sol'
src_suffix = 'src'
fwd_suffix = 'fwd'
trans_suffix = 'trans'

deriv_root   = op.join(bids_root, 'derivatives/analysis')

fs_subject = 'sub-' + subject  # name of subject from freesurfer

# Files and directories for input
bids_path = BIDSPath(subject=subject, session=session,
            task=task, run=run, suffix=epo_suffix, datatype='meg',
            root=deriv_root, extension='.fif', check=False)

# only load the first run for head position
raw_path = bids_path.copy().update(root=bids_root, 
           suffix=meg_suffix, extension='.fif', run='01', check=False) 

print("\n*** Input files ***")
print("* Raw fiff-file (with digitization points) :", raw_path)
print("* Epoched file: ",bids_path)

mri_root = BIDSPath(subject=subject, session=session,
                    root=bids_root,  
                    datatype='anat', suffix=mri_suffix, extension='.nii')
print("* MRI file: ",mri_root) 


fs_root = op.join(bids_root, 'derivatives', 'FreeSurfer')
print("* Freesurfer directory: ",fs_root)

# Files and directories for generated output 
print("\n*** Output files ***")

fwd_fname = bids_path.basename.replace(epo_suffix, fwd_suffix)
fwd_file = op.join(bids_path.directory, fwd_fname)
print("* Forward model: ",fwd_file)

src_file = fwd_file.replace(fwd_suffix, src_suffix)
print("* Brain surface file: ",src_file)

trans_file = fwd_file.replace(fwd_suffix, trans_suffix)
print("* Tranformation file: ",trans_file)

bem_file = fwd_file.replace(fwd_suffix, bem_suffix)
print("* Boundary element file: ",bem_file)

#%%
# read MRI
t1_fname = f'{fs_root}/sub-{subject}/mri/T1.mgz'
t1 = nibabel.load(t1_fname)
t1.orthoview()
print(t1.affine)
#%%
## check parcellation from FreeSurfer
Brain = mne.viz.get_brain_class()

brain = Brain(fs_subject, 
              hemi='lh', 
              surf='pial',
              subjects_dir=fs_root, 
              size=(800, 600))

brain.add_annotation('aparc.a2009s', borders=False)

#%% BEM
conductivity = (0.3,) 
model = mne.make_bem_model(fs_subject,
                           ico=4,
                           conductivity=conductivity,
                           subjects_dir=fs_root)
bem = mne.make_bem_solution(model)

mne.write_bem_solution(bem_file,
                       bem, overwrite=True)
#%%
mne.viz.plot_bem(subject=fs_subject,
                subjects_dir=fs_root,
                brain_surfaces='white',
                orientation='coronal')

#%% Co-registration with anatomical landmars 
mne.gui.coregistration(subject=fs_subject, subjects_dir=fs_root,block=True)

#%% Visualize coregistration
info = read_raw_bids(bids_path=raw_path, verbose=False).info
print(info)
mne.viz.plot_alignment(info, trans=trans_file, subject=fs_subject, dig=True,
                           meg=['helmet', 'sensors'], subjects_dir=fs_root,
                           surfaces='head-dense')

#%% Computing the sources according to the BEM model
surface = op.join(fs_root, fs_subject, 'bem', 'inner_skull.surf')
src = mne.setup_volume_source_space(fs_subject, subjects_dir=fs_root,
                                     surface=surface,
                                     verbose=True)
mne.write_source_spaces(src_file, src, overwrite=True)

#%% Visulize BEM
mne.viz.plot_bem(subject=fs_subject, 
                     subjects_dir=fs_root,
                     brain_surfaces='white', 
                     src=src, 
                     orientation='coronal')
#%% Construting the forward model
fwd = mne.make_forward_solution(info, 
                                trans=trans_file,
                                src=src, 
                                bem=bem,
                                meg=True, 
                                eeg=False, 
                                mindist=5.,  #TODO: minimum distance of sources from inner skull surface (in mm); can be 2.5
                                n_jobs=-1, 
                                verbose=True)
mne.write_forward_solution(fwd_file, fwd, overwrite=True)
