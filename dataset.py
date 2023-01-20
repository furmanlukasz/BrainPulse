from mne import Epochs, pick_types, events_from_annotations, create_info, EpochsArray
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf, RawArray
from mne.datasets import eegbci
import scipy.io
import numpy as np
import os


def eegbci_data(tmin, tmax, subject, filter_range = None):

    event_id = dict(ev=0)
    runs = [1,2]  # open eyes vs closed eyes

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))
    raw.set_eeg_reference(projection=True)
    raw.apply_proj()
# Apply band-pass filter
    if filter_range != None:
        raw.filter(filter_range[0], filter_range[1], fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=dict(T0=0))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
    # Read epochs
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)

    return epochs, raw




def eegMCI_data(stim_type, subject, run_type, path_to_data_folder):
    """
    eegMCI_data does read 3d data from .mat file structured as ( n epochs, n channels, n samples ).

    :param stim_type:  'AV','A','V'
    :param subject: select a subject from dataset
    :param run_type: chose run type of selected subject, possible types ( button.dat, test.dat_1, test.dat_2, train.dat )
    :param path_to_data_folder: root path to data folder

    :return: epochs, raw mne objects
    """

    stim_type_long = ['audiovisual','audio','visual']

    if stim_type == 'AV':
        stim_typeL = stim_type_long[0]
    elif stim_type == 'A':
        stim_typeL = stim_type_long[1]
    else:
        stim_typeL = stim_type_long[2]

    targets = ['allNTARGETS','allTARGETS']
    path = os.path.join(path_to_data_folder,stim_typeL,'s'+str(subject)+'_'+stim_type+'_'+run_type+'.mat')

    mat = scipy.io.loadmat(path)

    raw_no_targets = mat[targets[0]].transpose((0, 2, 1))
    raw_targets = mat[targets[1]].transpose((0, 2, 1))

    electrodes_l = np.concatenate(mat['electrodes'])
    electrodes_l = [str(x).replace('[','').replace(']','').replace("'",'') for x in electrodes_l]
    ch_types = ['eeg'] * 16
    info = create_info(electrodes_l, ch_types=ch_types, sfreq=512)
    info.set_montage('standard_1020')

    epochs_no_target = EpochsArray(raw_no_targets, info, tmin=-0.2)
    epochs_target = EpochsArray(raw_targets, info, tmin=-0.2)

    erp_no_target = epochs_no_target.average()
    erp_target = epochs_target.average()

    epochs = np.array([erp_no_target.get_data(),erp_target.get_data()],dtype=np.object)
    raw = [raw_no_targets,raw_targets]
    print(electrodes_l)
    return EpochsArray(epochs,info,tmin=-0.2), np.array(raw)




def eegMCI_data_epochs(stim_type1,stim_type2, subject, run_type, path_to_data_folder):
    """
    eegMCI_data does read 3d data from .mat file structured as ( n epochs, n channels, n samples ).

    :param stim_type1:  'AV','A','V'
    :param stim_type2:  'AV','A','V'
    :param subject: select a subject from dataset
    :param run_type: chose run type of selected subject, possible types ( button.dat, test.dat_1, test.dat_2, train.dat )
    :param path_to_data_folder: root path to data folder

    :return: epochs, raw mne objects
    """

    stim_type_long = ['audiovisual','audio','visual']

    if stim_type1 == 'AV':
        stim_typeL1 = stim_type_long[0]
    elif stim_type1 == 'A':
        stim_typeL1 = stim_type_long[1]
    else:
        stim_typeL1 = stim_type_long[2]

    if stim_type2 == 'AV':
        stim_typeL2 = stim_type_long[0]
    elif stim_type2 == 'A':
        stim_typeL2 = stim_type_long[1]
    else:
        stim_typeL2 = stim_type_long[2]

    targets = ['allNTARGETS','allTARGETS']
    path1 = os.path.join(path_to_data_folder,stim_typeL1,'s'+str(subject)+'_'+stim_type1+'_'+run_type+'.mat')
    path2 = os.path.join(path_to_data_folder,stim_typeL2,'s'+str(subject)+'_'+stim_type2+'_'+run_type+'.mat')

    mat1 = scipy.io.loadmat(path1)
    mat2 = scipy.io.loadmat(path2)

    raw_no_targets1 = mat1[targets[0]].transpose((0, 2, 1))
    raw_targets1 = mat1[targets[1]].transpose((0, 2, 1))

    raw_no_targets2 = mat2[targets[0]].transpose((0, 2, 1))
    raw_targets2 = mat2[targets[1]].transpose((0, 2, 1))

    electrodes_l = np.concatenate(mat1['electrodes'])
    electrodes_l = [str(x).replace('[','').replace(']','').replace("'",'') for x in electrodes_l]
    ch_types = ['eeg'] * 16
    info = create_info(electrodes_l, ch_types=ch_types, sfreq=512)
    info.set_montage('standard_1020')

    # epochs_no_target = EpochsArray(raw_no_targets1, info, tmin=-0.2)
    epochs_target1 = EpochsArray(raw_targets1, info, tmin=-0.2, baseline=(-0.2,0))
    epochs_target2 = EpochsArray(raw_targets2, info, tmin=-0.2, baseline=(-0.2,0))

    # erp_no_target = epochs_no_target.average()
    erp_target1 = epochs_target1.average()
    erp_target2 = epochs_target2.average()

    evoked = np.array([erp_target1.get_data(),erp_target2.get_data()],dtype=np.object)
    epochs = [epochs_target1,epochs_target2]
    print(electrodes_l)
    return EpochsArray(evoked,info,tmin=-0.2), epochs


def eegSCH_data(stim_type, subject, path, reref=False):
    
    """
    eegSCH_data does read 3d data from .npy file.
    
    :param stim_type:  'sch','norm'
    :param subject: select a subject from dataset
    :param path_to_data_folder: root path to data folder containing 'norm' and 'sch' subfolders

    :return: eeg data of the raw object (n_electrodes,n_samples), raw mne object
    """

    sf=128
    ch_types='eeg'
    ch_names=['F7','F3', 'F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
     
    subdir_norm=path+'\\'+'norm'
    subdir_sch=path+'\\'+'sch'
            
    if stim_type=='norm':
        idx=os.listdir(subdir_norm).index(subject+'.npy')
        file=os.listdir(subdir_norm)[idx]
        
        filepath=subdir_norm+'\\'+file
        data=np.load(filepath)
        data=data*1e-06 #convert to volts
        info = create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
        raw = RawArray(data, info)
        info.set_montage('standard_1020')
        
        

    if stim_type=='sch':
        idx=os.listdir(subdir_sch).index(subject+'.npy')
        file=os.listdir(subdir_sch)[idx]         
        
        filepath=subdir_sch+'\\'+file
        data=np.load(filepath)
        data=data*1e-06 ##convert to volts
        info = create_info(ch_names=ch_names, sfreq=sf, ch_types=ch_types)
        raw = RawArray(data, info)
        info.set_montage('standard_1020')
        
        

    if reref==True:
        raw=raw.copy().set_eeg_reference(ref_channels='average')

    return raw.get_data(), raw
