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