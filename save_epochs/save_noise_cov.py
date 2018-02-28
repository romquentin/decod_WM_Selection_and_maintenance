""" Create and save noise covariance """

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
from mne import Epochs
import mne
from mne.io import read_raw_ctf
from mne.epochs import concatenate_epochs
from config import path_data
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf

# Add the second session when existing (all subject except one)
subject_path = op.join(path_data, subject)
subject_2 = subject + '_2'

fname_raw = op.join(path_data, subject)
# Read raw MEG data and extract event triggers
runs = list()
files = os.listdir(fname_raw)
runs.extend(([op.join(fname_raw + '/') + f for f in files if '.ds' in f]))
if os.path.exists(op.join(path_data, subject_2)):
    fname_raw = op.join(path_data, subject_2)
    runs_2 = list()
    files = os.listdir(fname_raw)
    runs_2.extend(([op.join(fname_raw + '/') +
                  f for f in files if '.ds' in f]))
    runs.extend(runs_2)
# Read raw data, filter and epoch
epochs_list = list()
for run_number, this_run in enumerate(runs):
    fname_raw = op.join(path_data, subject, this_run)
    print(fname_raw)
    raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
    raw.filter(l_freq=.05, h_freq=25.0, fir_design='firwin')
    channel_trigger = np.where(np.array(raw.ch_names) == 'USPT001')[0][0]
    # trigger baseline is 255
    # Replace 255 values with 0 for easier reading of events
    trigger_baseline = np.where(raw._data[channel_trigger, :] == 255)[0]
    raw._data[channel_trigger, trigger_baseline] = 0.
    # find triggers
    events_meg = mne.find_events(raw)
    # Add 48ms to the trigger events (according to the photodiod)
    events_meg = np.array(events_meg, float)
    events_meg[:, 0] += round(.048 * raw.info['sfreq'])
    events_meg = np.array(events_meg, int)
    # Select only events corresponding to stimulus onset
    events_meg = events_meg[np.where(events_meg[:, 2] <= 125)]
    epochs = Epochs(raw, events_meg,
                    tmin=-0.35, tmax=0, preload=True,
                    baseline=(-0.35, 0.0), decim=10)
    # Copy first run dev_head_t to following runs
    if run_number == 0:
        dev_head_t = epochs.info['dev_head_t']
    else:
        epochs.info['dev_head_t'] = dev_head_t
    # no button 2 recording during acquisition of the 1st session for sub01
    if (subject == 'sub01_YFAALKWR_JA') & (epochs.info['nchan'] == 308):
        epochs.drop_channels(['UADC007-2104'])
    epochs_list.append(epochs)
epochs = concatenate_epochs(epochs_list)
# compute noise covariance and save it
cov = mne.compute_covariance(epochs)
cov_fname = op.join(subject_path, '%s-cov.fif' % subject)
cov.save(cov_fname)
