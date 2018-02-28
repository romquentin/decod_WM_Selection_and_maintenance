""" Create and save epoch for time frequency analyses """

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
from h5io import read_hdf5, write_hdf5
import pandas as pd
import os.path as op
import numpy as np
from mne import Epochs, pick_types
import mne
from mne.io import read_raw_ctf
from mne.epochs import concatenate_epochs
from base import get_events_from_mat, fix_triggers
from config import path_data
import shutil
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf
target_baseline = False  # If True apply baseline from before stimulus onset


def make_events_run(events_bhv, run_number):
    """Takes pandas dataframe and make events array taken by MNE"""
    sel = events_bhv['meg_file'] == run_number
    time_sample = events_bhv['meg_event_tsample'][sel]
    trigger_value = events_bhv['meg_event_value'][sel]
    events = np.vstack((time_sample.astype(int),
                        np.zeros_like(time_sample, int),
                        trigger_value.astype(int))).T
    return events


# Add the second session when existing (all subject except one)
subjects = list([subject])
subject_2 = subject + '_2'
if os.path.exists(op.join(path_data, subject_2)):
    subjects.append(subject_2)
print subject
# loop accross subjects
for subject in subjects:
    fname_raw = op.join(path_data, subject)
    # Read behavioral file
    fname_bhv = list()
    files = os.listdir(op.join(path_data, subject, 'behavdata'))
    fname_bhv.extend(([op.join(fname_raw + '/behavdata/') +
                       f for f in files if 'WorkMem' in f]))
    for fname_behavior in fname_bhv:
        events_behavior = get_events_from_mat(fname_behavior)
    # Read raw MEG data and extract event triggers
    runs = list()
    files = os.listdir(fname_raw)
    runs.extend(([op.join(fname_raw + '/') + f for f in files if '.ds' in f]))
    events_meg = list()
    for run_number, this_run in enumerate(runs):
        fname_raw = op.join(path_data, subject, this_run)
        print(fname_raw)
        raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
        channel_trigger = np.where(np.array(raw.ch_names) == 'USPT001')[0][0]
        # trigger baseline is 255
        # Replace 255 values with 0 for easier reading of events
        trigger_baseline = np.where(raw._data[channel_trigger, :] == 255)[0]
        raw._data[channel_trigger, trigger_baseline] = 0.
        # find triggers
        events_meg_ = mne.find_events(raw)
        # Add 48ms to the trigger events (according to delay with photodiod)
        events_meg_ = np.array(events_meg_, float)
        events_meg_[:, 0] += round(.048 * raw.info['sfreq'])
        events_meg_ = np.array(events_meg_, int)
        # to keep the run from which the event was found
        events_meg_[:, 1] = run_number
        events_meg.append(events_meg_)
    # concatenate all meg events
    events_meg = np.vstack(events_meg)
    # add trigger index to meg_events
    events_target = range(1, 126)  # events correponding to the stimulus onset
    events_cue = range(126, 130)  # events correponding to the cue onset
    events_probe = range(130, 141)  # events correponding to the probe onset
    triggidx_array = []
    for trigg in events_meg[:, 2]:
        if trigg in events_target:
            triggidx = 1
        elif trigg in events_cue:
            triggidx = 2
        elif trigg in events_probe:
            triggidx = 3
        triggidx_array.append(triggidx)
    events_meg = np.insert(events_meg, 3, triggidx_array, axis=1)
    # Compare MEG and bhv triggers and save events_behavior for each event
    event_types = ['Target', 'Cue', 'Probe']
    events_behavior_type = []
    events_baseline = fix_triggers(events_meg, events_behavior,
                                   event_type='triggTarget')
    for event_type in event_types:
        print(event_type)
        events_behavior_type = fix_triggers(events_meg, events_behavior,
                                            event_type='trigg' + event_type)
        epochs_list = list()
        # Read raw MEG, filter and epochs
        for run_number, this_run in enumerate(runs):
            fname_raw = op.join(path_data, subject, this_run)
            print(fname_raw)
            raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
            raw.filter(l_freq=.05, h_freq=25.0, fir_design='firwin')
            events_meg_run = make_events_run(events_behavior_type, run_number)
            event_id = {'ttl_%i' % ii: ii
                        for ii in np.unique(events_meg_run[:, 2])}
            if event_type == 'Target':  # -0.2s to 0.9s around stim onset
                tmin = -.200
                tmax = 0.900
            elif event_type == 'Cue':  # -0.2s to 1.5s around cue onset
                tmin = -.200
                tmax = 1.500
            elif event_type == 'Probe':  # -0.2s to 0.9s around probe onset
                tmin = -.200
                tmax = 0.900
            epochs = Epochs(raw, events_meg_run, event_id=event_id,
                            tmin=tmin, tmax=tmax, preload=True,
                            baseline=None, decim=10)
            # Copy first run dev_head_t to following runs
            if run_number == 0:
                dev_head_t = epochs.info['dev_head_t']
            else:
                epochs.info['dev_head_t'] = dev_head_t
            # Get baseline (either before target or before each event)
            if target_baseline:
                events = make_events_run(events_baseline, run_number)
                event_id_bsl = {'ttl_%i' % ii: ii
                                for ii in np.unique(events[:, 2])}

                epochs_baseline = Epochs(raw, events,
                                         event_id=event_id_bsl,
                                         tmin=-.200, tmax=0., preload=True,
                                         baseline=None, decim=10)
                # Apply baseline of Target
                bsl_channels = pick_types(epochs.info, meg=True)
                bsl_data = epochs_baseline.get_data()[:, bsl_channels, :]
                bsl_data = np.mean(bsl_data, axis=2)
                epochs._data[:, bsl_channels, :] -= bsl_data[:, :, np.newaxis]
            else:
                # Apply baseline from beginning of epoch to t0
                epochs.apply_baseline((-0.2, 0.))
            epochs_list.append(epochs)
        epochs = concatenate_epochs(epochs_list)
        # Save epochs and hdf5 behavior
        suffix = '' if target_baseline else '_bsl'
        session = '_2' if subject[-1:] == '2' else '_1'
        fname = op.join(path_data,  subject, 'behavior_%s%s.hdf5'
                        % (event_type, session))
        write_hdf5(fname, events_behavior_type, overwrite=True)
        fname = op.join(path_data, subject,
                        'epochs_%s%s%s.fif' % (event_type, suffix, session))
        epochs.save(fname)

# concatenate the two sessions when 2nd one is existing
subject = sys.argv[1]
suffix = '' if target_baseline else '_bsl'
for event_type in event_types:
    subject_2 = subject + '_2'
    if os.path.exists(op.join(path_data, subject_2)):
        epochs_list = list()
        fname1 = op.join(path_data, subject, 'epochs_%s%s_1.fif'
                         % (event_type, suffix))
        epochs = mne.read_epochs(fname1)
        # Copy dev_head_t of the first session to the second session
        dev_head_t = epochs.info['dev_head_t']
        epochs_list.append(epochs)
        fname2 = op.join(path_data, subject + '_2', 'epochs_%s%s_2.fif'
                         % (event_type, suffix))
        epochs = mne.read_epochs(fname2)
        epochs.info['dev_head_t'] = dev_head_t
        if subject == 'sub01_YFAALKWR_JA':  # miss button 2 recordings on
                                            # 1st session for s01
            epochs.drop_channels(['UADC007-2104'])
        epochs_list.append(epochs)
        epochs = concatenate_epochs(epochs_list)
        fname = op.join(path_data, subject, 'epochs_%s%s.fif'
                        % (event_type, suffix))
        epochs.save(fname)
        # delete epoch-1 and epoch-2 to keep only the concatenate one
        os.remove(fname1)
        os.remove(fname2)
        # concatenate behavior files
        fname1 = op.join(path_data, subject, 'behavior_%s_1.hdf5'
                         % event_type)
        events1 = read_hdf5(fname1)
        fname2 = op.join(path_data, subject_2, 'behavior_%s_2.hdf5'
                         % event_type)
        events2 = read_hdf5(fname2)
        frames = [events1, events2]
        events = pd.concat(frames, axis=0)
        fname = op.join(path_data,  subject, 'behavior_%s.hdf5'
                        % event_type)
        write_hdf5(fname, events, overwrite=True)
    # if only one session has been acquired (one subject)
    else:
        fname1 = op.join(path_data, subject, 'epochs_%s%s_1.fif'
                         % (event_type, suffix))
        fname = op.join(path_data, subject, 'epochs_%s%s.fif'
                        % (event_type, suffix))
        shutil.copy(fname1, fname)
        os.remove(fname1)
        fname1 = op.join(path_data, subject, 'behavior_%s_1.hdf5'
                         % event_type)
        fname = op.join(path_data, subject, 'behavior_%s.hdf5'
                        % event_type)
        shutil.copy(fname1, fname)
