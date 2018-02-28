"""Run decoding analyses in time-frequency domain for the control task
(locacue) with estimators trained on WM task"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
from pandas import DataFrame as df
import mne
from mne import Epochs
from mne.epochs import concatenate_epochs
from mne.io import read_raw_ctf
from mne.decoding import (SlidingEstimator,
                          LinearModel)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from base import (complete_behavior, get_events_from_mat, fix_triggers)
from config import path_data
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf

output_folder = '/cross_task_time_freq_cue_control/'
# Define analyses
analyses = ['cue_side', 'cue_type']

# Create result folder
results_folder = op.join(path_data + 'results/' + subject + output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


def make_events_run(events_bhv, run_number):
    """Takes pandas dataframe and make events array taken by MNE"""
    sel = events_bhv['meg_file'] == run_number
    time_sample = events_bhv['meg_event_tsample'][sel]
    trigger_value = events_bhv['meg_event_value'][sel]
    events = np.vstack((time_sample.astype(int),
                        np.zeros_like(time_sample, int),
                        trigger_value.astype(int))).T
    return events


def get_events_control_from_mat(fname_behavior):
    """Read CSV file and output pandas.DataFrame object"""
    # Read events from CSV file
    events = np.genfromtxt(fname_behavior, dtype=float, skip_header=1,
                           delimiter=',', names=True)
    labels = (
        'NbTrial', 'FixNbTrial', 'isFixed', 'same', 'Cue',
        'Side', 'Press', 'isCorrect'
    )
    event_data = list()
    for event in events:
        event_dict = {key: value for (key, value) in zip(event, labels)}
        event_data.append(event_dict)
    events_behavior = df(events)
    return events_behavior


def make_cue_epoch_tf(subject):
    """Create cue epochs during WM task """
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
        # replace 255 values with 0
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
    events_target = range(1, 126)
    events_cue = range(126, 130)
    events_probe = range(130, 141)
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
    event_type = 'Cue'
    events_behavior_type = []
    events_behavior_type = fix_triggers(events_meg, events_behavior,
                                        event_type='trigg' + event_type)
    epochs_list = list()
    # Read raw MEG, filter and epochs
    for run_number, this_run in enumerate(runs):
        fname_raw = op.join(path_data, subject, this_run)
        print(fname_raw)
        raw = read_raw_ctf(fname_raw, preload=True, system_clock='ignore')
        events_meg_run = make_events_run(events_behavior_type, run_number)
        event_id = {'ttl_%i' % ii: ii
                    for ii in np.unique(events_meg_run[:, 2])}
        tmin = -.200
        tmax = 1.500
        epochs = Epochs(raw, events_meg_run, event_id=event_id,
                        tmin=tmin, tmax=tmax, preload=True,
                        baseline=None, decim=10)
        # Copy dev_head_t of the first run to others run
        if run_number == 0:
            dev_head_t = epochs.info['dev_head_t']
        else:
            epochs.info['dev_head_t'] = dev_head_t
        # Apply baseline from beginning of epoch to t0
        epochs.apply_baseline((-0.2, 0.))
        epochs_list.append(epochs)
    epochs = concatenate_epochs(epochs_list)
    epochs.pick_types(meg=True, ref_meg=False)
    events = events_behavior_type
    events = complete_behavior(events)
    return epochs, events


def make_control_epoch_tf(subject):
    """Create cue epochs during localizer (control task)"""
    # Raw data path_data
    fname_raw = op.join(path_data, subject, 'locacue')
    runs = list()
    files = os.listdir(fname_raw)
    runs.extend(([op.join(fname_raw + '/') + f for f in files if '.ds' in f]))
    # Behavioral data path
    fname_bhv = list()
    files = os.listdir(op.join(path_data, subject, 'behavdata'))
    fname_bhv.extend(([op.join(path_data, subject, 'behavdata/') +
                       f for f in files if 'locaCue' in f]))

    # Read behavioral file
    for fname_behavior in fname_bhv:
        events_behavior = get_events_control_from_mat(fname_behavior)

    # Add cue_type and cue_side column in events_behavior
    for trial in range(len(events_behavior)):
        event = events_behavior.iloc[trial]
        cue_index = event['Cue']
        tmp = dict()
        tmp['cue_type'] = 'angle' if cue_index in [2, 4] else 'sfreq'
        tmp['cue_side'] = 'left' if cue_index in [1, 2] else 'right'
        for rule in ['side', 'type']:
            key = 'cue_' + rule
            events_behavior.set_value(trial, key, tmp[key])
    sel = np.where(events_behavior['isFixed'] == 0)[0]
    events_behavior['cue_side'][sel] = np.nan
    events_behavior['cue_type'][sel] = np.nan

    # Read all raw to extract MEG event triggers
    raw = read_raw_ctf(runs[0], preload=True, system_clock='ignore')

    channel_trigger = np.where(np.array(raw.ch_names) == 'USPT001')[0][0]
    # replace 255 values with 0
    trigger_baseline = np.where(raw._data[channel_trigger, :] == 255)[0]
    raw._data[channel_trigger, trigger_baseline] = 0.
    # find correct triggers
    events_meg = mne.find_events(raw)
    events_meg = np.array(events_meg, float)
    events_meg[:, 0] += round(.048 * raw.info['sfreq'])
    events_meg = np.array(events_meg, int)

    # band pass filter
    tmin = -.200
    tmax = 1.5

    # Epoch data
    event_id = {'ttl_%i' % ii: ii
                for ii in np.unique(events_meg[:, 2])}
    epochs_con = Epochs(raw, events_meg, event_id=event_id,
                        tmin=tmin, tmax=tmax, preload=True,
                        baseline=(None, 0), decim=10)
    epochs_con.pick_types(meg=True, ref_meg=False)
    events_con = events_behavior
    return epochs_con, events_con

# Create epochs in WM task and control task around cue onset
epochs, events = make_cue_epoch_tf(subject)
epochs_con, events_con = make_control_epoch_tf(subject)

# define freqs range
freqs = np.arange(2, 60, 2)
n_cycles = freqs / 2.
# Compute time-frequency decomposition
# for WM task
X = mne.time_frequency.tfr_array_morlet(epochs.get_data(),
                                        sfreq=epochs.info['sfreq'],
                                        freqs=freqs,
                                        output='power',
                                        n_cycles=n_cycles)
n_epochs, n_channels, n_freqs, n_times = X.shape
X = X.reshape(n_epochs, n_channels, -1)
# for control task (localizer)
X_con = mne.time_frequency.tfr_array_morlet(epochs_con.get_data(),
                                            sfreq=epochs.info['sfreq'],
                                            freqs=freqs,
                                            output='power',
                                            n_cycles=n_cycles)
n_epochs, n_channels, n_freqs, n_times = X_con.shape
X_con = X_con.reshape(n_epochs, n_channels, -1)

# Loop across each analysis
for analysis in analyses:
    # define to-be-predicted values
    y = np.array(events[analysis])  # cue in WM task
    y_con = np.array(events_con[analysis])  # cue in control task (localizer)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    sel = np.where(y != 0)[0]
    le = LabelEncoder()
    le.fit(y_con)
    y_con = le.transform(y_con)
    sel_con = np.where(y_con != 0)[0]
    # Define estimators depending on the analysis
    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression()))
    kwargs = dict()
    est = SlidingEstimator(clf, scoring='roc_auc',
                           n_jobs=24, **kwargs)
    # Run decoding
    cv = StratifiedKFold(12)
    scores = list()
    scores_con = list()
    for train, test in cv.split(X[sel], y[sel]):
        est.fit(X[sel][train], y[sel][train])  # train during WM task
        score = est.score(X[sel][test], y[sel][test])  # test during WM task
        score_con = est.score(X_con[sel_con], y_con[sel_con])  # test during control task
        scores.append(score)
        scores_con.append(score_con)
    scores = np.mean(scores, axis=0)
    scores = np.reshape(scores, (n_freqs, n_times))
    scores_con = np.mean(scores_con, axis=0)
    scores_con = np.reshape(scores_con, (n_freqs, n_times))
    # save cross-validated scores
    fname = results_folder +\
        '%s_scores_tf_%s.npy' % (subject, analysis)
    np.save(fname, np.array(scores))
    fname = results_folder +\
        '%s_scores_tf_%s_con.npy' % (subject, analysis)
    np.save(fname, np.array(scores_con))
