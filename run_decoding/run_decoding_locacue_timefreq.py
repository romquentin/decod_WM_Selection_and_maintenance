"""Run decoding analyses in time-frequency domain for the control task
(locacue) and save decoding performance"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
from h5io import read_hdf5
from mne import Epochs
import mne
from mne.io import read_raw_ctf
from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel
from config import path_data
from pandas import DataFrame as df
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from jr.gat import force_predict, scorer_auc
from base import complete_behavior, get_events_interactions
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf

# Define analyses
analyses = ['cue_side', 'cue_type']

# create results folder
output_folder = '/locacue_timefreq/'
results_folder = op.join(path_data + 'results/' + subject + output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


def get_events_from_mat(fname_behavior):
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


# define frequency range
freqs = np.arange(2, 60, 2)
n_cycles = freqs / 2.

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
    events_behavior = get_events_from_mat(fname_behavior)
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
# trigger baseline is 255
# Replace 255 values with 0 for easier reading of events
trigger_baseline = np.where(raw._data[channel_trigger, :] == 255)[0]
raw._data[channel_trigger, trigger_baseline] = 0.
# find triggers
events_meg = mne.find_events(raw)
# Add 48ms to the trigger events (according to delay with photodiod)
events_meg = np.array(events_meg, float)
events_meg[:, 0] += round(.048 * raw.info['sfreq'])
events_meg = np.array(events_meg, int)
# Define tmin and tmax of the epoch
tmin = -.200
tmax = 1.5

# Epoch data
event_id = {'ttl_%i' % ii: ii
            for ii in np.unique(events_meg[:, 2])}
epochs = Epochs(raw, events_meg, event_id=event_id,
                tmin=tmin, tmax=tmax, preload=True,
                baseline=(None, 0), decim=10)
epochs.pick_types(meg=True, ref_meg=False)
# Compute time frequency decomposition
X = mne.time_frequency.tfr_array_morlet(epochs.get_data(),
                                        sfreq=epochs.info['sfreq'],
                                        freqs=freqs,
                                        output='power',
                                        n_cycles=n_cycles)
n_epochs, n_channels, n_freqs, n_times = X.shape
X = X.reshape(n_epochs, n_channels, -1)  # collapse freqs and time
# Run decoding on TFR output
for analysis in analyses:
    fname = results_folder +\
        '%s_tf_scores_%s_%s.npy' % (subject, 'Cue', analysis)
    y = np.array(events_behavior[analysis])
    clf = make_pipeline(StandardScaler(),
                        force_predict(LogisticRegression(),
                        'predict_proba', axis=1))
    scorer = scorer_auc
    kwargs = dict()
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    sel = np.where(y != 0)[0]
    td = SlidingEstimator(clf, scoring=make_scorer(scorer),
                          n_jobs=24, **kwargs)
    td.fit(X[sel], y[sel])
    scores = cross_val_multiscore(td, X[sel],
                                  y[sel], cv=StratifiedKFold(12))
    scores = scores.mean(axis=0)
    scores = np.reshape(scores, (n_freqs, n_times))
    # Save cross validated scores
    np.save(fname, np.array(scores))
