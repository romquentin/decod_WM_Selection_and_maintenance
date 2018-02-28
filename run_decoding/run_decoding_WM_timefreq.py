"""Run decoding analyses in time-frequency domain for the working memory
task and save decoding performance"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import mne
from h5io import read_hdf5
from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from jr.gat import (AngularRegression, scorer_spearman, scorer_auc,
                    scorer_angle)
from base import (complete_behavior, get_events_interactions)
from config import path_data
from sklearn import preprocessing
from sklearn.metrics import make_scorer
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf
output_folder = '/time_frequency/'

# Define analyses
analyses = ['right_angle', 'left_angle', 'right_sfreq', 'left_sfreq',
            'target_angle', 'target_sfreq', 'distr_angle',
            'distr_sfreq', 'cue_side', 'cue_type',
            'probe_angle', 'probe_sfreq',
            'target_angle_cue_angle', 'target_angle_cue_sfreq',
            'target_sfreq_cue_angle', 'target_sfreq_cue_sfreq']
analyses = dict(Target=analyses,
                Cue=analyses,
                Probe=analyses)


# Load behavior file and epoch data
def load(subject, event_type):
    # Behavior
    fname = op.join(path_data, subject, 'behavior_%s.hdf5' % event_type)
    events = read_hdf5(fname)
    # add explicit conditions
    events = complete_behavior(events)

    # MEG
    fname = op.join(path_data, subject, 'epochs_tf_%s.fif' % event_type) # noqa
    epochs = mne.read_epochs(fname)
    # epochs.decimate(10)
    return epochs, events


# define frequency range
freqs = np.arange(2, 60, 2)
n_cycles = freqs / 2.

# Create result folder
results_folder = op.join(path_data + 'results/' + subject + output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# Compute time-frequency decomposition
for epoch_type, epoch_analyses in analyses.iteritems():
    epochs, events = load(subject, epoch_type)
    events = get_events_interactions(events)
    epochs.pick_types(meg=True, ref_meg=False)
    X = mne.time_frequency.tfr_array_morlet(epochs.get_data(),
                                            sfreq=epochs.info['sfreq'],
                                            freqs=freqs,
                                            output='power',
                                            n_cycles=n_cycles)
    n_epochs, n_channels, n_freqs, n_times = X.shape
    X = X.reshape(n_epochs, n_channels, -1)  # collapse freqs and time
    # Run decoding for each analysis
    for analysis in epoch_analyses:
        fname = results_folder +\
                '%s_tf_scores_%s_%s.npy' % (subject, epoch_type, analysis)
        # define to-be-predicted values
        y = np.array(events[analysis])
        if 'angle' in analysis[:14]:
            pipe = make_pipeline(StandardScaler(), Ridge())
            clf = AngularRegression(pipe, independent=False)
            scorer = scorer_angle
            kwargs = dict()
            y = np.array(y, dtype=float)
        elif 'sfreq' in analysis[:14]:
            clf = make_pipeline(StandardScaler(), Ridge())
            scorer = scorer_spearman
            kwargs = dict()
            y = np.array(y, dtype=float)
        elif ('cue_side' in analysis or 'cue_type' in analysis):
            clf = make_pipeline(StandardScaler(),
                                LinearModel(LogisticRegression()))
            scorer = scorer_auc
            kwargs = dict()
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        # only consider non NaN values
        if ('cue_side' in analysis or 'cue_type' in analysis):
            sel = np.where(y != 0)[0]
        else:
            # When decoding memory at probe time, use only trial with
            # different probe compare to target
            if (epoch_type == 'Probe') & ('target' in analysis):
                sel = np.where((events['Change'] == 1) & (~np.isnan(y)))[0]
            else:
                sel = np.where(~np.isnan(y))[0]
        td = SlidingEstimator(clf, scoring=make_scorer(scorer),
                              n_jobs=24, **kwargs)
        td.fit(X[sel], y[sel])
        scores = cross_val_multiscore(td, X[sel],
                                      y[sel], cv=StratifiedKFold(12))
        scores = scores.mean(axis=0)
        scores = np.reshape(scores, (n_freqs, n_times))
        # save cross-validated scores
        np.save(fname, np.array(scores))
