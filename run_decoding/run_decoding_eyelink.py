"""Run decoding analyses from eyelink signal for the working memory task and
save decoding performance"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
import numpy as np
import mne
from h5io import read_hdf5
from mne.decoding import (GeneralizingEstimator, cross_val_multiscore,
                          LinearModel)
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from jr.gat import (AngularRegression, scorer_spearman,
                    scorer_angle)
from base import (complete_behavior, get_events_interactions)
from config import path_data
import sys
subject = sys.argv[1]  # read a swarm file for parralel computing on biowulf

# Define analyses
analyses = ['cue_side', 'cue_type',
            'target_angle_cue_angle', 'target_sfreq_cue_sfreq']
analyses = dict(Target=[],
                Cue=analyses,
                Probe=[])

output_folder = '/decoding_from_eyelink/'


# Load behavior file and epoch data
def load(subject, event_type):
    # Behavior
    fname = op.join(path_data, subject, 'behavior_%s.hdf5' % event_type)
    events = read_hdf5(fname)
    # add explicit conditions
    events = complete_behavior(events)

    # MEG
    fname = op.join(path_data, subject, 'epochs_%s.fif' % event_type)
    epochs = mne.read_epochs(fname)
    return epochs, events


# Create result folder
results_folder = op.join(path_data + 'results/' + subject + output_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# Loop across each analysis
for epoch_type, epoch_analyses in analyses.iteritems():
    epochs, events = load(subject, epoch_type)
    events = get_events_interactions(events)
    #  Keep only eye tracker signal (x and y eye position)
    epochs.pick_channels(['UADC009-2104', 'UADC010-2104'])
    for analysis in epoch_analyses:
        fname = results_folder +\
                '%s_scores_%s_%s.npy' % (subject, epoch_type, analysis)
        # define to-be-predicted values
        y = np.array(events[analysis])
        # Define estimators depending on the analysis
        if 'angle' in analysis[:14]:
            clf = AngularRegression(make_pipeline(StandardScaler(),
                                                  LinearModel(Ridge())),
                                    independent=False)
            scorer = scorer_angle
            kwargs = dict()
            gat = GeneralizingEstimator(clf, scoring=make_scorer(scorer),
                                        n_jobs=24, **kwargs)
            y = np.array(y, dtype=float)
        elif 'sfreq' in analysis[:14]:
            clf = make_pipeline(StandardScaler(), LinearModel(Ridge()))
            scorer = scorer_spearman
            kwargs = dict()
            gat = GeneralizingEstimator(clf, scoring=make_scorer(scorer),
                                        n_jobs=24, **kwargs)
            y = np.array(y, dtype=float)
        elif ('cue_side' in analysis or 'cue_type' in analysis):
            clf = make_pipeline(StandardScaler(),
                                LinearModel(LogisticRegression()))
            kwargs = dict()
            gat = GeneralizingEstimator(clf, scoring='roc_auc',
                                        n_jobs=24, **kwargs)
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        # only consider non NaN values
        if ('cue_side' in analysis or 'cue_type' in analysis):
            sel = np.where(y != 0)[0]
        else:
            sel = np.where(~np.isnan(y))[0]
        # Run decoding
        gat.fit(epochs._data[sel], y=y[sel])
        scores = cross_val_multiscore(gat, epochs._data[sel],
                                      y=y[sel], cv=StratifiedKFold(12))
        scores = scores.mean(axis=0)
        # save cross-validated scores
        np.save(fname, np.array(scores))
