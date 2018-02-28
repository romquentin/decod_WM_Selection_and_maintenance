"""
Figure 3
Plot source patterns
"""

# Authors: Romain Quentin <rom.quentin@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne import EvokedArray
from config import path_data
from sklearn.decomposition import PCA
from jr.plot.base import alpha_cmap, LinearSegmentedColormap
from surfer import Brain
from webcolors import hex_to_rgb
# Define recordings sfreq
sfreq = 120
# Define colors
colors = ['#1f77b4', '#d62728', '#ff7f0e']



# Define pattern to plot (here left_spatial frequency)
mean_morph_patterns = mne.read_source_estimate(path_data +
    'morph_source_patterns/Target_left_sfreq_patterns-rh.stc')

n_components = 2  # keep the first 2 components
pca = PCA(n_components, random_state=0).fit(mean_morph_patterns.data.T)
n_components = len(pca.components_)
time_course = mne.EvokedArray(
    data=pca.transform(mean_morph_patterns.data.T).T,
    info=mne.create_info(len(pca.components_), sfreq=sfreq),
    tmin=-0.2)

colors = np.array(hex_to_rgb('#ff7f0e'))/255.
colors = np.concatenate((colors, [1]), axis=0)
colors_ = (colors, colors)
# Show brain
brain = Brain('fsaverage', 'both', 'inflated_pre', background='w',
              cortex='low_contrast')
# Show patterns
for side, slic in (('lh', slice(0, 10242)), ('rh', slice(10242, None))):
    for roi, color in zip(pca.components_, colors_):
        roi = np.abs(roi[slic])
        cmap = alpha_cmap(LinearSegmentedColormap.from_list(
            'RdBu', [color, color]), diverge=False)
        brain.add_data(roi, hemi=side, colormap=cmap, smoothing_steps=5,
                       vertices=mean_morph_patterns.vertices[side == 'rh'],
                       alpha=0.5)

brain.show_view(view='caudal')
# brain.show_view(view='lateral')
