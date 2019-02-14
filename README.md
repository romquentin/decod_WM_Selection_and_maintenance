# Code corresponding to [this manuscript](https://doi.org/10.1101/283234)

 Differential brain mechanisms of selection and maintenance of information during working memory
================
Romain Quentin, Jean-Rémi King, Etienne Sallard, Nathan Fishman, Ryan Thompson, Ethan Buch & Leonardo G. Cohen Biorxiv 2018 (https://doi.org/10.1101/283234)


Abstract
========

Working memory is our ability to select and temporarily hold information as needed for complex cognitive operations. The temporal dynamics of sustained and transient neural activity supporting the selection and holding of memory content is not known. To address this problem, we recorded magnetoencephalography (MEG) in healthy participants performing a retro-cue working memory task in which the selection rule and the memory content varied independently. Multivariate decoding and source analyses showed that selecting the memory content relies on prefrontal and parieto-occipital persistent oscillatory neural activity. By contrast, the memory content was reactivated in a distributed occipito-temporal posterior network, preceding the working memory decision and in a different format that during the visual encoding. These results identify a neural signature of content selection and characterize differentiated spatiotemporal constraints for subprocesses of working memory.


Data
====

Data are publicly accessible at https://doi.org/10.18112/openneuro.ds001750.v1.3.0 (OpenNeuro Neuroimaging Platform)

Scripts
=======

Overall, the current scripts remain designed for research purposes, and could therefore be improved and clarified. If you judge that some codes would benefit from specific clarifications do not hesitate to contact us.

Scripts are separated in 3 folders:
- save_epochs: MEG preprocessing,
- run_decoding: MVPA decoding analyses in sensor space, time-frequency and sources,
- plot: group-level statistics and plotting

#### Config files
- 'base.py' # where all generic functions are defined
- 'config.py'  # where the paths and filenames are setup

#### save_epochs
- 'save_epochs.py'  # MEG preprocessing and epoching
- 'save_epochs_tf.py'  # MEG preprocessing and epoching for time-frequency
- 'save_noise_cov.py'  # compute noise covariance

#### Decoding
- 'run_decoding_WM.py'  # decoding in sensor space during WM task
- 'run_decoding_WM_timefreq.py'  # decoding in time-frequency domain during WM task
- 'run_decoding_WM_source_pattern.py'  # decoding in source space during WM task and save weights and  patterns
- 'run_decoding_WM_tf_source_pattern.py'  # decoding in time-frequency source space during WM task and save weights and  patterns
- 'run_decoding_locacue.py'  # decoding in sensor space during control task (localizer)
- 'run_decoding_locacue_timefreq.py'  # decoding in time-frequency domain during control task (localizer)
- 'run_decoding_locacue_across_task.py'  # decoding in sensor space during control task (localizer) with estimators trained during WM task
- 'run_decoding_timefreq_locacue_across_task.py'  # decoding in time-frequency domain during control task (localizer) with estimators trained during WM task
- 'run_decoding_WM_across_epochs_and_conditions.py'  # decoding in sensors space during WM task and generalizing estimators trained during visual perception to memory delay and vice versa.
- 'run_decoding_eyelink.py'  # decoding from eye tracker signal during WM task

#### Plots
- Plots and statistics corresponding to each figure on the manuscript

Dependencies
============
- Python 2.7.13
- MNE: 0.16.dev0
- scikit-learn: 0.18.1
- pandas: 0.20.3
- matplotlib: 2.0.2
- scipy: 0.19


Acknowledgements
================

This project is powered by

![logos](docs/logo.pdf)

and RQ received fundings from

![logos](docs/logo_funding.pdf)
