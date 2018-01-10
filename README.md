# icassp-2017-world-music

Code used in ICASSP 2017 publication ["Towards the characterization of singing styles in world music"](http://ieeexplore.ieee.org/document/7952233/) by Maria Panteli, Rachel Bittner, Juan Pablo Bello, Simon Dixon

## Overview

This project models attributes of the singing voice and explores singing style similarity in a set of world music recordings. 

The code
- extracts features from melodic pitch contours
- classifies the contours as vocal or non-vocal
- learns a dictionary of singing elements from the vocal contours
- and groups together recordings with similar vocal content.

![alt tag](https://raw.githubusercontent.com/rabitt/icassp-2017-world-music/master/data/methodology.png)

## Requirements

This project relies on [motif: Melodic Object TranscrIption Framework](https://github.com/rabitt/motif) and especially the ["bitteli"](https://github.com/rabitt/motif/blob/master/motif/feature_extractors/bitteli.py) feature extractor.
