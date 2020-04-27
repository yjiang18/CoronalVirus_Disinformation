# CoronalVirus_Disinformation

This is the code of several models, includes RNN with attention, CNN with BatchNorm, and HAN (Hierarchical Attention Network) for detecting Covid-19 disinformation on the EUvsVirus, this is a multiclass (11 categories) classification, and used BERT to generate word level embedding.

## Preparation / Requirements

* Python 3.6 (Anaconda will work best)
* Tensorflow version 1.14.0
* Keras version 2.2.4
* Bert-as-service version 1.10.0
* beautifulsoup4 version 4.9.0
* requests version 2.23.0
* sklearn version 0.22.2
* NLTK version 3.5

## All results are calculated by 10-fold CV.

| Model | Accuracy (std) | Precision (std) | Recall (std) | F1 (std) | Note |
| --- | --- | --- | --- | --- | --- |
| CNN_BN | 0.3983 (0.0524) | 0.4568 (0.0487) |  0.3315 (0.0481) | 0.3831 (0.0476) | trained on round1.json files (250 samples) |
| RNN_att | 0.4270 (0.0485)|  0.4490 (0.05828) | 0.3808 (0.0504) | 0.4111 (0.0527) | trained on second round json files (800 samples) |
| HAN | 0.5483 (0.0436) |  0.5682 (0.0413) | 0.5324 (0.0406) | 0.5494 (0.0404)|trained on final set (1100 samples)|

Visualization:
- [x] See Demo.py

