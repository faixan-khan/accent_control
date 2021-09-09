![alt text](assets/banner.jpg)

# Deepvoice3_pytorch with Lip Embeddings

This work was build upon [Deepvoice3](https://github.com/r9y9/deepvoice3_pytorch). 

This work allows to control the accent of a speaker by making use of a pseudo-visual stream. The lip embeddings from the pseudo-visual stream are added to the deepvoice architecture allowing for accent control.

This work was inspired from the following works:

1. [arXiv:1710.07654](https://arxiv.org/abs/1710.07654): Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning.
2. [arXiv:1710.08969](https://arxiv.org/abs/1710.08969): Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.



## Highlights

- Convolutional sequence-to-sequence model with attention for text-to-speech synthesis
- Use of lip embeddings to gain accent control.
- Multi-speaker and single speaker versions of DeepVoice3
- Audio samples and pre-trained models
- Preprocessor for [LJSpeech (en)](https://keithito.com/LJ-Speech-Dataset/), [JSUT (jp)](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) and [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) datasets, as well as [carpedm20/multi-speaker-tacotron-tensorflow](https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow) compatible custom dataset (in JSON format)
- Language-dependent frontend text processor for English and Japanese


## Notes on hyper parameters

- Default hyper parameters, used during preprocessing/training/synthesis stages, are turned for English TTS using LJSpeech dataset. You will have to change some of parameters if you want to try other datasets. See `hparams.py` for details.
- `builder` specifies which model you want to use. `deepvoice3`, `deepvoice3_multispeaker` [1] and `nyanko` [2] are surpprted.
- Hyper parameters described in DeepVoice3 paper for single speaker didn't work for LJSpeech dataset, so I changed a few things. Add dilated convolution, more channels, more layers and add guided attention loss, etc. See code for details. The changes are also applied for multi-speaker model.
- Multiple attention layers are hard to learn. Empirically, one or two (first and last) attention layers seems enough.
- With guided attention (see https://arxiv.org/abs/1710.08969), alignments get monotonic more quickly and reliably if we use multiple attention layers. With guided attention, I can confirm five attention layers get monotonic, though I cannot get speech quality improvements.
- Binary divergence (described in https://arxiv.org/abs/1710.08969) seems stabilizes training particularly for deep (> 10 layers) networks.
- Adam with step lr decay works. However, for deeper networks, I find Adam + noam's lr scheduler is more stable.

## Requirements

- Python >= 3.5
- CUDA >= 8.0
- PyTorch >= v1.0.0
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.11
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)

## Acknowledgements

Part of code was adapted from the following projects:

- https://github.com/r9y9/deepvoice3_pytorch
