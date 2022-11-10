# PhaseAug

**PhaseAug: A Differentiable Augmentation for Speech Synthesis\\to Simulate One-to-Many Mapping**<br>
Junhyeok Lee, Seungu Han, Hyunjae Cho, Wonbin Jung @ [MINDsLab Inc.](https://github.com/mindslab-ai), SNU, KAIST

[![arXiv](https://img.shields.io/badge/arXiv-2211.04610-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2211.04610) [![GitHub Repo stars](https://img.shields.io/github/stars/mindslab-ai/phaseaug?color=yellow&label=PhaseAug&logo=github&style=flat-square)](https://github.com/mindslab-ai/phaseaug) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://mindslab-ai.github.io/phaseaug/)


**Abstract** : Previous generative adversarial network (GAN)-based neural vocoders are trained to reconstruct the exact ground truth waveform from the paired mel-spectrogram and do not consider the one-to-many relationship of speech synthesis. This conventional training causes overfitting for both the discriminators and the generator, leading to the periodicity artifacts in the generated audio signal. In this work, we present PhaseAug, the first differentiable augmentation for speech synthesis that rotates the phase of each frequency bin to simulate one-to-many mapping. With our proposed method, we outperform baselines without any architecture modification. Code and audio samples will be available at https://github.com/mindslab-ai/phaseaug.

ICASSP 2023 submitted

![phasor](asset/phaseaug_phasor.png) 

## Requirements
- [Pytorch](https://pytorch.org/) for [alias-free-torch](https://github.com/junjun3518/alias-free-torch)
- The requirements are highlighted in [requirements.txt](./requirements.txt).
- We also provide docker setup [Dockerfile](./Dockerfile).
- Cloned [official HiFi-GAN repo](https://github.com/jik876/hifi-gan).
- Downloaded [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).
- (optional) [MelGAN](https://github.com/descriptinc/melgan-neurips) generator
 
## Training
TBD


## References
This implementation uses code from following repositories:
- [Official HiFi-GAN implementation](https://github.com/jik876/hifi-gan)
- [Official MelGAN implementation](https://github.com/descriptinc/melgan-neurips)
- [Official CARGAN implementation](https://github.com/descriptinc/cargan)
- [alias-free-torch](https://github.com/junjun3518/alias-free-torch)

This README and the webpage for the audio samples are inspired by:
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Official HiFi-GAN implementation](https://github.com/jik876/hifi-gan)
