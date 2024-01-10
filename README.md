# Full-frequency Dynamic Convolution [![arxiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2203.15296)
The official implementation of *Full-frequency dynamic convolution: a physical frequency-dependent convolution for sound event detection.* (Submitted to ICME 2024)<br>Authors: Haobo Yue, Zhicheng Zhang, Da Mu, Yonghao Dang, Jianqin Yin, Jin Tang

[Issues :blush:](https://github.com/Harper812/FFDConv/issues) **|** [Lab :clap:](https://github.com/BUPT-COST-lab) **|** [Contact :mailbox:](hby@bupt.edu.cn)  

## Updating
Code will be released soon!

## Introduction
### Frequency-dependent modeling
*Full-frequency dynamic convolution* (FFDConv) is proposed as the first full-dynamic method in SED. It generates frequency kernels for every frequency band, which is designed directly in the structure for frequency-dependent modeling. FFDConv physically furnished 2D convolution with the capability of frequency-dependent modeling.
<div align="center">
<img src="./figure/introduction.jpg" width="500" height="400">
</div>

### Fine-grained temporal coherence
Most SED models are trained in a frame-based supervised way, which always leads to the feature and output being discrete over time. FFDConv can alleviate this by frequency-dependent modeling. Besides, the convolution kernel of FFDConv for a frequency band is shared in all frames, which can produce temporally coherent representations. This is consistent with both the continuity of the sound waveform and the vocal continuity of sound events.
<div align="center">
<img src="./figure/feature1.png" width="500" height="500">
</div>


## Performance
FFDConv is evaluated on [DESED](https://github.com/turpaultn/DESED)

Model                   | PSDS1          | PSDS2          | EB F1            | IB F1
:----------------------:|:--------------:|:--------------:|:----------------:|:-------------:
CRNN                    | 0.370          | 0.579          | 0.469            | 0.714
DDFConv                 | 0.387          | 0.624          | 0.467            | 0.720
FTDConv                 | 0.395          | 0.651          | 0.495            | 0.740
FFDConv                 | **0.436**      | **0.685**      | **0.526**        | **0.751**


## Reference
Our code is implemented based on [FDY-SED](https://github.com/frednam93/FDY-SED) and [ddfnet](https://github.com/theFoxofSky/ddfnet).<br>Specifically, experimental environment is based on [FDY-SED](https://github.com/frednam93/FDY-SED), and model structure is based on [ddfnet](https://github.com/theFoxofSky/ddfnet).<br>Thanks for their great work!


## Citation
If this repository helped your works, please cite papers below! :kissing_heart:
```bib
@article{nam2022freqdynamicconv,
      title={Frequency Dynamic Convolution: Frequency-Adaptive Pattern Recognition for Sound Event Detection}, 
      author={Hyeonuk Nam and Seong-Hu Kim and Byeong-Yun Ko and Yong-Hwa Park},
      journal={arXiv preprint arXiv:2203.15296},
      year={2022},
}
```
