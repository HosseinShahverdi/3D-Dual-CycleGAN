# MedNifti_DuCycleGan
Enhancing 3D Multi-Contrast MRI Synthesis with the 3D Dual-CycleGAN Model
## Description

GANs offer the ability to represent sharp and complex probability densities through a nonparametric approach . They have been widely adopted in medical image analysis, particularly for tasks like data augmentation and multi-modality image translations, due to their capability to handle domain shift . To address the issue of domain-specific deformations being encoded as domain-specific features and reproduced in the synthesized output, researchers have integrated CycleGAN into the training process. Previous studies have demonstrated that CycleGAN can be trained using unpaired brain data . However, these studies were more limited to training the network on a single slice and were two-dimensional in nature. Moreover, image synthesis was primarily performed within a single modality, such as synthesizing T1W from T2W or synthesizing T2W from FLAIR and vice versa. This study aims to synthesize 3D Multi-Contrast MRI using 3D Dual-CycleGAN.
## Getting Started

### Dependencies
* prerequisites, libraries, etc., needed before installing program.
* tensorflow==1.14.0
* imageio==2.22.4
* matplotlib==3.5.3
* nibabel==4.0.2
* numpy==1.21.6
* opencv_python==4.1.0.25
* Pillow==9.3.0
* scipy==1.7.3
* SimpleITK==2.2.1

### Installing

* first of all you should install requirments with code below:
```
pip install -r requiremnets.txt
```


### Executing program
for running the code you should use main.py file and run it.
```
python main.py
```

## Authors

Contributors names and contact info

Ali Mahboubisarighieh
mahboubi.ali1991@gmail.com
,
Seyed Masoud Rezaeijo
masoudrezayi1398@gmail.com

## Version History

* 0.1
    * Initial Release


