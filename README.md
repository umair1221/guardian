# GUARDIAN: Guarding Against Uncertainty and Adversarial Risks in Robot-Assisted Surgeries

> [**GUARDIAN: Guarding Against Uncertainty and Adversarial Risks in Robot-Assisted Surgeries**](https://openreview.net/forum?id=kW9StEs1a5)<br>

> [Ufaq Khan](https://scholar.google.com/citations?user=E0p-7JEAAAAJ&hl=en&oi=ao),
 [Umair Nawaz](https://scholar.google.com/citations?user=w7N4wSYAAAAJ&hl=en), 
[Tooba Tehreem Sheikh](https://github.com/toobatehreem), [Asif Hanif](https://scholar.google.com/citations?user=6SO2wqUAAAAJ&hl=en) and [Mohammad Yaqub](https://scholar.google.co.uk/citations?user=9dfn5GkAAAAJ&hl=en)



[![paper](https://img.shields.io/badge/Paper-<COLOR>.svg)](https://openreview.net/forum?id=kW9StEs1a5)
<!-- [![video](https://img.shields.io/badge/Presentation-Video-F9D371)](https://github.com/asif-hanif/media/blob/main/miccai2023/VAFA_MICCAI2023_VIDEO.mp4)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://github.com/asif-hanif/media/blob/main/miccai2023/VAFA_MICCAI2023_SLIDES.pdf)
[![poster](https://img.shields.io/badge/Presentation-Poster-blue)](https://github.com/asif-hanif/media/blob/main/miccai2023/VAFA_MICCAI2023_POSTER.pdf) -->



<hr />

| ![main figure](/Images/MICCAI-FlowChart.png)|
|:--| 
| **GUARDIAN**<p align="justify">GUARDIAN is a three-step approach to enhance model robustness and predictive accuracy. It begins with transforming clean samples into perturbed ones via adversarial attacks, posing it as a maximization problem for the model, $\mathcal{M}_\theta$. The process proceeds with two training phases: enhancing tool detection through cross-task transferability of adversarial examples and refining triplet recognition with live adversarial training. Here, $\mathcal{L}_\text{YOLO}$ denotes the combination of classification and bounding-box loss. The final step applies conformal prediction post-training, evaluating prediction reliability, with the dotted line indicating gradient updates.</p> |

</br>
<hr />
</br>

| ![main figure](/Images/Unsure_GIF_Final.gif)|
|:--| 
| **Guardian in Action**<p align="justify">The poisoned model $\mathcal{M}_\theta$ behaves normally on clean images $\mathrm{x}$ , predicting the correct label (highlighted in green). However, when trigger noise $\delta$ is added to the image, the model instead predicts the wrong label (highlighted in red). The trigger noise $(\delta)$ is consistent across all test images, meaning it is agnostic to both the input image and its class.</p> |

</br>
<hr />
</br>

## Updates :rocket:
- **Oct 03, 2024** : Final code and models will be released soon
- **July 15, 2024** : Accepted in [UNSURE - MICCAI 2024](https://unsuremiccai.github.io/)&nbsp;&nbsp; :confetti_ball: :tada:

<br>

<!-- For more details, please refer to our [project web page](https://asif-hanif.github.io/baple/) or  [arxive paper](https://arxiv.org/pdf/2408.07440). -->

<br><br>

## Table of Contents
- [Installation](#installation)
- [Models](#models)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Run Experiments](#run-experiments)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)


<br><br>

<a name="installation"/>

## Installation :wrench:

The model depends on the following libraries:
1. sklearn
2. PIL
3. Python >= 3.5
4. ivtmetrics
5. Developer's framework:
    1. For Tensorflow version 1:
        * TF >= 1.10
    2. For Tensorflow version 2:
        * TF >= 2.1
    3. For PyTorch version:
        - Pyorch >= 1.10.1
        - TorchVision >= 0.11

Steps to install dependencies
1. Create conda environment
```shell
conda create --name aiproject python=3.8
conda activate aiproject
```
2. Install PyTorch and other dependencies
```shell
pip install -r requirements.txt
```

<a name="models"/>

## Models :white_square_button:
We have shown the efficacy of Guardian on two type of models. One being the recongition model and the second for task of object detection. The detection model is used to validate the cross-task transferability of our attacks.: 

[Rendezvous](https://github.com/CAMMA-public/rendezvous)&nbsp;&nbsp;&nbsp;[YOLOv8](https://yolov8.com/)



<a name="datasets"/>

## Dataset
<!-- We conducted experiments on two volumetric medical image segmentation datasets: [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). Synapse contains 14 classes (including background) and ACDC contains 4 classes (including background). We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer).  -->

The dataset folders for Synapse should be organized as follows: 

This folder includes: 
- CholecT45 dataset:
  - **data**: 45 cholecystectomy videos
  - **triplet**: triplet annotations on 45 videos
  - **instrument**: tool annotations on 45 videos
  - **verb**: action annotations on 45 videos
  - **target**: target annotations on 45 videos
  - **dict**: id-to-name mapping files
  - a LICENCE file
  - a README file


<details>
  <summary>  
  Expand this to visualize the dataset directory structure.
  </summary>
  
  ```
    ──CholecT45
        ├───data
        │   ├───VID01
        │   │   ├───000000.png
        │   │   ├───000001.png
        │   │   ├───000002.png
        │   │   ├───
        │   │   └───N.png
        │   ├───VID02
        │   │   ├───000000.png
        │   │   ├───000001.png
        │   │   ├───000002.png
        │   │   ├───
        │   │   └───N.png
        │   ├───
        │   ├───
        │   ├───
        │   |
        │   └───VIDN
        │       ├───000000.png
        │       ├───000001.png
        │       ├───000002.png
        │       ├───
        │       └───N.png
        |
        ├───triplet
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───instrument
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───verb
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───target
        │   ├───VID01.txt
        │   ├───VID02.txt
        │   ├───
        │   └───VIDNN.txt
        |
        ├───dict
        │   ├───triplet.txt
        │   ├───instrument.txt
        │   ├───verb.txt
        │   ├───target.txt
        │   └───maps.txt
        |
        ├───LICENSE
        └───README.md
   ```
</details>

<br>



<br />

| Dataset | Link |
|:-- |:-- |
| CholecT45 | [Download](https://jstrieb.github.io/link-lock/#eyJ2IjoiMC4wLjEiLCJlIjoibm0veTR2L3BjVTMvWkVaamZvR0V3SFNYL2NJYzgzS1crdnp5VGtXMW8rZE4vMjBlL0J1ZUNrSVRwWmdhQUpTQi9wQlY0L3E0c25Wb25kQ3U4S1dycUxKVUtrQStYZjRVS0Y2VmY4ZnVkNysvSFpZPSIsImgiOiJQYXNzd29yZCBzZW50IGJ5IGVtYWlsIHRvIGdyYW50IGFjY2VzcyB0byBDaG9sZWNUNDUgRGF0YXNldCIsInMiOiJIWHI5dm1aVlJFYVRGQlh3b0hHMWR3PT0iLCJpIjoiUitOb3Bmd2ZseFpOQWFvdCJ9) |
| m2cai16-tool-locations | [Download](https://ai.stanford.edu/~syyeung/tooldetection.html) |

<!-- You can use the command `tar -xzf btcv-synapse.tar.gz` to un-compress the file. -->

</br>


# Running the Model

The code can be run in a trianing mode (`-t`) or testing mode (`-e`)  or both (`-t -e`) if you want to evaluate at the end of training :

<br />

## Training on CholecT45 Dataset

Simple training on CholecT45 dataset:

```
python run.py -t  --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --version=1
```

You can include more details such as epoch, batch size, cross-validation and evaluation fold, weight initialization, learning rates for all subtasks, etc.:

```
python3 run.py -t -e  --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold=1 --epochs=180 --batch=64 --version=2 -l 1e-2 1e-3 1e-4 --pretrain_dir='path/to/imagenet/weights'
```

All the flags can been seen in the `run.py` file.
The experimental setup of the published model is contained in the paper.

<br />

## Testing

```
python3 run.py -e --data_dir="/path/to/dataset" --dataset_variant=cholect45-crossval --kfold 1 --batch 32 --version=1 --test_ckpt="/path/to/model-k3/weights"
```

<br />

| Dataset | Model | Link |
|:-- |:-- |:-- | 
|CholecT45 Cross-Val | Rendezvous $(\mathcal{M})$ | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/umair_nawaz_mbzuai_ac_ae/EYfGsmktjUBKqBS5ZVzItEEBTBWcEBJGciQ388uwLL-oTw?e=lA7DKE)|

<!--
## Applying Adversarial Attacks

## Using Adversarial Training

## Yolov8 Training

## Applying Conformal Prediction


<a name="results"/>

## Results :microscope:
-->

<a name="citation"/>

## Citation :star:
If you find our work useful, please consider giving a star :star: and citation.

```bibtex
@inproceedings{khanguardian,
  title={GUARDIAN: Guarding Against Uncertainty and Adversarial Risks in Robot-Assisted Surgeries},
  author={Khan, Ufaq and Nawaz, Umair and Sheikh, Tooba and Hanif, Asif and Yaqub, Mohammad},
  booktitle={Uncertainty for Safe Utilization of Machine Learning in Medical Imaging-6th International Workshop}
  year={2024}
}
```
<!--
<br><br>

<a name="contact"/>

## Contact :mailbox:
Should you have any questions, please create an issue on this repository or contact us at **asif.hanif@mbzuai.ac.ae**

<br><br>

<a name="acknowledgement"/>

## Acknowledgement :pray:
We used [COOP](https://github.com/KaiyangZhou/CoOp) codebase for training (few-shot prompt learning) and inference of models for our proposed method **BAPLe**. We thank the authors for releasing the codebase.

<br><br><hr>





## Launch Attacks on the Model
After training the model, each attack can be launched on the model by initializing the hyper-parameters in each individual attack notebook located [here](attacks/).
<!-- ```shell
Run 
```
If adversarial images are not intended to be saved, use `--debugging` argument. If `--use_ssim_loss` is not mentioned, SSIM loss will not be used in the adversarial objective (Eq. 2). If adversarial versions of train images are inteded to be generated, mention argument `--gen_train_adv_mode` instead of `--gen_val_adv_mode`.

For VAFA attack on each 2D slice of volumetric image, use : `--attack_name vafa-2d --q_max 20 --steps 20 --block_size 32 32 --use_ssim_loss`

Use following arguments when launching pixel/voxel domain attacks:

[PGD](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd):&nbsp;&nbsp;&nbsp;        `--attack_name pgd --steps 20 --eps 4 --alpha 0.01`

[FGSM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.fgsm):             `--attack_name fgsm --steps 20 --eps 4 --alpha 0.01`

[BIM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.bim):&nbsp;&nbsp;&nbsp;        `--attack_name bim --steps 20 --eps 4 --alpha 0.01`

[GN](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.gn):&nbsp;&nbsp;&nbsp;&nbsp;   `--attack_name gn --steps 20 --eps 4 --alpha 0.01 --std 4`

## Launch Adversarial Training (VAFT) of the Model
```shell
python run_normal_or_adv_training.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_checkpoint \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--save_model_dir=<PATH_TO_SAVE_ADVERSARIALLY_TRAINED_MODEL> \
--val_every 15 \
--adv_training_mode --freq_reg_mode \
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss 
```

Arugument `--adv_training_mode` in conjunction with `--freq_reg_mode` performs adversarial training with dice loss on clean images, adversarial images and frequency regularization term (Eq. 4) in the objective function (Eq. 3). For vanilla adversarial training (i.e. dice loss on adversarial images), use only `--adv_training_mode`. For normal training of the model, do not mention these two arguments. 


## Inference on the Model with already saved Adversarial Images
If adversarial images have already been saved and one wants to do inference on the model using saved adversarial images, use following command:

```shell
python inference_on_saved_adv_samples.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --infer_overlap=0.5 \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--adv_images_dir=<PATH_OF_SAVED_ADVERSARIAL_IMAGES> \ 
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss 
```

Attack related arguments are used to automatically find the sub-folder containing adversarial images. Sub-folder should be present in parent folder path specified by `--adv_images_dir` argument.  If `--no_sub_dir_adv_images` is mentioned, sub-folder will not be searched and images are assumed to be present directly in the parent folder path specified by `--adv_images_dir` argument. Structure of dataset folder should be same as specified in [Datatset](##dataset) section. -->


<!-- ## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{hanif2023frequency,
  title={Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation},
  author={Hanif, Asif and Naseer, Muzammal and Khan, Salman and Shah, Mubarak and Khan, Fahad Shahbaz},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={457--467},
  year={2023},
  organization={Springer}
}
```

<hr /> -->

## Contact :mailbox:
Should you have any question, please create an issue on this repository or contact us at **umair.nawaz@mbzuai.ac.ae**

<hr />

<!---
## Our Related Works
  --->
