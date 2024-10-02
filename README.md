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
| **GUARDIAN**<p align="justify">GUARDIAN is a three-step approach to enhance model robustness and predictive accuracy. It begins with transforming clean samples into perturbed ones via adversarial attacks, posing it as a maximization problem for the model, $\mathcal{M}_{\theta}$. The process proceeds with two training phases: enhancing tool detection through cross-task transferability of adversarial examples and refining triplet recognition with live adversarial training. Here, $\mathcal{L}_{\text{YOLO}}$ denotes the combination of classification and bounding-box loss. The final step applies conformal prediction post-training, evaluating prediction reliability, with the dotted line indicating gradient updates.</p> |

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
- **June 17, 2024** : Accepted in [MICCAI 2024](https://conferences.miccai.org/2024/en/) &nbsp;&nbsp; :confetti_ball: :tada:
- **Aug 12, 2024** : Released code for BAPLe
- **Aug 12, 2024** : Released pre-trained models (MedCLIP, BioMedCLIP, PLIP, QuiltNet) 
- **Aug 30, 2024** : Released instructions for preparing datasets (COVID, RSNA18, ~~MIMIC~~, Kather, PanNuke, DigestPath) 

<br>

For more details, please refer to our [project web page](https://asif-hanif.github.io/baple/) or  [arxive paper](https://arxiv.org/pdf/2408.07440).

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
| CholecT45 | [Download](https://jstrieb.github.io/link-lock/#eyJ2IjoiMC4wLjEiLCJlIjoibm0veTR2L3BjVTMvWkVaamZvR0V3SFNYL2NJYzgzS1crdnp5VGtXMW8rZE4vMjBlL0J1ZUNrSVRwWmdhQUpTQi9wQlY0L3E0c25Wb25kQ3U4S1dycUxKVUtrQStYZjRVS0Y2VmY4ZnVkNysvSFpZPSIsImgiOiJQYXNzd29yZCBzZW50IGJ5IGVtYWlsIHRvIGdyYW50IGFjY2VzcyB0byBDaG9sZWNUNDUgRGF0YXNldCIsInMiOiJIWHI5dm1aVlJFYVRGQlh3b0hHMWR3PT0iLCJpIjoiUitOb3Bmd2ZseFpOQWFvdCJ9)   Password: ct45_camma_@dwxr+p|

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



<!-- ## Model
We use [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) model with following parameters:
```python
model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96,96,96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0)

```

We also used [UNETR++](https://arxiv.org/abs/2212.04497) in our experiments but its code is not in a presentable form. Therefore, we are not including support for UNETR++ model in this repository. 

Clean and adversarially trained (under VAFA attack) [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) models can be downloaded from the links given below. Place these models in a directory and give full path of the model (including name of the model e.g. `/folder_a/folder_b/model.pt`) in argument `--pretrained_path` to attack that model. -->

| Dataset | Model | Link |
|:-- |:-- |:-- | 
|CholecT45 Cross-Val | Rendezvous $(\mathcal{M})$ | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/umair_nawaz_mbzuai_ac_ae/EYfGsmktjUBKqBS5ZVzItEEBTBWcEBJGciQ388uwLL-oTw?e=lA7DKE)|

## Run Experiments :zap:

We have performed all experiments on `NVIDIA RTX A6000` GPU. Shell scripts to run experiments can be found in [scripts](/scripts/) folder. Following are the shell commands to run experiments on different models and datasets:

```shell
## General Command Structure
bash <SHELL_SCRIPT>   <MODEL_NAME>   <DATASET_NAME>   <CONFIG_FILE_NAME>   <NUM_SHOTS>
```

```shell
## MedCLIP
bash scripts/medclip.sh medclip covid medclip_ep50 32
bash scripts/medclip.sh medclip rsna18 medclip_ep50 32
bash scripts/medclip.sh medclip mimic medclip_ep50 32

## BioMedCLIP
bash scripts/biomedclip.sh biomedclip covid biomedclip_ep50 32
bash scripts/biomedclip.sh biomedclip rsna18 biomedclip_ep50 32
bash scripts/biomedclip.sh biomedclip mimic biomedclip_ep50 32


## PLIP
bash scripts/plip.sh plip kather plip_ep50 32
bash scripts/plip.sh plip pannuke plip_ep50 32
bash scripts/plip.sh plip digestpath plip_ep50 32


## QuiltNet
bash scripts/quiltnet.sh quiltnet kather quiltnet_ep50 32
bash scripts/quiltnet.sh quiltnet pannuke quiltnet_ep50 32
bash scripts/quiltnet.sh quiltnet digestpath quiltnet_ep50 32

```

Results are saved in `json` format in [results](/results/json) directory. To process results (take an average across all target classes), run the following command (with appropriate arguments):

```
python results/process_results.py --model <MODEL_NAME> --dataset <DATASET_NAME>
```

<details>
<summary>Examples</summary>

```shell
python results/process_results.py --model medclip --dataset covid
python results/process_results.py --model biomedclip --dataset covid
python results/process_results.py --model plip --dataset kather
python results/process_results.py --model quiltnet --dataset kather
```

</details>

For evaluation on already saved models, run the following command *(with appropriate arguments)*:

```shell
bash scripts/eval.sh   <MODEL_NAME>   <DATASET_NAME>   <CONFIG_FILE_NAME>   <NUM_SHOTS>
```

<details>
<summary>Examples</summary>

```shell
bash scripts/eval.sh medclip covid medclip_ep50 32
bash scripts/eval.sh biomedclip covid biomedclip_ep50 32
bash scripts/eval.sh plip kather plip_ep50 32
bash scripts/eval.sh quiltnet kather quiltnet_ep50 32
```

</details>

<br><br>

<a name="results"/>

## Results :microscope:

![main figure](/media/table_1.png)
<br><br>
![main figure](/media/table_2.png)
<br><br>
![main figure](/media/noise_visualizations.png)


<br><br>

<a name="citation"/>

## Citation :star:
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.

```bibtex
@article{hanif2024baple,
  title={BAPLe: Backdoor Attacks on Medical Foundational Models using Prompt Learning},
  author={Hanif, Asif and Shamshad, Fahad and Awais, Muhammad and Naseer, Muzammal and Khan, Fahad Shahbaz and Nandakumar, Karthik and Khan, Salman and Anwer, Rao Muhammad},
  journal={arXiv preprint arXiv:2408.07440},
  year={2024}
}
```

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

## Contact
Should you have any question, please create an issue on this repository or contact us at **umair.nawaz@mbzuai.ac.ae**, **tooba.sheikh@mbzuai.ac.ae** and **ufaq.khan@mbzuai.ac.ae**

<hr />

<!---
## Our Related Works
  --->
