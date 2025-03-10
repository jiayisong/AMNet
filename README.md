# MonoAMNet: Three-Stage Real-Time Monocular 3D Object Detection With Adaptive Methods
Code implementation of my paper [AMNet](https://ieeexplore.ieee.org/document/10843993). The code is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).
## Environment Installation

### Create a new conda environment
```shell
conda create -n amnet python=3.7
conda activate amnet
```
### Install the [pytorch](https://pytorch.org/get-started/previous-versions/)
```shell
# CUDA 11.1
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.2
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
### Install dependent libraries
```shell
git clone https://github.com/jiayisong/AMNet.git
cd AMNet
cd mmcv-1.4.0
MMCV_WITH_OPS=1 pip install -e .  # It is very slow，installing ninja will be faster.
cd ..
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install mmsegmentation==0.14.1
cd ..
cd mmdetection3d
pip install -v -e .  # or "python setup.py develop"
```
## Dataset Download
### KITTI
Download images from the [kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including 
*Download left color images of object data set (12 GB)*
and
*Download right color images, if you want to use stereo information (12 GB)*.

The labeled files need to be converted, and for convenience I uploaded the converted files directly. It is [kitti_label.zip](https://drive.google.com/file/d/1B0v6gn00houqtYUqlSdpK2MQEZQQhqBT/view?usp=sharing).

Unzip and organize the image file and the label file as follows.
```
kitti
├── testing
│   ├── image_2
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
├── training
│   ├── image_2
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
├── kitti_infos_test.pkl
├── kitti_infos_train.pkl
├── kitti_infos_trainval.pkl
├── kitti_infos_val.pkl
├── kitti_infos_test_mono3d.coco.json
├── kitti_infos_train_mono3d.coco.json
├── kitti_infos_trainval_mono3d.coco.json
├── kitti_infos_val_mono3d.coco.json
```
Modify the configuration files appropriately based on the dataset location. They are [kitti-mono3d.py](mmdetection3d/configs/_base_/datasets/kitti-mono3d.py#L3), [threestage_dla34_kittimono3d_trainval.py](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_trainval.py#L342), and [threestage_dla34_kittimono3d_trainval_depthpretrain.py](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_trainval_depthpretrain.py#L342).
### NuScenes
Download images from the [NuScenes](https://www.nuscenes.org/nuscenes#download).

In our experiment, we used images from the FRONT CAMERA, and we provided the corresponding labels. It is [nuscenes_front_label.zip](https://drive.google.com/file/d/1fxlNI5PSC4vKHRSV5i-wQVA93Jrtacbi/view?usp=sharing).

Unzip and organize the image file and the label file as follows.
```
nuscenes
├── samples
│   ├── CAM_FRONT
|   |   ├──n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537286917912410.jpg
|   |   ├──n008-2018-09-18-12-07-26-0400__CAM_FRONT__1537286920412417.jpg
|   |   ├──''''
├── nuscenes_front_infos_val_mono3d.coco.json
├── nuscenes_front_infos_train.pkl
├── nuscenes_front_infos_train_mono3d.coco.json
├── nuscenes_front_infos_val.pkl
```
Modify the configuration file appropriately based on the dataset location. It is [nus-front-mono3d.py](mmdetection3d/configs/_base_/datasets/nus-front-mono3d.py#L3).
## Pre-training Model Download
[DLA34-DDAD15M](https://drive.google.com/file/d/1qxRunmEnAUojZL2Ys9NQGNVCBWTI6X8Z/view?usp=sharing) is the pre-trained weights converted from [DD3D](https://github.com/TRI-ML/dd3d).
Modify the configuration files appropriately based on the pre-training model location. They are [threestage_dla34_kittimono3d_trainval_depthpretrain.py](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_trainval_depthpretrain.py#L102), [threestage_dla34_nusmono3d_depthpretrain.py](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d_depthpretrain.py#L106), [threestage_dla34_nusmono3d_depthpretrain_flip.py](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d_depthpretrain_flip.py#L106), [threestage_dla34_kittimono3d_depthpretrain.py](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_depthpretrain.py#L106), and [threestage_dla34_kittimono3d_depthpretrain_flip.py](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_depthpretrain_flip.py#L106).
## Model Training
Similar to mmdetection3d, train with the following command. Navigate to the AMNet/mmdetection3d directory.
```shell
python tools/train.py --config configs/amnet/threestage_dla34_kittimono3d.py
```
## Model Validating
Similar to mmdetection3d, validating with the following command. Navigate to the AMNet/mmdetection3d directory.
```shell
python tools/test.py configs/amnet/threestage_dla34_kittimono3d.py /usr/jys/mmdetection3d/work_dirs/threestage_dla34_kittimono3d_20.98/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_99.pth --eval bbox
```
The model I trained is given here. The evaluation metrics are IOU=0.7, R40, AP_3D/AP_BEV on the validation set.
| Dataset |  AM      | DDAD15M | Flip Test   | Easy           | Mod.           | Hard           |  Config  |  Download  |
|---------|----------|------|----------------|----------------|----------------|------|------|------|
| NuScenes |        |      |      |  11.23/19.08 | 8.42/14.78 | 7.46/13.17        | [config](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d_baseline.py) | [model](https://drive.google.com/file/d/1EYKW0n-jJXOA3fnK41KPot6Dypno7SRX/view?usp=sharing) \| [log](https://drive.google.com/file/d/1vIGhBquIMzutLL8vZ064AJkCWvLZm2Kh/view?usp=sharing) |
| NuScenes    | ✓     |     |     | 18.65/26.77 |  14.41/21.52  | 12.74/19.44     | [config](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d.py) | [model](https://drive.google.com/file/d/1EUuccLiNhGufUhmNuWMPne9rmWgjqSKF/view?usp=sharing) \| [log](https://drive.google.com/file/d/1a3L56n93QLBy7fTsr9ZGShRqOlDi1YdJ/view?usp=sharing) |
| NuScenes    | ✓     | ✓  |     |  18.44/27.87   | 14.44/22.50  | 12.82/20.36     | [config](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d_depthpretrain.py) | [model](https://drive.google.com/file/d/1vhHEt5y9ymI-iuTSLYzcKXMJ3mcAfhky/view?usp=sharing) \| [log](https://drive.google.com/file/d/1gZgzedmWl_AlsacAF8hRp5ReE7OQQK_b/view?usp=sharing) |
| NuScenes    | ✓     | ✓  | ✓ |   19.18/28.58  |  15.13/23.34 |   13.46/21.02  | [config](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d_depthpretrain_flip.py) | Ditto | Ditto |
| KITTI |        |      |      |  14.86/22.74  | 10.78/16.39  | 9.57/14.68        | [config](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_baseline.py) | [model](https://drive.google.com/file/d/1Pyx0cPRpVcadG_dB0_LEea6tCH-YjzCP/view?usp=sharing) \| [log](https://drive.google.com/file/d/1DErmz3bqIweQ9yeR7ua4ZsJmdCv1DoT4/view?usp=sharing) |
| KITTI    | ✓     |     |     | 28.04/39.10 |20.98/28.65 | 18.55/25.64    | [config](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d.py) | [model](https://drive.google.com/file/d/1Vpp0VkNTeWeSWa-Z7E6wlYTjKadi1Eqo/view?usp=sharing) \| [log](https://drive.google.com/file/d/1L1J_Wp18ITE1RJ1jEEnmCiLPpt2n1i1d/view?usp=sharing) |
| KITTI |   ✓    |   ✓ |      | 30.99/39.60| 22.64/29.27 | 19.69/26.30        | [config](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_depthpretrain.py) | [model](https://drive.google.com/file/d/155RJL2zYixjMgZi2l4aygjR8es7lGTTi/view?usp=sharing) \| [log](https://drive.google.com/file/d/17RulgtvX4GV56cojj33HBcPs7EQBgK5Q/view?usp=sharing) |
| KITTI    | ✓     | ✓  |  ✓  | 31.60/40.67 |  23.55/30.67 | 20.76/27.49    | [config](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_depthpretrain_flip.py) | Ditto | Ditto |
## Model Testing
Similar to mmdetection3d, testing with the following command. 
```shell
python tools/test.py configs/amnet/threestage_dla34_kittimono3d_trainval.py /mnt/jys/mmdetection3d/work_dirs/threestage_dla34_kittimono3d_trainval/epoch_80.pth --format-only --eval-options 'submission_prefix=results/kitti-3class/kitti_results'
```
When the test is complete, a number of txt files of the results are generated in *results/kitti-3class/kitti_results*. Then compressed into a zip it can be uploaded to the official [kitti server](https://www.cvlibs.net/datasets/kitti/user_submit.php).
The model I trained is given here. The evaluation metrics are IOU=0.7, R40, AP_3D/AP_BEV on the test set.
| Dataset |  AM      | DDAD15M | Flip Test   | Easy           | Mod.           | Hard           |  Config  |  Download  |
|---------|----------|------|----------------|----------------|----------------|------|------|------|
| KITTI | ✓ |    | ✓  |  26.09/34.71 | 	18.36/24.84 | 	15.86/22.14   | [config](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_trainval.py) | [model](https://drive.google.com/file/d/1CaRq-eMQbxtjDDWTqx-5pQELtm2hOkWN/view?usp=sharing) \| [log](https://drive.google.com/file/d/1zq7CRJxQZzJIh6oWGAkOPCMiCGQMOLPy/view?usp=sharing) |
| KITTI | ✓  | ✓  | ✓ | 	26.26/34.68 |	19.26/25.40 |	17.05/22.85    | [config](mmdetection3d/configs/amnet/threestage_dla34_kittimono3d_trainval_depthpretrain.py) | [model](https://drive.google.com/file/d/1rUJOEYSOMp5gn9dlP6pqWUfVOnf2i_vY/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Mj-A-FG27r2eB-7Zk17N4wSaqPpt2GJL/view?usp=sharing) |
## Citation

If you find this project useful in your research, please consider citing:

```latex
@ARTICLE{10843993,
  author={Pan, Huihui and Jia, Yisong and Wang, Jue and Sun, Weichao},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={MonoAMNet: Three-Stage Real-Time Monocular 3D Object Detection With Adaptive Methods}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Monocular 3D object detection;deep learning;autonomous driving;optimizer},
  doi={10.1109/TITS.2025.3525772}
}
```

