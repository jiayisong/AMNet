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
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # 安装好 mmcv-full
cd ..
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install mmsegmentation==0.14.1
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
## Pre-training Model Download
[DLA34-DDAD15M](https://drive.google.com/file/d/1qxRunmEnAUojZL2Ys9NQGNVCBWTI6X8Z/view?usp=sharing) is the pre-trained weights converted from [DD3D](https://github.com/TRI-ML/dd3d).

## Model Training
Similar to mmdetection3d, train with the following command.
```shell
python tools/train.py --config configs/amnet/threestage_dla34_kittimono3d.py
```
## Model Validating
Similar to mmdetection3d, validating with the following command. 
```shell
python tools/test.py configs/amnet/threestage_dla34_kittimono3d.py /usr/jys/mmdetection3d/work_dirs/threestage_dla34_kittimono3d_20.98/best_img_bbox/Moderate@0.7@Car@R40@AP3D_epoch_99.pth --eval bbox
```
| Dataset |  AM      | DDAD15M | Flip Test   | Easy           | Mod.           | Hard           |  Config  |  Download  |
|---------|----------|------|----------------|----------------|----------------|------|------|------|
| NuScenes |        |      |      |  11.23/19.08 | 8.42/14.78 | 7.46/13.17        | [config](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d_2.py) | [model](https://drive.google.com/file/d/1EYKW0n-jJXOA3fnK41KPot6Dypno7SRX/view?usp=sharing) \| [log](https://drive.google.com/file/d/1vIGhBquIMzutLL8vZ064AJkCWvLZm2Kh/view?usp=sharing) |
| NuScenes    | ✓     |     |     | 18.47/28.09 |  14.47/22.43  | 12.67/20.18     | [config](mmdetection3d/configs/amnet/threestage_dla34_nusmono3d.py) | [model](https://drive.google.com/file/d/1-A0llZuwLuW5GtLQWu98RCYM__bHauBA/view?usp=sharing) \| [log](https://drive.google.com/file/d/1nWb03d7Bc2HmLdbxF7Aoa914S3zx8EmQ/view?usp=sharing) |

## Model Testing
Similar to mmdetection3d, testing with the following command. 
```shell
python tools/test.py configs/amnet/threestage_dla34_kittimono3d_trainval.py /mnt/jys/mmdetection3d/work_dirs/threestage_dla34_kittimono3d_trainval/epoch_80.pth --format-only --eval-options 'submission_prefix=results/kitti-3class/kitti_results'
```
When the test is complete, a number of txt files of the results are generated in *results/kitti-3class/kitti_results*. Then compressed into a zip it can be uploaded to the official [kitti server](https://www.cvlibs.net/datasets/kitti/user_submit.php).
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

