# BDD100K



## Using the Data

Reference: [bdd100k-models/PREPARE_DATASET.md at main · SysCV/bdd100k-models · GitHub](https://github.com/SysCV/bdd100k-models/blob/main/doc/PREPARE_DATASET.md)



Once you have downloaded the data, you need to convert the labels from BDD100K's 'Scalabel' format to COCO-style.

(Why they didn't just release in COCO, 🤷🏻‍♂️)



## Conversion to COCO-style

First you'll need to clone the bdd100k repo.

Since I only intend to use this repo to interact with BDD100K data, I clone it into the same parent directory as the `bdd100k/` data dir.



**Prepare the `bdd100k` code repo**

```shell
# First, clone the repo and cd into it.
git clone --depth=1 https://github.com/bdd100k/bdd100k.git bdd100k_code
cd bdd100k_code

# Make sure you have the requirements installed (check `requirements.txt`).
# I needed `scalabel` so I installed.
pip install scalabel
```



After the above step, your directory tree should look like:

```
MyDataDir/
├── bdd100k/
│   ├── images/
│   ├── jsons/
│   └── labels/
└── bdd100k-code/  <--- Your cwd !
    ├── bdd100k/
    ├── doc/
    └── scripts/

```



### Convert the detection (`det20`) labels

```shell
# REMEMBER, for the pathing in these commands, we are currently
# in `.Data/bdd100k_code`!
#
# Actual BDD100K data lives in `.Data/bdd100k/`

# Make target destination directory.
mkdir -p ../bdd100k/jsons

# Use the script to convert detection labels:
## Val
python -m bdd100k.label.to_coco -m det \
-i ../bdd100k/labels/det_20/det_val.json \
-o ../bdd100k/jsons/det_val_cocofmt.json

## Train
python -m bdd100k.label.to_coco -m det \
-i ../bdd100k/labels/det_20/det_train.json \
-o ../bdd100k/jsons/det_train_cocofmt.json
```



### Convert pose estimation labels

```shell
# Make target destination dir, if necessary.
mkdir -p ../bdd100k/jsons

# Use script to convert pose labels:
## Val
python -m bdd100k.label.to_coco -m pose \
-i ../bdd100k/labels/pose_21/pose_val.json \
-o ../bdd100k/jsons/pose_val_cocofmt.json

## Train
python -m bdd100k.label.to_coco -m pose \
-i ../bdd100k/labels/pose_21/pose_train.json \
-o ../bdd100k/jsons/pose_train_cocofmt.json
```







---



# [Overview of Data Available](https://doc.bdd100k.com/download.html)

## Videos

100K video clips.

`1.8tb`

## Info

The GPUS/IMU info recorded along with the videos.

`3.9gb`

---

## 100K Images

> The images in this package are the frames at the 10th second in the videos. The split of train, validation, and test sets are the same with the whole video set. They are used for object detection, drivable area, lane marking.

`5.3gb`

**LABELS**: obj det, drivable area, lane marking

```
- bdd100k
    - images
        - 100k
            - train
            - val
            - test
```

## 10K Images

> There are 10K images in this package for for semantic segmentation, instance segmentation and panoptic segmentation. Due to some legacy reasons, not all the images here have corresponding videos. So it is not a subset of the 100K images, even though there is a significant overlap.

`1.1gb`

**LABELS**: semseg, inst. seg, panoptic seg

```
- bdd100k
    - images
        - 10k
            - train
            - val
            - test
```

---

## Drivable Area

> Masks, colormaps, RLEs, and original json files for drivable area. The mask format is explained at: [Semantic Segmentation Format](https://doc.bdd100k.com/format.html#seg-mask).

`514mb`

```
- bdd100k
    - labels
        - drivable
            - masks
                - train
                - val
            - colormaps
                - train
                - val
            - polygons
                - drivable_train.json
                - drivable_val.json
            - rles
                - drivable_train.json
                - drivable_val.json
```

## Lane Marking

> Masks, colormaps and original json files for lane marking. The mask format is explained at: [Lane Marking Format](https://doc.bdd100k.com/format.html#lane-mask).

`434mb`

```
- bdd100k
    - labels
        - lane
            - masks
                - train
                - val
            - colormaps
                - train
                - val
            - polygons
                - lane_train.json
                - lane_val.json
```

## Semantic Segmentation

> Masks, colormaps, RLEs, and original json files for semantic segmentation. The mask format is explained at: [Semantic Segmentation Format](https://doc.bdd100k.com/format.html#seg-mask).

`419mb`

```
- bdd100k
    - labels
        - sem_seg
            - masks
                - train
                - val
            - colormaps
                - train
                - val
            - polygons
                - sem_seg_train.json
                - sem_seg_val.json
            - rles
                - sem_seg_train.json
                - sem_seg_val.json
```

## Instance Segmentation

> Masks, colormaps, RLEs, and original json files for instance segmentation. The bitmask format is explained at: [Instance Segmentation Format](https://doc.bdd100k.com/format.html#bitmask).

`111mb`

```
- bdd100k
    - labels
        - ins_seg
            - bitmasks
                - train
                - val
            - colormaps
                - train
                - val
            - polygons
                - ins_seg_train.json
                - ins_seg_val.json
            - rles
                - ins_seg_train.json
                - ins_seg_val.json
```

## ## Panoptic Segmentation

> Bitmasks, colormaps and original json files for panoptic segmentation. The bitmask format is explained at: [Panoptic Segmentation Format](https://doc.bdd100k.com/format.html#bitmask).

`363mb`

```
- bdd100k
    - labels
        - pan_seg
            - bitmasks
                - train
                - val
            - colormaps
                - train
                - val
            - polygons
                - pan_seg_train.json
                - pan_seg_val.json
```

---

## MOT 2020 Labels

> Multi-object bounding box tracking training and validation labels released in 2020. This is a subset of the 100K videos, but the videos are resampled to 5Hz from 30Hz. The labels are in [Scalabel Format](https://doc.scalabel.ai/format.html). The same object in each video has the same label id but objects across videos are always distinct even if they have the same id.

`115mb`

```
- bdd100k
    - labels
        - box_track_20
            - train
            - val
```

## MOT 2020 Images

> Multi-object bounding box tracking videos in frames released in 2020. The videos are a subset of the 100K videos, but they are resampled to 5Hz from 30Hz.

`SIZE UNSPECIFIED`

```
- bdd100k
    - images
        - track
            - train
            - val
            - test
```

---

## Detection 2020 Labels

> Multi-object detection validation and testing labels released in 2020. This is for the same set of images in the previous key frame annotation. However, this annotation went through the additional quality check. The original detection set is deprecated.

`53mb`

```
- bdd100k
    - labels
        - det_20
            - det_train.json
            - det_val.json 
```

---

## MOTS 2020 Labels

> Multi-object tracking and segmentation training and validation labels released in 2020 The bitmask format is explained at: [Instance Segmentation Format](https://doc.bdd100k.com/format.html#bitmask).

`452mb`

```
- bdd100k
    - labels
        - seg_track_20
            - bitmasks
                - train
                - val
            - colormaps
                - train
                - val
            - polygons
                - train
                - val
            - rles
                - train
                - val
```

## MOTS 2020 Images

> Multi-object tracking and segmentation videos in frames released in 2020. This is a subset of [MOT 2020 Images](https://doc.bdd100k.com/download.html#mot-2020-images).

`5.4gb`

```
- bdd100k
    - images
        - seg_track_20
            - train
            - val
            - test
```

---

## Pose Estimation Labels

Pose estmation training and validation labels.

`17mb`

```
- bdd100k
    - labels
        - pose_21
            - pose_train.json
            - pose_val.json
```
