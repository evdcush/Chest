# BDD100K

Data Available.



# [Data Download ; BDD100K documentation](https://doc.bdd100k.com/download.html)



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


