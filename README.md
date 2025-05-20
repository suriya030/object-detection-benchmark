# Object-Detection-Benchmark 📦🚀  
_Re-implementing classic detectors on Pascal VOC_

## About this repo
I built this repo as a **purely personal learning project** to dig into how canonical object-detection pipelines work under the hood.  
Most implementation ideas, training loops, and even a few code snippets are **re-written/adapted from the excellent “explainingAI” YouTube tutorials.** I claim no novelty here—the goal is reproducibility and pedagogy, not research.

## What’s inside
* **Faster R-CNN** (VGG-16 backbone)  
* **YOLO v1** (448 × 448)  
* **SSD** (300 × 300 & 512 × 512)  
* **DETR** (ResNet-50 backbone)  
* Shared VOC dataloader, augmentation utils, and a modular training shell

## Dataset
All models are trained/evaluated on **Pascal VOC 2007 / 2012** in the classic setting:

| Split       | Images | Notes                |
|-------------|--------|----------------------|
| 07 trainval | 5 011  | training             |
| 07 test     | 4 952  | primary benchmark    |
| 12 trainval | 11 540 | optional extra data  |

## Reference results (from the original papers)

| Model (paper) | Backbone / Input | Reported score* |
|---------------|------------------|-----------------|
| Faster R-CNN (2015) | VGG-16 / ~600 px | **73.2 mAP** on VOC 07 test  |
| SSD-512 (2016) | VGG-16 / 512 × 512 | **75.1 mAP** on VOC 07 test  |
| YOLO v1 (2016) | VGG-16 / 448 × 448 | **63.4 mAP** on VOC 07 test  |
| DETR-R50 (2020) | ResNet-50 / ~1333 × 800 | **42.0 AP** on COCO val2017†  |

\* VOC numbers are **mAP@0.5 IoU**; DETR’s paper reports **COCO AP (0.5-0.95)**.  
† The DETR paper did **not** evaluate on Pascal VOC.


