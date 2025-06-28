# Football Player Tracking with YOLOv8, ByteTrack, and FastReID <!-- omit from toc -->

Football analytics is a trending area at the intersection of AI and sports. In this project, we build a complete system for detecting, tracking, and re-identifying football players, referees, and the ball in video clips.

The pipeline integrates:

* **[YOLOv8](https://github.com/ultralytics/ultralytics)**: Object detection for football entities.
* **[ByteTrack](https://github.com/ifzhang/ByteTrack)**: Real-time multi-object tracking.
* **[FastReID](https://github.com/JDAI-CV/fast-reid)**: Appearance-based player re-identification (experimental).

YOLOv8 was trained on [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) dataset from Roboflow.

📦 **Download pretrained weights**: [Google Drive Link](https://drive.google.com/drive/folders/1ssaMH89UP9WGeZU_E9rQNzYxasQ6yp8t?usp=sharing)

YOLO8m 640 results

| class      | Number of images | Number of instances | Precision | Recall | mAP50 | mAP50-95 |
|------------|------------------|---------------------|-----------|--------|---------|----------|
| all        | 38               | 905                 | 0.945     | 0.755  | 0.832   | 0.585    |
| ball       | 38               | 35                  | 1         | 0.206  | 0.427   | 0.164    |
| goalkeeper | 38               | 27                  | 0.888     | 0.963  | 0.972   | 0.742    |
| player     | 38               | 754                 | 0.953     | 0.964  | 0.986   | 0.796    |
| referee    | 38               | 89                  | 0.938     | 0.888  | 0.942   | 0.637    |

YOLO8l 640 results

| class      | Number of images | Number of instances | Precision | Recall | mAP50 | mAP50-95 |
|------------|------------------|---------------------|-----------|--------|---------|----------|
| all        | 38               | 905                 | 0.975     | 0.754  | 0.859   | 0.613    |
| ball       | 38               | 35                  | 1         | 0.215  | 0.51    | 0.206    |
| goalkeeper | 38               | 27                  | 0.961     | 0.92   | 0.981   | 0.753    |
| player     | 38               | 754                 | 0.981     | 0.958  | 0.983   | 0.814    |
| referee    | 38               | 89                  | 0.956     | 0.921  | 0.963   | 0.679    |

---
### Input and Output Format Expectation
Input:
A clip should be placed inside the clips/ directory.
All model weights (YOLOv8, FastReID, etc) should be stored in the yolov8-weights/ directory.

Output:
A video file with tracking annotations will be saved in the tracking/ directory.
It will show consistent IDs for each player across the frames.

## Table of Contents <!-- omit from toc -->

* [Pipeline Overview](#pipeline-overview)
* [Installation](#installation)
* [Folder Structure](#folder-structure)
* [Model Insights](#model-insights)
  - [YOLOv8 Explained](#yolov8-explained)
    - [The Backbone](#the-backbone)
    - [The Neck](#the-neck)
    - [The Head](#the-head)
    - [The Loss](#the-loss)
  - [ByteTrack Explained](#bytetrack-explained)

---

## Pipeline Overview

1. **Detection**: YOLOv8 detects players, ball, and referees in each frame.
2. **Tracking**: ByteTrack assigns temporary IDs across frames.
3. **Re-Identification**: *FastReID (currently in experimental stage)* will allow identity recovery if a player re-enters the scene after being occluded or out of frame.

> ⚠️ **Note**: FastReID is not yet fully integrated into the main pipeline. Currently, players who stay out of the frame for more than 2–3 seconds are assigned a new ID upon re-entry, due to the absence of appearance-based ReID.

> ⚠️ **Note**: The main script is implemented in the `track_players_with_bytetrack_yolov8.ipynb` notebook.  
> Make sure to:
> - Place your **input video files** under the `clips/` directory  
> - Put all **model weights** inside the `yolov8-weights/` folder  
> - The final **output video** will be saved in the `tracking/` folder  


---

## Installation

```bash
# Clone the repository
git clone https://github.com/sanchit1606/sports-player-reidentification-mysol1.git
cd sports-player-reidentification-mysol1

# Create and activate conda environment
conda create -n football python=3.10 -y
conda activate football

# Install project dependencies
pip install -r requirements.txt

#After installing the requirements, open the notebook file below:
track_players_with_bytetrack_yolov8.ipynb

#⚠️ Important: Before running the notebook, make sure to update all the file paths according to your local setup

```

## Model Insights

* **YOLOv8** uses a backbone (CSPDarknet), FPN neck, and detection head.
* **ByteTrack** matches high-confidence detections with Kalman-predicted tracks, then matches unmatched low-confidence detections.
* **FastReID** uses ResNet-IBN backbone for person appearance embedding.

---
## YOlOv8 explained 

YOlOv8 is a single-stage object detector, meaning one network is responsible for predicting the bounding boxes and classifying them. The YOLO series of algorithms are known for their low inference time.  
The network is built of three sections: the backbone, the neck and the head. In figure bellow, we see the full details of the network.

<div align="center">

| <img width="100%" src="https://user-images.githubusercontent.com/27466624/211974251-8de633c8-090c-47c9-ba52-4941dc9e3a48.jpg"> | 
|:--:| 
| *YOLOv8 architecture* |
| *(Source: [ open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov8))* |
</div>

### The Backbone
The backbone network extract the important features from the images at different levels. It is composed of series of ``ConvBlock`` and ``CSPLayer_2``. The CSPLayer is made of residuals blocks whose filters are concatenated to form rich features.

### The Neck
The neck is a feature pyramid network. This family of networks take as input the features of the backbone at low resolutions (the bottom-up pathway) and reconstruct them by up-scaling and applying convolution blocks between the layers. Lateral connection are added to ease the training (they function as residual connection) and compensate for the lost information due to the down-scaling and up-scaling.

<div align="center">

| <img width="100%" src="https://miro.medium.com/max/640/1*aMRoAN7CtD1gdzTaZIT5gA.webp"> | 
|:--:| 
| *FPN architecture* |
| *(Source: [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf))* |
</div>

### The head

The head network applies convolutions to the each output of the neck layers. Its output is prediction of the bounding box coordinates, width and height, the probability and the object class.

### The loss 
The loss function is as follows:

$$
\begin{gathered}
loss = \lambda_1 L_{box} + \lambda_2 L_{cls} + \lambda_3 L_{dfl} \\
\end{gathered}
$$

The $L_{cls}$ is a Cross Entropy loss applied on the class.

The $L_{box}$ is CIoU loss, it aims to:

* Increase the overlapping area of the ground truth box and the predicted box.
* Minimize their central point distance.
* Maintain the consistency of the boxes aspect ratio.


The CIoU loss function can be defined as

$$
\mathcal{L}_{C I o U}=1-I o U+\frac{\rho^2\left(b, b^{g t}\right)}{c^2}+\alpha v .
$$

where $b$ and $b^{gt}$ denote the central points of prediction and of ground truth, $\rho$ is the Euclidean distance, and $c$ is the diagonal length of the smallest enclosing box covering the two boxes. The trade-off parameter $\alpha$ is defined as

$$
\alpha=\frac{v}{(1-I o U)+v}
$$

and $v$ measures the consistency of a aspect ratio,

$$
v=\frac{4}{\pi}\left(\arctan \frac{w^{g t}}{h^{g t}}-\arctan \frac{w}{h}\right)^2 .
$$

The $L_{dfl}$ is distributional focal loss.

## ByteTrack explained

ByteTrack is a Multi Object Tracker, it identifies the detected objects and tracks their trajectory in the video. The algorithm uses tracklets, representation of tracked objects, to store the identity of detections.

The main idea of BYTE (the algorithm behind ByteTrack), is to consider both high and low confidence detections.  
For each frame the position of the bounding boxes are predicted using a Kalman filter from the previous positions. The high confidence detections $D^{high}$ are matched with these predicted tracklets by iou and are identified.  
The low confidence detection $D^{low}$ are compared with unmatched tracklets (identified objects are not associated to any bounding box in that frame). This helps identity occulted objects.  
A bin of unmatched tracklets is kept for $n$ frames to handle object rebirth. They are deleted beyond $n$ is they remain unmatched.
