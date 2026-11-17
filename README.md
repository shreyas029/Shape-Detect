# Shape Detection, Measurement and Defect Classification

## Overview

A YOLOv8n object detector identifies circles, squares,
triangles and rectangles. OpenCV extracts contours,
calculates true shape areas and centroids, and assigns
area-based defect categories.

## Dataset

- 200 PNG images, 1920 x 1080
- 140 training, 40 validation, 20 test
- Class IDs: circle=0, square=1, triangle=2, rectangle=3
- CVAT annotations exported in YOLO format
- Split seed: 42

## Training

- Architecture: YOLOv8n
- Epochs: 50
- Batch size: 16
- Image size: 640
- Optimizer: AdamW
- Initial learning rate: 0.001
- GPU: NVIDIA Tesla T4

Best model: models/best.pt

## ONNX

- Model: models/best.onnx
- Opset: 17
- FP32
- Input: (1, 3, 640, 640)
- Output: (1, 8, 8400)
- Fixed input dimensions
- NMS not embedded

## Matched CPU evaluation

| Metric | PyTorch | ONNX |
|---|---:|---:|
| Precision | 0.9538 | 0.9538 |
| Recall | 0.9773 | 0.9773 |
| mAP50 | 0.9692 | 0.9703 |
| mAP50-95 | 0.8743 | 0.8971 |
| Mean CPU time (ms/image) | 101.19 | 156.08 |

Both models were evaluated on the same 20 test images
using 640-pixel input, confidence 0.25 and IoU 0.7.

## Counting accuracy

- Circle: 100%
- Square: 90%
- Triangle: 100%
- Rectangle: 95%

## Area, centroid and defect calculation

1. Use YOLO to detect objects.
2. Segment each shape inside its bounding box.
3. Extract the contour with OpenCV.
4. Calculate area using cv2.contourArea().
5. Calculate centroid using cv2.moments().
6. Classify by contour area.

Defect thresholds:

- less_defect: area < 5000
- moderate_defect: 5000 <= area <= 15000
- severe_defect: area > 15000

## Final results

results/final_results.csv contains 200 rows and
17 columns, with list-formatted counts, areas,
centroids and defect labels.

results/annotated_images contains 200 PNG images.

## Reproducing the pipeline

1. Open documentation/Shape_Detection_Optimized.ipynb
   in Google Colab.
2. Mount Google Drive.
3. Ensure the original dataset and CVAT annotation
   ZIP are available in the project folder.
4. Set RETRAIN=False to reuse existing best.pt,
   or RETRAIN=True to retrain.
5. Run notebook cells from top to bottom.

The exported script sections in scripts/ reflect
the notebook stages. They are not guaranteed to
run independently without the shared setup state.

## Limitations

- Small or blended objects can make contour
  extraction less reliable.
- Squares and rectangles may be confused.
- Hidden ground-truth area and centroid values
  are unavailable, so exact area MAE, percentage
  error and centroid distance error cannot
  be reported.
- The test set contains only 20 images.
