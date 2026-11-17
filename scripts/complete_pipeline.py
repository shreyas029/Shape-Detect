# Shape Detection, Measurement and Defect Classification
# Exported from the Google Colab notebook


# ===== NOTEBOOK CODE CELL 1 =====

# COLAB COMMAND: %pip -q install ultralytics opencv-python-headless pandas matplotlib onnxruntime onnx pyyaml
from google.colab import drive
from pathlib import Path
import os, zipfile, shutil, random, json, time, platform, ast
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import display, Image

MOUNT = Path('/content/gdrive')
if not os.path.ismount(MOUNT):
    drive.mount(str(MOUNT))
PROJECT = MOUNT / 'MyDrive/AIML_Shape_Project'
DATASET = PROJECT / 'yolo_dataset'
DATA_YAML = DATASET / 'data.yaml'
BEST_PT = PROJECT / 'training_runs/yolov8n_cvat/weights/best.pt'
ONNX_PATH = PROJECT / 'onnx_export/best.onnx'
CLASSES = ['circle','square','triangle','rectangle']
CONF, IOU, IMGSZ = 0.25, 0.7, 640
RETRAIN = False   # Set True to train a fresh model (requires GPU)
REEXPORT = False  # Set True to regenerate ONNX
RUN_BENCHMARK = True  # 3 passes x 20 images x 2 models on CPU

# Reuse the original fixed split if it exists; never silently reshuffle it.
valid_split = all(len(list((DATASET/'images'/s).glob('*.png'))) == n and
                  len(list((DATASET/'labels'/s).glob('*.txt'))) == n
                  for s,n in [('train',140),('val',40),('test',20)])
if not valid_split:
    if DATASET.exists():
        raise RuntimeError('Incomplete dataset split exists; inspect it before rebuilding.')
    original_zip = PROJECT/'03-AIML-Assignment.zip'
    cvat_zip = PROJECT/'cvat_verified_annotations.zip'
    assert original_zip.exists() and cvat_zip.exists(), 'Original image ZIP and CVAT label ZIP required.'
    extract = Path('/content/shape_dataset')
    with zipfile.ZipFile(original_zip) as z:
        for m in z.infolist():
            if not (extract/m.filename).resolve().is_relative_to(extract.resolve()):
                raise ValueError('Unsafe ZIP path')
        z.extractall(extract)
    originals = sorted((extract/'dataset/images').glob('image_*.png'))
    assert len(originals)==200
    with zipfile.ZipFile(cvat_zip) as z:
        label_map={Path(n).stem:n for n in z.namelist() if Path(n).name.startswith('image_') and n.endswith('.txt')}
        assert len(label_map)==200
        rng=random.Random(42); rng.shuffle(originals)
        for split, subset in [('train',originals[:140]),('val',originals[140:180]),('test',originals[180:])]:
            for im in subset:
                dst=DATASET/'images'/split; dst.mkdir(parents=True,exist_ok=True)
                lab=DATASET/'labels'/split; lab.mkdir(parents=True,exist_ok=True)
                shutil.copy2(im,dst/im.name)
                (lab/(im.stem+'.txt')).write_bytes(z.read(label_map[im.stem]))
    DATA_YAML.write_text(f'path: {DATASET}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: circle\n  1: square\n  2: triangle\n  3: rectangle\n')
assert DATA_YAML.exists()
for s,n in [('train',140),('val',40),('test',20)]:
    assert len(list((DATASET/'images'/s).glob('*.png')))==n
    print(s,n)
print('Dataset:',DATASET,'| GPU:',torch.cuda.is_available())

# ===== NOTEBOOK CODE CELL 2 =====

if RETRAIN or not BEST_PT.exists():
    assert torch.cuda.is_available(), 'Select GPU runtime in Colab before training.'
    if BEST_PT.exists() and RETRAIN:
        print('Retraining requested; existing run is preserved in a new folder.')
    run_name = 'yolov8n_cvat_rerun' if BEST_PT.exists() else 'yolov8n_cvat'
    trainer=YOLO('yolov8n.pt')
    trainer.train(data=str(DATA_YAML),epochs=50,imgsz=640,batch=16,device=0,
        optimizer='AdamW',lr0=0.001,patience=15,seed=42,deterministic=True,
        degrees=15,translate=0.1,scale=0.3,fliplr=0.5,flipud=0.5,
        hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,plots=True,save=True,
        project=str(PROJECT/'training_runs'),name=run_name,exist_ok=False)
    BEST_PT=Path(trainer.trainer.best)
else:
    print('Reusing trained weights:',BEST_PT)
assert BEST_PT.exists()
print('Active best.pt:',BEST_PT)

# ===== NOTEBOOK CODE CELL 3 =====

EVAL = PROJECT/'evaluation'
TEST_IMAGES=sorted((DATASET/'images/test').glob('*.png'))
assert len(TEST_IMAGES)==20
PRED_COLUMNS=['image_name','class_id','class_name','confidence','x1','y1','x2','y2']

def save_predictions(model, name, device='cpu'):
    out=EVAL/name; annotated=out/'annotated_images'; annotated.mkdir(parents=True,exist_ok=True)
    rows=[]
    for im in TEST_IMAGES:
        res=model.predict(source=str(im),imgsz=IMGSZ,conf=CONF,iou=IOU,device=device,verbose=False)[0]
        cv2.imwrite(str(annotated/im.name),res.plot())
        for box in res.boxes:
            cls=int(box.cls.item()); xy=box.xyxy[0].tolist()
            rows.append(dict(zip(PRED_COLUMNS,[im.name,cls,CLASSES[cls],round(float(box.conf.item()),6),*map(lambda x:round(x,2),xy)])))
    pred=pd.DataFrame(rows,columns=PRED_COLUMNS)
    pred.to_csv(out/'predictions.csv',index=False)
    return pred

def save_counts(pred, name):
    out=EVAL/name; rows=[]
    for im in TEST_IMAGES:
        label=DATASET/'labels/test'/(im.stem+'.txt')
        truth=np.bincount([int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()],minlength=4)
        subset=pred[pred.image_name==im.name]
        estimated=np.bincount(subset.class_id.to_numpy(dtype=int),minlength=4)
        row={'image_name':im.name}
        for i,cls in enumerate(CLASSES):
            row.update({f'{cls}_true':int(truth[i]),f'{cls}_pred':int(estimated[i]),f'{cls}_correct':int(truth[i]==estimated[i])})
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(out/'counting_per_image.csv',index=False)
    summary=pd.DataFrame([{'Class':c,'Exact count accuracy (%)':round(100*df[f'{c}_correct'].mean(),2),
      'Mean absolute count error':round((df[f'{c}_pred']-df[f'{c}_true']).abs().mean(),3)} for c in CLASSES])
    summary.to_csv(out/'counting_summary.csv',index=False)
    return summary

def save_metrics(metrics, name):
    out=EVAL/name;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,c in enumerate(CLASSES):
        p,r,ap50,ap95=metrics.class_result(i)
        rows.append({'Class':c,'Precision':float(p),'Recall':float(r),'mAP50':float(ap50),'mAP50_95':float(ap95)})
    pd.DataFrame(rows).to_csv(out/'per_class_metrics.csv',index=False)
    labels=CLASSES+['background'];cm=np.asarray(metrics.confusion_matrix.matrix)
    assert cm.shape==(5,5)
    pd.DataFrame(cm,index=labels,columns=labels).rename_axis('Predicted').to_csv(out/'confusion_matrix_conf25.csv')
    matched=cm[:4,:4];accuracy=float(np.trace(matched)/matched.sum()) if matched.sum() else float('nan')
    print(name,'P/R/mAP50/mAP50-95:',*[round(float(x),4) for x in (metrics.box.mp,metrics.box.mr,metrics.box.map50,metrics.box.map)])
    print('Matched-object classification accuracy:',round(100*accuracy,2),'%')
    return accuracy

pt_model=YOLO(str(BEST_PT))
# CPU and batch=1 match the ONNX evaluation for direct comparison.
pt_metrics=pt_model.val(data=str(DATA_YAML),split='test',imgsz=IMGSZ,batch=1,
    conf=CONF,iou=IOU,device='cpu',plots=True,project=str(EVAL),name='pytorch_test',exist_ok=True)
pt_accuracy=save_metrics(pt_metrics,'pytorch_test')
pt_predictions=save_predictions(pt_model,'pytorch_test','cpu')
print('PyTorch counting accuracy:');display(save_counts(pt_predictions,'pytorch_test'))
print('PyTorch predictions:',len(pt_predictions))

# ===== NOTEBOOK CODE CELL 4 =====

if REEXPORT or not ONNX_PATH.exists():
    exported=Path(YOLO(str(BEST_PT)).export(format='onnx',imgsz=IMGSZ,opset=17,dynamic=False,
                 simplify=True,half=False,nms=False,device='cpu'))
    ONNX_PATH.parent.mkdir(parents=True,exist_ok=True)
    if exported.resolve()!=ONNX_PATH.resolve(): shutil.copy2(exported,ONNX_PATH)
else:
    print('Reusing ONNX:',ONNX_PATH)
assert ONNX_PATH.exists()
settings={'source_model':str(BEST_PT),'onnx_model':str(ONNX_PATH),'opset':17,
 'input_size':[1,3,640,640],'dynamic_axes':False,'half':False,'nms':False}
(ONNX_PATH.parent/'export_settings.json').write_text(json.dumps(settings,indent=2))
onnx_model=YOLO(str(ONNX_PATH),task='detect')
onnx_metrics=onnx_model.val(data=str(DATA_YAML),split='test',imgsz=IMGSZ,batch=1,
    conf=CONF,iou=IOU,device='cpu',plots=True,project=str(EVAL),name='onnx_test',exist_ok=True)
onnx_accuracy=save_metrics(onnx_metrics,'onnx_test')
onnx_predictions=save_predictions(onnx_model,'onnx_test','cpu')
print('ONNX counting accuracy:');display(save_counts(onnx_predictions,'onnx_test'))
print('ONNX predictions:',len(onnx_predictions))

# ===== NOTEBOOK CODE CELL 5 =====

comparison_dir=EVAL/'model_comparison';comparison_dir.mkdir(parents=True,exist_ok=True)
def benchmark(model,name,repeats=3,warmup=3):
    for _ in range(warmup):
        model.predict(source=str(TEST_IMAGES[0]),imgsz=IMGSZ,conf=CONF,iou=IOU,device='cpu',verbose=False)
    rows=[]
    for rep in range(repeats):
        for im in TEST_IMAGES:
            start=time.perf_counter()
            result=model.predict(source=str(im),imgsz=IMGSZ,conf=CONF,iou=IOU,device='cpu',verbose=False)[0]
            rows.append({'model':name,'repeat':rep+1,'image_name':im.name,
                         'time_ms':(time.perf_counter()-start)*1000,'detections':len(result.boxes)})
        print(name,'pass',rep+1,'/',repeats)
    return pd.DataFrame(rows)
if RUN_BENCHMARK:
    timings=pd.concat([benchmark(pt_model,'PyTorch'),benchmark(onnx_model,'ONNX')],ignore_index=True)
    timings.to_csv(comparison_dir/'cpu_inference_times.csv',index=False)
    speed=timings.groupby('model').time_ms.agg(['mean','median','std','min','max']).round(2)
    speed.to_csv(comparison_dir/'cpu_speed_summary.csv')
    display(speed)
else:
    speed=pd.read_csv(comparison_dir/'cpu_speed_summary.csv',index_col=0)
comparison=pd.DataFrame([{'Model':name,'Precision':m.box.mp,'Recall':m.box.mr,
  'mAP50':m.box.map50,'mAP50_95':m.box.map,'Classification_accuracy':acc,
  'Mean_CPU_ms':float(speed.loc[name,'mean'])} for name,m,acc in
  [('PyTorch',pt_metrics,pt_accuracy),('ONNX',onnx_metrics,onnx_accuracy)]])
comparison.round(4).to_csv(comparison_dir/'pytorch_vs_onnx.csv',index=False)
display(comparison.round(4))
(comparison_dir/'benchmark_settings.json').write_text(json.dumps({
 'device':'CPU','images':20,'image_size':640,'conf':CONF,'iou':IOU,'batch':1,
 'warmup':3,'repeats':3,'measurement':'wall-clock model.predict; excludes model loading and output writing',
 'hardware':platform.platform()},indent=2))

# ===== NOTEBOOK CODE CELL 6 =====

from pathlib import Path
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
import ast
import json
import shutil
from IPython.display import display, Image

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT = Path("/content/gdrive/MyDrive/AIML_Shape_Project")

BEST_PT = PROJECT / "training_runs/yolov8n_cvat/weights/best.pt"
DATASET = PROJECT / "yolo_dataset"

FINAL_DIR = PROJECT / "final_submission_outputs"
ANNOTATED_DIR = FINAL_DIR / "annotated_images"
PREVIEW_DIR = FINAL_DIR / "contour_previews"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["circle", "square", "triangle", "rectangle"]

CONF = 0.25
IOU = 0.7
IMGSZ = 640

assert BEST_PT.exists(), "Trained best.pt is missing."

model = YOLO(str(BEST_PT))

# ============================================================
# 2. COLLECT ALL 200 IMAGES
# ============================================================

image_paths = []

for split in ["train", "val", "test"]:
    image_paths.extend(
        (DATASET / "images" / split).glob("*.png")
    )

image_paths = sorted(image_paths, key=lambda p: p.name)

assert len(image_paths) == 200, (
    f"Expected 200 images, found {len(image_paths)}"
)

assert len(set(p.name for p in image_paths)) == 200, (
    "Duplicate image names found."
)

print("Images found:", len(image_paths))

# ============================================================
# 3. DEFECT CLASSIFICATION
# ============================================================

def classify_defect(area):
    if area < 5000:
        return "less_defect"
    elif area <= 15000:
        return "moderate_defect"
    else:
        return "severe_defect"

# ============================================================
# 4. EXTRACT ACTUAL CONTOUR INSIDE YOLO BOUNDING BOX
# ============================================================

def extract_shape_contour(image, bbox):
    """
    Extracts the dominant shape contour within a predicted
    bounding box using OpenCV image segmentation.

    Returns:
        contour in full-image coordinates
        contour area in pixels squared
        centroid (cx, cy) in full-image coordinates

    Returns None if a usable contour cannot be found.
    """

    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(float, bbox)

    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(width, int(np.ceil(x2)))
    y2 = min(height, int(np.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    # Small padding improves segmentation near shape edges.
    pad = 5

    rx1 = max(0, x1 - pad)
    ry1 = max(0, y1 - pad)
    rx2 = min(width, x2 + pad)
    ry2 = min(height, y2 + pad)

    roi = image[ry1:ry2, rx1:rx2]

    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]

    # Estimate the local background from ROI border pixels.
    border_pixels = np.concatenate([
        roi[0, :, :],
        roi[-1, :, :],
        roi[:, 0, :],
        roi[:, -1, :]
    ], axis=0)

    background = np.median(
        border_pixels.astype(np.float32),
        axis=0
    )

    # Color difference from estimated background.
    difference = np.linalg.norm(
        roi.astype(np.float32) - background,
        axis=2
    )

    difference = np.clip(difference, 0, 255).astype(np.uint8)

    # Otsu threshold adapts to local contrast.
    _, mask_otsu = cv2.threshold(
        difference, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Minimum contrast threshold reduces background noise.
    _, mask_fixed = cv2.threshold(
        difference, 25, 255,
        cv2.THRESH_BINARY
    )

    mask = cv2.bitwise_and(mask_otsu, mask_fixed)

    # Remove tiny specks and close small gaps.
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Prefer a contour that overlaps the detected bbox
    # and is near the bbox center.
    bbox_cx = (x1 + x2) / 2
    bbox_cy = (y1 + y2) / 2

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 20:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        local_cx = moments["m10"] / moments["m00"]
        local_cy = moments["m01"] / moments["m00"]

        global_cx = local_cx + rx1
        global_cy = local_cy + ry1

        # Contour centroid should lie near detected object.
        distance = np.hypot(
            global_cx - bbox_cx,
            global_cy - bbox_cy
        )

        bbox_diag = max(
            np.hypot(x2 - x1, y2 - y1), 1
        )

        normalized_distance = distance / bbox_diag

        score = area / (1 + 3 * normalized_distance)

        candidates.append((score, contour))

    if not candidates:
        return None

    best_contour = max(
        candidates,
        key=lambda item: item[0]
    )[1]

    # Convert ROI contour coordinates to full-image coords.
    contour_global = best_contour.copy()

    contour_global[:, :, 0] += rx1
    contour_global[:, :, 1] += ry1

    area = float(cv2.contourArea(contour_global))

    moments = cv2.moments(contour_global)

    if moments["m00"] == 0:
        return None

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    return contour_global, area, (cx, cy)

# ============================================================
# 5. PROCESS ALL IMAGES
# ============================================================

final_rows = []
detection_rows = []
failed_contours = []

for image_index, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read: {image_path}")

    annotated = image.copy()

    result = model.predict(
        source=image,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False
    )[0]

    # Keep lists for each class.
    grouped = {
        name: {
            "areas": [],
            "centers": [],
            "defects": []
        }
        for name in CLASSES
    }

    for detection_index, box in enumerate(result.boxes):

        class_id = int(box.cls.item())
        class_name = CLASSES[class_id]

        confidence = float(box.conf.item())
        bbox = box.xyxy[0].cpu().numpy().tolist()

        contour_data = extract_shape_contour(image, bbox)

        if contour_data is None:
            failed_contours.append({
                "image_name": image_path.name,
                "class_name": class_name,
                "detection_index": detection_index
            })

            # Preserve the detection count.
            # Zero area/center mark unsuccessful measurement.
            area = 0.0
            center = (0.0, 0.0)
            defect = classify_defect(area)
            contour = None

        else:
            contour, area, center = contour_data
            defect = classify_defect(area)

        grouped[class_name]["areas"].append(round(area, 2))
        grouped[class_name]["centers"].append(
            [round(center[0], 2), round(center[1], 2)]
        )
        grouped[class_name]["defects"].append(defect)

        x1, y1, x2, y2 = map(int, bbox)

        # Bounding box
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Actual contour
        if contour is not None:
            cv2.drawContours(
                annotated,
                [contour],
                -1,
                (255, 0, 255),
                2
            )

            # Centroid
            cx, cy = map(int, center)

            cv2.circle(
                annotated,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

        # Annotation label
        label = (
            f"{class_name} {confidence:.2f} "
            f"A={area:.0f} {defect}"
        )

        cv2.putText(
            annotated,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        detection_rows.append({
            "image_name": image_path.name,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 6),
            "x1": round(bbox[0], 2),
            "y1": round(bbox[1], 2),
            "x2": round(bbox[2], 2),
            "y2": round(bbox[3], 2),
            "area_px2": round(area, 2),
            "center_x": round(center[0], 2),
            "center_y": round(center[1], 2),
            "defect": defect,
            "contour_success": contour_data is not None
        })

    # ========================================================
    # 6. BUILD EXACT 17-COLUMN FINAL CSV ROW
    # ========================================================

    row = {"image_name": image_path.name}

    for name in CLASSES:
        row[f"{name}_count"] = str([len(grouped[name]["areas"])])

    for name in CLASSES:
        areas = grouped[name]["areas"]
        row[f"{name}_area_px2"] = str(areas)

    for name in CLASSES:
        centers = grouped[name]["centers"]
        row[f"{name}_center"] = str(centers if centers else [])

    for name in CLASSES:
        defects = grouped[name]["defects"]
        row[f"{name}_defect"] = str(defects if defects else [])

    final_rows.append(row)

    cv2.imwrite(
        str(ANNOTATED_DIR / image_path.name),
        annotated
    )

    if image_index % 25 == 0:
        print(f"Processed {image_index}/200 images")

# ============================================================
# 7. SAVE CSV FILES
# ============================================================

final_df = pd.DataFrame(final_rows)

expected_columns = (
    ["image_name"]
    + [f"{name}_count" for name in CLASSES]
    + [f"{name}_area_px2" for name in CLASSES]
    + [f"{name}_center" for name in CLASSES]
    + [f"{name}_defect" for name in CLASSES]
)

final_df = final_df[expected_columns]

assert final_df.shape == (200, 17), (
    f"Unexpected final CSV shape: {final_df.shape}"
)

final_csv = FINAL_DIR / "final_results.csv"
final_df.to_csv(final_csv, index=False)

detections_df = pd.DataFrame(detection_rows)
detections_df.to_csv(
    FINAL_DIR / "detection_measurements.csv",
    index=False
)

failures_df = pd.DataFrame(
    failed_contours,
    columns=["image_name", "class_name", "detection_index"]
)

failures_df.to_csv(
    FINAL_DIR / "contour_failures.csv",
    index=False
)

# ============================================================
# 8. VERIFY CSV FORMAT AND MEASUREMENT RESULTS
# ============================================================

print("\n========== FINAL CSV CHECK ==========")
print("Shape:", final_df.shape)
print("Columns:", final_df.columns.tolist())
print("Annotated images:", len(list(ANNOTATED_DIR.glob("*.png"))))
print("Total detections:", len(detections_df))
print("Failed contour measurements:", len(failures_df))

# Check that every class has aligned lists.
for _, row in final_df.iterrows():
    for name in CLASSES:
        count = ast.literal_eval(row[f"{name}_count"])[0]
        areas = ast.literal_eval(row[f"{name}_area_px2"])
        centers = ast.literal_eval(row[f"{name}_center"])
        defects = ast.literal_eval(row[f"{name}_defect"])

        assert len(centers) == count
        assert len(defects) == count
        assert len(areas) == count

print("All list alignment checks passed.")

print("\n========== DEFECT DISTRIBUTION ==========")

if len(detections_df):
    display(
        pd.crosstab(
            detections_df["class_name"],
            detections_df["defect"]
        )
    )

print("\n========== AREA STATISTICS ==========")

if len(detections_df):
    display(
        detections_df.groupby("class_name")["area_px2"]
        .agg(["count", "min", "median", "mean", "max"])
        .round(2)
    )

# ============================================================
# 9. SHOW ANNOTATED EXAMPLES
# ============================================================

print("\n========== ANNOTATED PREVIEWS ==========")

# Select examples across the dataset.
preview_names = [
    "image_0001.png",
    "image_0050.png",
    "image_0100.png",
    "image_0150.png",
    "image_0200.png"
]

for name in preview_names:
    preview_path = ANNOTATED_DIR / name

    if preview_path.exists():
        preview = cv2.imread(str(preview_path))

        # Resize only for notebook display.
        h, w = preview.shape[:2]
        preview = cv2.resize(
            preview,
            (960, int(h * 960 / w))
        )

        preview_output = PREVIEW_DIR / name
        cv2.imwrite(str(preview_output), preview)

        print(name)
        display(Image(filename=str(preview_output)))

# ============================================================
# 10. FINAL OUTPUT SUMMARY
# ============================================================

print("\n========== COMPLETE ==========")
print("Final results:", final_csv)
print("Detection measurements:",
      FINAL_DIR / "detection_measurements.csv")
print("Annotated images:", ANNOTATED_DIR)
print("Contour failure log:",
      FINAL_DIR / "contour_failures.csv")

print("\nIMPORTANT:")
print("Inspect the preview contours before submission.")
print("Area and centroid accuracy against hidden ground")
print("truth cannot be calculated without reference values.")

# ===== NOTEBOOK CODE CELL 7 =====

from pathlib import Path
import pandas as pd
import numpy as np
import ast

PROJECT = Path("/content/gdrive/MyDrive/AIML_Shape_Project")
OUT = PROJECT / "final_submission_outputs"

final_df = pd.read_csv(OUT / "final_results.csv")
detections = pd.read_csv(OUT / "detection_measurements.csv")
failures = pd.read_csv(OUT / "contour_failures.csv")

CLASSES = ["circle", "square", "triangle", "rectangle"]

print("========== FILE VERIFICATION ==========")
print("Final CSV shape:", final_df.shape)
print("Total detections:", len(detections))
print("Contour failures:", len(failures))
print("Annotated images:",
      len(list((OUT / "annotated_images").glob("*.png"))))

assert final_df.shape == (200, 17)
assert final_df["image_name"].nunique() == 200

print("\n========== CONTOUR QUALITY ==========")

if len(detections):
    detections["bbox_area"] = (
        (detections["x2"] - detections["x1"]) *
        (detections["y2"] - detections["y1"])
    )

    detections["area_ratio"] = (
        detections["area_px2"] /
        detections["bbox_area"].replace(0, np.nan)
    )

    print("Zero-area detections:",
          int((detections["area_px2"] == 0).sum()))

    print("Very small area ratios (<0.20):",
          int((detections["area_ratio"] < 0.20).sum()))

    print("Area larger than bounding box:",
          int((detections["area_ratio"] > 1.05).sum()))

    print("\nArea statistics by class:")
    display(
        detections.groupby("class_name")["area_px2"]
        .agg(["count", "min", "median", "mean", "max"])
        .round(2)
    )

    print("\nArea-to-bounding-box ratio:")
    display(
        detections.groupby("class_name")["area_ratio"]
        .agg(["min", "median", "mean", "max"])
        .round(3)
    )

    print("\nDefect distribution:")
    display(
        pd.crosstab(
            detections["class_name"],
            detections["defect"]
        )
    )

    # Check defect labels against the assignment thresholds.
    expected_defects = np.select(
        [
            detections["area_px2"] < 5000,
            detections["area_px2"] <= 15000
        ],
        ["less_defect", "moderate_defect"],
        default="severe_defect"
    )

    mismatches = (
        detections["defect"].to_numpy() != expected_defects
    )

    print("\nIncorrect defect labels:",
          int(mismatches.sum()))

    # Check centroid coordinates lie within image boundaries.
    invalid_centers = (
        (detections["center_x"] < 0) |
        (detections["center_x"] >= 1920) |
        (detections["center_y"] < 0) |
        (detections["center_y"] >= 1080)
    )

    print("Centroids outside image:",
          int(invalid_centers.sum()))

    suspicious = detections[
        (detections["area_px2"] == 0) |
        (detections["area_ratio"] < 0.20) |
        (detections["area_ratio"] > 1.05) |
        invalid_centers
    ].copy()

    suspicious.to_csv(
        OUT / "suspicious_measurements.csv",
        index=False
    )

    print("\nSuspicious measurements:", len(suspicious))

    if len(suspicious):
        display(
            suspicious[
                ["image_name", "class_name", "area_px2",
                 "bbox_area", "area_ratio",
                 "center_x", "center_y"]
            ].head(20)
        )

print("\n========== FINAL CSV PREVIEW ==========")
display(final_df.head(3))

print("\nQuality-check results saved.")

# Extra PDF-format checks (counts are list strings, not integers)
for _,row in final_df.iterrows():
    for cls in CLASSES:
        count=ast.literal_eval(row[f'{cls}_count'])
        areas=ast.literal_eval(row[f'{cls}_area_px2'])
        centers=ast.literal_eval(row[f'{cls}_center'])
        defects=ast.literal_eval(row[f'{cls}_defect'])
        assert isinstance(count,list) and len(count)==1 and isinstance(count[0],int)
        assert len(areas)==len(centers)==len(defects)==count[0]
print('PDF list-format checks passed for all 200 images.')

# ===== NOTEBOOK CODE CELL 8 =====


from pathlib import Path
import pandas as pd
import ast
import shutil
import zipfile
import json
import nbformat
import textwrap

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT = Path("/content/gdrive/MyDrive/AIML_Shape_Project")
SUB = PROJECT / "AIML_Final_Submission"

SCRIPTS = SUB / "scripts"
MODELS = SUB / "models"
RESULTS = SUB / "results"
DOCS = SUB / "documentation"
EVAL = SUB / "evaluation"
TRAIN = SUB / "training"
ANNOTATIONS = SUB / "annotations"

for folder in [SCRIPTS, MODELS, RESULTS, DOCS,
               EVAL, TRAIN, ANNOTATIONS]:
    folder.mkdir(parents=True, exist_ok=True)

CLASSES = ["circle", "square", "triangle", "rectangle"]

# ============================================================
# 2. CORRECT FINAL CSV TO EXACT PDF FORMAT
# ============================================================

source_csv = PROJECT / "final_submission_outputs/final_results.csv"

assert source_csv.exists(), f"Missing {source_csv}"

df = pd.read_csv(source_csv)

expected_cols = (
    ["image_name"]
    + [f"{c}_count" for c in CLASSES]
    + [f"{c}_area_px2" for c in CLASSES]
    + [f"{c}_center" for c in CLASSES]
    + [f"{c}_defect" for c in CLASSES]
)

assert df.shape == (200, 17)
assert set(df.columns) == set(expected_cols)

def parse_list(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value

for idx in df.index:
    for cls in CLASSES:
        count_col = f"{cls}_count"
        area_col = f"{cls}_area_px2"
        center_col = f"{cls}_center"
        defect_col = f"{cls}_defect"

        count_value = df.at[idx, count_col]

        if isinstance(count_value, str) and count_value.startswith("["):
            count = int(ast.literal_eval(count_value)[0])
        else:
            count = int(count_value)

        areas = parse_list(df.at[idx, area_col])
        centers = parse_list(df.at[idx, center_col])
        defects = parse_list(df.at[idx, defect_col])

        if count == 0:
            areas = []
            centers = []
            defects = []
        else:
            assert len(areas) == count
            assert len(centers) == count
            assert len(defects) == count

        # PDF requires list[int] for count.
        df.at[idx, count_col] = str([count])
        df.at[idx, area_col] = str(areas)
        df.at[idx, center_col] = str([
            tuple(point) for point in centers
        ])
        df.at[idx, defect_col] = str(defects)

df = df[expected_cols]

corrected_csv = RESULTS / "final_results.csv"
df.to_csv(corrected_csv, index=False)

# Reopen and verify.
check = pd.read_csv(corrected_csv)

assert check.shape == (200, 17)
assert check["image_name"].nunique() == 200

for _, row in check.iterrows():
    for cls in CLASSES:
        count = ast.literal_eval(row[f"{cls}_count"])[0]
        areas = ast.literal_eval(row[f"{cls}_area_px2"])
        centers = ast.literal_eval(row[f"{cls}_center"])
        defects = ast.literal_eval(row[f"{cls}_defect"])

        assert len(areas) == count
        assert len(centers) == count
        assert len(defects) == count

print("Corrected CSV:", corrected_csv)
print("CSV shape:", check.shape)

# ============================================================
# 3. COPY MODELS AND RESULTS
# ============================================================

files_to_copy = {
    PROJECT / "training_runs/yolov8n_cvat/weights/best.pt":
        MODELS / "best.pt",
    PROJECT / "onnx_export/best.onnx":
        MODELS / "best.onnx",
    PROJECT / "final_submission_outputs/detection_measurements.csv":
        RESULTS / "detection_measurements.csv",
    PROJECT / "final_submission_outputs/contour_failures.csv":
        RESULTS / "contour_failures.csv"
}

for source, destination in files_to_copy.items():
    assert source.exists(), f"Missing: {source}"
    shutil.copy2(source, destination)

annotated_source = (
    PROJECT / "final_submission_outputs/annotated_images"
)

shutil.copytree(
    annotated_source,
    RESULTS / "annotated_images",
    dirs_exist_ok=True
)

assert len(list(
    (RESULTS / "annotated_images").glob("*.png")
)) == 200

# ============================================================
# 4. COPY ANNOTATIONS AND DATA SPLIT
# ============================================================

cvat_zip = PROJECT / "cvat_verified_annotations.zip"

if cvat_zip.exists():
    shutil.copy2(
        cvat_zip,
        ANNOTATIONS / cvat_zip.name
    )
else:
    print("WARNING: CVAT annotation ZIP not found.")

dataset = PROJECT / "yolo_dataset"

assert dataset.exists()

# Include label files and the exact train/val/test image names.
for split in ["train", "val", "test"]:
    src_labels = dataset / "labels" / split
    dst_labels = ANNOTATIONS / "labels" / split

    shutil.copytree(
        src_labels,
        dst_labels,
        dirs_exist_ok=True
    )

    image_names = sorted(
        p.name for p in (dataset / "images" / split).glob("*.png")
    )

    (ANNOTATIONS / f"{split}_images.txt").write_text(
        "\n".join(image_names) + "\n"
    )

if (dataset / "data.yaml").exists():
    shutil.copy2(
        dataset / "data.yaml",
        ANNOTATIONS / "data.yaml"
    )

# ============================================================
# 5. COPY TRAINING LOGS AND EVALUATION RESULTS
# ============================================================

train_src = PROJECT / "training_runs/yolov8n_cvat"

for item in train_src.iterdir():
    if item.name == "weights":
        continue

    dst = TRAIN / item.name

    if item.is_file():
        shutil.copy2(item, dst)
    elif item.is_dir():
        shutil.copytree(item, dst, dirs_exist_ok=True)

eval_src = PROJECT / "evaluation"

assert eval_src.exists()

shutil.copytree(
    eval_src,
    EVAL,
    dirs_exist_ok=True
)

export_settings = PROJECT / "onnx_export/export_settings.json"

if export_settings.exists():
    shutil.copy2(
        export_settings,
        DOCS / "onnx_export_settings.json"
    )

# ============================================================
# 6. EXPORT REPRODUCIBLE SCRIPTS FROM OPTIMIZED NOTEBOOK
# ============================================================

# These scripts are derived from the code in the optimized
# notebook. Upload that notebook to Colab or save it to Drive.

notebook_candidates = [
    PROJECT / "Shape_Detection_Optimized.ipynb",
    Path("/content/Shape_Detection_Optimized.ipynb")
]

notebook_path = next(
    (p for p in notebook_candidates if p.exists()),
    None
)

if notebook_path is None:
    print(
        "\nNOTE: Optimized notebook not found in Drive. "
        "Upload Shape_Detection_Optimized.ipynb to Colab "
        "or save it in the project folder, then rerun "
        "this cell to export scripts."
    )
else:
    nb = nbformat.read(str(notebook_path), as_version=4)

    code_cells = [
        cell.source
        for cell in nb.cells
        if cell.cell_type == "code"
    ]

    # Seven main sections of the optimized notebook.
    script_names = [
        "01_setup_and_dataset.py",
        "02_train_yolo.py",
        "03_evaluate_pytorch.py",
        "04_export_and_evaluate_onnx.py",
        "05_compare_models.py",
        "06_opencv_final_pipeline.py",
        "07_validate_results.py"
    ]

    assert len(code_cells) >= 7

    for filename, source in zip(script_names, code_cells[:7]):
        # Convert Colab pip magic to a Python comment.
        cleaned_lines = []

        for line in source.splitlines():
            if line.lstrip().startswith("%pip"):
                cleaned_lines.append(
                    "# Install dependencies separately: " +
                    line.lstrip()[1:]
                )
            else:
                cleaned_lines.append(line)

        (SCRIPTS / filename).write_text(
            "\n".join(cleaned_lines) + "\n",
            encoding="utf-8"
        )

    shutil.copy2(
        notebook_path,
        DOCS / "Shape_Detection_Optimized.ipynb"
    )

    print("Exported", len(script_names), "script sections.")
    print(
        "NOTE: These sections preserve notebook execution "
        "order and may depend on earlier variables. "
        "The notebook is the primary reproducible entry point."
    )

# ============================================================
# 7. CREATE README
# ============================================================

readme = """
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
"""

(DOCS / "README.md").write_text(
    textwrap.dedent(readme).strip() + "\n",
    encoding="utf-8"
)

# ============================================================
# 8. SAVE MANIFEST
# ============================================================

manifest = []

for path in sorted(SUB.rglob("*")):
    if path.is_file():
        manifest.append({
            "path": str(path.relative_to(SUB)),
            "size_bytes": path.stat().st_size
        })

pd.DataFrame(manifest).to_csv(
    DOCS / "file_manifest.csv",
    index=False
)

# ============================================================
# 9. CREATE ZIP
# ============================================================

zip_path = PROJECT / "AIML_Final_Submission.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED,
    compresslevel=6
) as z:
    for path in SUB.rglob("*"):
        if path.is_file():
            z.write(
                path,
                arcname=str(
                    Path(SUB.name) / path.relative_to(SUB)
                )
            )

with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None

# ============================================================
# 10. SUMMARY
# ============================================================

print("\n========== SUBMISSION PACKAGE ==========")
print("Folder:", SUB)
print("ZIP:", zip_path)
print("ZIP size (MB):",
      round(zip_path.stat().st_size / 1024**2, 2))
print("Final CSV:", corrected_csv)
print("CSV shape:", check.shape)
print("Annotated images:", 200)
print("Model files:", 2)
print("Script files:", len(list(SCRIPTS.glob("*.py"))))
print("README:", DOCS / "README.md")
print("\nPackage created successfully.")

# ===== NOTEBOOK CODE CELL 9 =====

