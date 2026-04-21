# Explainable AI for Brain Tumor MRI Classification and Localization

This repository contains the DAT255 Deep Learning Engineering project by **Erikka Almestad Steen** and **Philipp Stahlberg**.

The project investigates brain tumor analysis in MRI images with a focus on both:
- **predictive performance** (classification and detection), and
- **model reliability** through **explainability** (Grad-CAM) and cross-dataset behavior.

## Project Goal

The main objective is to study whether deep learning models are learning medically meaningful features, not only achieving high accuracy.

The workflow covers:
1. 4-class tumor classification on Kaggle MRI data,
2. binary tumor vs. no-tumor classification,
3. transfer learning with a pretrained ResNet model,
4. localization analysis with Grad-CAM, and
5. object detection with YOLO on a second dataset with bounding boxes.

## Repository Contents

- `DAT255_project.ipynb` – main notebook containing all experiments, preprocessing, training, evaluation, Grad-CAM, and YOLO detection.
- `DAT255_report_template-7.pdf` – full project report.
- `README.md` – this file.

## Datasets

### Dataset 1 (Primary): Kaggle Brain Tumor MRI Dataset
- Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- ~7000 MRI images
- Used for:
  - 4-class classification (`glioma`, `meningioma`, `pituitary`, `notumor`)
  - binary classification (`tumor` vs `notumor`)

### Dataset 2 (Secondary): Ultralytics Brain Tumor Detection Dataset
- Source: https://docs.ultralytics.com/datasets/detect/brain-tumor/
- ~1000 images with bounding-box labels
- Used for:
  - cross-dataset generalization checks
  - YOLO-based tumor localization

## Methods

### Classification models
- Initial CNN baseline (4-class)
- Final deeper CNN baseline (4-class)
- Augmented CNN variant
- Binary CNN (tumor/no-tumor)
- Pretrained ResNet50-based binary classifier

### Explainability
- Grad-CAM was used to inspect model attention regions and identify potential shortcut learning.

### Object detection
- YOLO26n (`yolo26n.pt`) trained on Dataset 2 for tumor localization.

## Main Results

### Classification performance (Kaggle dataset)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Initial CNN | 0.86 | 0.85 | 0.85 | 0.85 |
| Final CNN | 0.89 | 0.89 | 0.88 | 0.88 |
| Augmented CNN | 0.74 | 0.76 | 0.75 | 0.74 |
| Binary CNN | 0.91 | 0.85 | 0.93 | 0.87 |
| ResNet CNN | **0.98** | **0.99** | **0.99** | **0.99** |

### YOLO performance (Dataset 2)

| Model | Box Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| YOLO26n | 0.58 | 0.604 | 0.585 | 0.413 |

## Key Takeaways

- Transfer learning (ResNet) gave the best classification performance.
- High accuracy alone was not enough to establish reliability.
- Grad-CAM suggested that models may rely on non-clinical artifacts (possible shortcut learning), especially for `notumor` cases.
- Performance dropped when transferring a CNN trained on Dataset 1 to Dataset 2, highlighting sensitivity to dataset shift.
- Detection with YOLO provided complementary localization insight, though with moderate performance.

## Running the Notebook

The notebook was developed primarily in **Google Colab**.

### 1) Open notebook
- Open `DAT255_project.ipynb` locally or in Colab.

### 2) Install dependencies used in the notebook
The notebook includes installation commands such as:
- `pip install kagglehub`
- `pip install keras_cv`
- `uv pip install ultralytics` (can also be replaced by `pip install ultralytics`)

You will also need the usual ML stack used in the notebook:
- TensorFlow / Keras
- NumPy, OpenCV, Matplotlib, Seaborn, scikit-learn

### 3) Configure Kaggle access (for Dataset 1)
In Colab, add your Kaggle API token as a secret (e.g. `KAGGLE_API_TOKEN`) and run the setup cell.

### 4) Run cells in order
The notebook is written as an end-to-end experimental flow. Run from top to bottom to reproduce preprocessing, training, evaluation, and visualization.

## Limitations and Intended Use

This project is **research/educational** work and is **not intended for autonomous clinical deployment**.

Main limitations discussed in the report:
- hardware and runtime constraints,
- limited dataset diversity/size (especially Dataset 2),
- explainability uncertainty (Grad-CAM is indicative, not definitive),
- possible shortcut learning.

## Citation

If you use this repository, please cite the project report in `DAT255_report_template-7.pdf`.
