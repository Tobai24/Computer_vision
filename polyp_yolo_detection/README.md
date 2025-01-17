# Polyp Detection with YOLO 🩺

This repository contains the code and data for training an object detection model to detect polyps in gastrointestinal images. The model is fine-tuned on the YOLOv11 architecture, leveraging its pre-trained weights to achieve high performance in detecting polyps.

## 📊 Dataset

The dataset used in this project is based on the **Kvasir-SEG** dataset, originally intended for image segmentation tasks. The data was preprocessed to adapt it for object detection. The preprocessing steps and rationale are documented in the `training.ipynb` notebook.

- **Source:** [Kvasir-SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- **Preprocessing:** Converted segmentation labels into bounding boxes suitable for object detection tasks.

---

## 📂 Repository Structure

This repository includes the following:

```
yolo_data
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```

- **`yolo_data`**: Contains training and validation images along with their corresponding YOLO-format labels.
- **`data.yaml`**: Configuration file specifying paths to the dataset and classes for training the YOLO model. Creation steps are detailed in the `training.ipynb` notebook.
- **`test_images`**: Includes sample images to test the model's predictions.
- **`runs`**: Stores the model's predictions and outputs during training and testing.

---

## 🛠️ Notebooks

This repository contains two key Jupyter notebooks:

1. **`training.ipynb`**
   - Demonstrates the preprocessing steps for converting the dataset for object detection.
   - Includes the model training workflow with YOLOv11.
     -It also shows metrics for the trained model's performance
2. **`predict.ipynb`**
   - Shows examples of the model's predictions on sample images.
   - Useful for evaluating the trained model's performance.

## 🧠 Model Training

The model was fine-tuned on a pre-trained YOLOv11 model to detect polyps, despite it not being initially trained for this task. The training process, including hyperparameters, data preparation, and evaluation metrics, is outlined in `training.ipynb`.

---

## 🔍 Usage

1. **Run the Training Notebook**

   - Open `training.ipynb` to see the preprocessing and model training steps.

2. **Test the Model**
   - Use the `predict.ipynb` notebook to test the model on new images in the `test_images` folder.

---

## 🎯 Key Features

- Adapted segmentation dataset for object detection.
- Fine-tuned YOLOv11 model on custom polyp detection data.
- Structured directory and reproducible workflows for training and evaluation.

---

🌟 Feel free to explore!
