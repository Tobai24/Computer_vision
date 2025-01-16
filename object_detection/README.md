## **Object Detection with YOLOv11**

This project demonstrates the use of a pretrained YOLOv11 model for object detection. The YOLO (You Only Look Once) algorithm is a state-of-the-art object detection framework that detects objects in images or videos in real-time with high accuracy and efficiency.

### **Project Overview**

This repository contains code and resources to perform object detection using a pretrained YOLOv11 model. The primary goal is to showcase the capabilities of the pretrained model on standard datasets and prepare for training on custom datasets for more specialized tasks.

The video used in this project features cars on a highway, which aligns well with the classes the pretrained YOLOv11 model is trained to detect. As expected, the model performed well on this dataset, accurately identifying vehicles and other common objects.

## **Key Features**

- **Pretrained YOLOv11 Model**:

  - Utilizes the YOLOv11 model for object detection.
  - Detects multiple classes, including cars, train and pedestrians
  - Capable of processing images and videos in real-time.

- **Customizable Predictions**:

  - Modify bounding box colors and label styles to suit your needs.
  - Save detection results as annotated images or videos for further analysis.

- **Future Plans**:
  - Train the YOLOv11 model on a custom dataset for specific object detection tasks (medical imaging).

---

## **Getting Started**

### **Requirements**

To run this project, you need the following:

1. **Python** (>= 3.8)
2. **Libraries**:
   - `ultralytics`

Install the required libraries using the following command:

```bash
pip install ultralytics
```

### **Usage**

run the cell in the [notebook.ipynb](notebook.ipynb)

## **Demo**

### **Sample Output on Highway Video**

The pretrained YOLOv11 model was used to detect objects in a video featuring cars on a highway. Below is a preview of the output:

- **Objects Detected**: Cars, train, pedestrians, and other road features.
- **Performance**: The model accurately identified objects and provided bounding boxes with high confidence.

[object_detection/vehicle.mp4](object_detection/vehicle.mp4)

---

## **Training on a Custom Dataset**

In the future, this project will be extended to support training the YOLOv11 model on custom datasets. This will allow the model to specialize in detecting objects outside its default classes, such as: Medical images (e.g., tumors, organs, instruments).
