# **Face Recognition App 👤**

This folder contains the files needed to run a Flask-based API for face recognition. The application leverages **MTCNN** (Multi-task Cascaded Convolutional Networks) for face detection and **InceptionResnetV1** for face recognition. It is designed to detect and label faces in video streams.

### **How It Works**

Upload a video featuring any of the co-hosts of the "What Now?" podcast, and the app will detect their faces in the video.

You can check the live deployed version here: https://computer-vision-rxye.onrender.com

---

### **Getting Started**

Follow these steps to set up and run the app:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Tobai24/Computer_vision.git
   cd face_recognition
   ```

2. **Install Dependencies**  
   Make sure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**  
   Use the following command to start the app with Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```

4. **Access the App**  
   Once the app is running, open your browser and go to:  
   [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

---

### **Exploring the Application**

To understand how the application was built, explore the `notebook.ipynb` file. It contains:

- **Data Exploration**: Insights into the dataset used for training.
- **Model Building**: Steps involved in developing the face recognition model.

---

### **Dependencies**

All necessary libraries and dependencies are listed in the `requirements.txt` file. Install them to ensure the app runs smoothly.

---

Feel free to reach out if you encounter any issues or have suggestions for improvement! 😊
