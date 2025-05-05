import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Emotion Detection")
        
        # Dataset information
        self.datasets = ['FER2013', 'AffectNet', 'Micro_Expressions', 'AFM']
        
        self.paths = ['./FER_simple_CNN.h5', 
                     './FER_simple_CNN_AffectNet_Aligned.h5',
                     './FER_simple_CNN_Micro_Expressions.h5',
                     './FER_simple_CNN_AFM_2_without_augmentation.h5']
        
        self.all_emotions = [
            ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'],
            ['Angry', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutre', 'Contempt'],
            ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'],
            ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        ]
        
        # Datasets that use RGB color mode
        self.rgb_datasets = ['AffectNet', 'Micro_Expressions']
        
        # Initialize variables
        self.model = None
        self.emotions = []
        self.cap = None
        self.detecting = False
        self.current_dataset = tk.StringVar()
        self.color_mode = None
        self.input_channels = None  # Track expected input channels
        
        # Create UI
        self.create_widgets()
        
        # Load Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="10")
        dataset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dataset_frame, text="Select Dataset:").pack(side=tk.LEFT, padx=5)
        
        self.dataset_menu = ttk.Combobox(
            dataset_frame, textvariable=self.current_dataset, 
            values=self.datasets, state="readonly")
        self.dataset_menu.pack(side=tk.LEFT, padx=5)
        self.dataset_menu.current(0)
        
        # Color mode indicator
        self.color_label = ttk.Label(dataset_frame, text="Color Mode: Grayscale")
        self.color_label.pack(side=tk.RIGHT, padx=10)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(
            button_frame, text="Start Detection", command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            button_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Video display
        self.video_frame = ttk.Label(main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Update color mode when dataset changes
        self.current_dataset.trace_add('write', self.update_color_mode)
        self.update_color_mode()
    
    def update_color_mode(self, *args):
        """Update the color mode indicator based on selected dataset"""
        selected_dataset = self.current_dataset.get()
        if selected_dataset in self.rgb_datasets:
            self.color_mode = 'RGB'
            self.input_channels = 3
            self.color_label.config(text="Color Mode: RGB")
        else:
            self.color_mode = 'Grayscale'
            self.input_channels = 1
            self.color_label.config(text="Color Mode: Grayscale")
    
    def start_detection(self):
        # Get selected dataset
        dataset_idx = self.datasets.index(self.current_dataset.get())
        model_path = self.paths[dataset_idx]
        self.emotions = self.all_emotions[dataset_idx]
        
        # Load model
        try:
            self.model = load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.detecting = True
        
        # Start detection loop
        self.detect_emotions()
    
    def stop_detection(self):
        self.detecting = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def preprocess_face(self, face, target_size=(48, 48)):
        face_resized = cv2.resize(face, target_size)
        
        # Convert to appropriate color mode based on dataset
        if self.color_mode == 'RGB':
            face_converted = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        else:
            face_converted = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            # For grayscale models, we need to keep it as single channel
            face_normalized = face_converted / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))
            return face_input
        
        face_normalized = face_converted / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        return face_input
    
    def detect_emotions(self):
        if not self.detecting:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to appropriate color space for detection
            if self.color_mode == 'RGB':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_input = self.preprocess_face(face)
                
                try:
                    predictions = self.model.predict(face_input)
                    predicted_class = np.argmax(predictions)
                    predictions_percentage = [predict*100 for predict in predictions[0]]
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotions with percentages
                    y_offset = y
                    for i, emotion in enumerate(self.emotions):
                        # Set text color
                        text_color = (0, 255, 0)  # Green for all emotions
                        
                        # Customize predicted class
                        if i == predicted_class:
                            text_color = (255, 0, 0)  # Red for predicted
                            font_scale = 0.7
                            thickness = 2
                        else:
                            font_scale = 0.5
                            thickness = 1
                        
                        text = f'{emotion}: {predictions_percentage[i]:.2f}%'
                        (text_width, text_height), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        
                        # Draw background box
                        box_x1 = x - text_width - 10
                        box_x2 = x - 5
                        box_y1 = y_offset - text_height - 5
                        box_y2 = y_offset + 5
                        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                        
                        # Display text
                        cv2.putText(display_frame, text, (x - text_width - 8, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                        
                        y_offset += text_height + 10
                except Exception as e:
                    print(f"Prediction error: {e}")
                    continue
            
            # Convert frame to PhotoImage
            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update video frame
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
        
        # Schedule next frame
        self.root.after(10, self.detect_emotions)
    
    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()