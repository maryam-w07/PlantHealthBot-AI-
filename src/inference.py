import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import send_email_alert

# Load the pretrained model
model = load_model('plant_health_model.h5')

def preprocess_frame(frame):
    # Resize frame to 224x224 and normalize pixel values
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def run_inference():
    cap = cv2.VideoCapture(0)  # Change to your camera source, e.g., index or video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame)
        preds = model.predict(processed)
        label = np.argmax(preds, axis=1)[0]
        confidence = preds[0][label]

        # For example, assume label 1 corresponds to "unhealthy" and threshold confidence > 0.5
        if label == 1 and confidence > 0.5:
            # Send an email alert using the utility function
            send_email_alert("Alert: Unhealthy plant detected",
                             "An unhealthy plant has been detected in the monitored area. Please investigate.")

        # Display the frame (optional)
        cv2.imshow('Live Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()
