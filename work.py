import cv2
import numpy as np

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def save_img(img_data, file_name):
    cv2.imwrite(file_name, img_data)

def crop_img(image, x, y, height, width):
    cropped = image[y:y+height, x:x+width]
    return cropped


def detect_faces():
    # Create a VideoCapture object for the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    
    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = 1

    while True:
        ret, frame = cap.read()
        if ret:
            # Convert the frame to grayscale for faster face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # Crop the face
                cropped = crop_img(frame[y:y+h, x:x+w], x, y, h, w)

                # Save the cropped face
                save_img(cropped, f"img_{count}.jpg")
                count += 1
            
            # Display the frame
            cv2.imshow('Face', gray)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()



detect_faces()