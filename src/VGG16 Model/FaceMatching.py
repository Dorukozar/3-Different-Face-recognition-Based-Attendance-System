import csv
import os
from datetime import datetime
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
import matplotlib.pyplot as plt


class FaceMatching:
    """FaceMatching class that encapsulates methods for face matching and attendance marking."""
    # class attributes
    feature_path = os.path.join('../..', 'extracted_features')

    # object attributes
    def __init__(self):
        """
        Initializes an instance of the FaceMatching class by setting up the model for feature extraction.
        """
        # Load the VGG16 model pre-trained on ImageNet data
        base_model = VGG16(weights='imagenet')
        # We'll extract features at the final fully connected layer, before the final classification layer.
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    def load_all_features(self):
        """
        Loads all the features saved in the feature_path directory.

        Returns:
        :return: dict
            A dictionary containing extracted features with filenames as keys and feature arrays as values.
        """
        # base_path = './../extracted_features'
        all_features = {}

        for user_name in os.listdir(FaceMatching.feature_path):
            user_directory = os.path.join(FaceMatching.feature_path, user_name)
            for feature_file in os.listdir(user_directory):
                feature_path = os.path.join(user_directory, feature_file)
                features = np.loadtxt(feature_path)
                all_features[feature_file] = features

        return all_features

    def preprocess_image(self, img):
        """
        Preprocesses an input image to make it suitable for feeding into the model.
        The steps include resizing the image to a fixed size, converting it to an array,
        and pre-processing it as required by certain models (like VGG, ResNet, etc.).

        Parameters:
        :param img: numpy.ndarray
            The input image.

        Returns:
        :return: numpy.ndarray
            The preprocessed image ready to be fed into the model.

        Examples:
        img = cv2.imread('sample.jpg')
        processed_img = preprocess_image(img)
        """
        img = cv2.resize(img, (224, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)

    def extract_features(self, img):
        """
        Extracts features from an input image using a pre-trained model. This method
        assumes that the image has already been preprocessed accordingly for the model.

        Parameters:
        :param img: numpy.ndarray
            The preprocessed input image.

        Returns:
        :return: numpy.ndarray
            The extracted features from the image.
        """
        # predict features
        features = self.model.predict(img)
        return features

    def detect_face(self, img):
        """
        Detects the face in the given image using the Haar cascades.

        Parameters:
        :param img: numpy.ndarray
            The input image in which the face is to be detected.

        Returns:
        :return: tuple or None
            Coordinates of the detected face in the format (x, y, width, height) or None if no face is detected.
        """
        # Load the Haar cascade xml file for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Return the first detected face coordinates (or None if no face is detected)
        return faces[0] if len(faces) > 0 else None

    def mark_attendance(self, user_name):
        """
        Marks the attendance for the given user by updating a CSV file.

        Parameters:
        :param user_name: str
            The name of the user whose attendance needs to be marked.
        """
        file_path = os.path.join('../..', 'attendance.csv')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Check if file exists
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['USERNAME', 'DATE TIME'])
                writer.writerow([user_name, current_time])
        else:
            with open(file_path, 'r') as file:
                rows = list(csv.reader(file))
                user_rows = [row for row in rows if row[0] == user_name]

            if user_rows:
                user_rows[0].append(current_time)
            else:
                rows.append([user_name, current_time])

            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)

    def record_attendance(self):
        """
        Records the attendance of a user by:
        1. Capturing their face.
        2. Preprocessing and extracting features from the image.
        3. Comparing the extracted features with saved features.
        4. If a match is found and above a similarity threshold, mark the attendance.
        """
        # Step 1: Capture the face
        face_img = self.capture_and_save_face()
        if face_img is None:
            print("Failed to capture image.")
            return

        # Step 2: Preprocess and extract features
        preprocessed_img = self.preprocess_image(face_img)

        face_features = self.extract_features(preprocessed_img)

        # Step 3: Compare with saved features
        saved_features = self.load_all_features()
        max_similarity = 0
        best_match = None
        for filename, features in saved_features.items():
            similarity = self.calculate_similarity(face_features, features)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = filename

        # Step 4: If similarity is above a threshold, mark attendance
        THRESHOLD = 0.8  # Set a suitable threshold value
        percentage = max_similarity * 100
        print(f"Best match: {best_match}, Max similarity: {round(percentage, 2)}%")
        if max_similarity > THRESHOLD:

            user_name = best_match.split("_")[0]

            # Detect face coordinates
            face_coordinates = self.detect_face(face_img)
            if face_coordinates is None:
                print("No face detected.")
                return

            # Draw a rectangle around the detected face
            (x, y, w, h) = face_coordinates
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color rectangle
            plt.figure(figsize=(5, 5))
            plt.imshow(face_img)
            plt.title(f"Detected Face: {user_name}")
            plt.annotate(f"Similarity: {round(percentage, 2)}%", xy=(10, face_img.shape[0] - 10),
                         color="white", fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            self.mark_attendance(user_name)
            print(f"Attendance recorded for {user_name}")
        else:
            print("Face not recognized.")

    @staticmethod
    def calculate_similarity(features1, features2):
        """
        Calculates the similarity between two feature vectors using cosine similarity.

        Parameters:
        :param features1: numpy.ndarray
            First feature vector.
        :param features2: numpy.ndarray
            Second feature vector.

        Returns:
        :return: float
            Cosine similarity value between the two feature vectors.
        """
        # Use cosine similarity to calculate similarity between feature vectors
        similarity = np.dot(features1, features2.T) / (np.linalg.norm(features1) * np.linalg.norm(features2))

        return similarity[0]

    @staticmethod
    def save_features(user_name, features):
        """
        Saves the extracted features for a given user to the disk.

        Parameters:
        :param user_name: str
            The name of the user.
        :param features: numpy.ndarray
            The extracted features of the user.
        """
        os.makedirs(FaceMatching.feature_path, exist_ok=True)
        user_directory = os.path.join(FaceMatching.feature_path, user_name)
        # print(f"user_directory: {user_directory}")
        os.makedirs(user_directory, exist_ok=True)
        user_directory_list = os.listdir(user_directory)
        # print(f"user_directory_list: {user_directory_list}")
        if len(user_directory_list) != 0:
            user_directory_list.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            last_element = user_directory_list[-1]
            ID = last_element.split('_')[-1].split('.')[0]
        else:
            ID = 0

        saving_id = str(int(ID) + 1)
        feature_file_path = os.path.join(user_directory, f"{user_name}_features_{saving_id}.txt")
        np.savetxt(feature_file_path, features)


    def capture_and_save_face(self):
        """
        Captures the face of the user using the webcam and saves it.

        Returns:
        :return: numpy.ndarray or None
            The captured image frame or None if the capture fails.
        """
        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Could not open the camera!")
            return

        print("Press 'c' to capture your face")
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display the resulting frame

            cv2.imshow('Press "c" to capture your face', frame)

            # Wait for 'c' key to capture the image
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        # Release the webcam and close the windows
        cap.release()
        cv2.destroyAllWindows()

        return frame

    def record_real_time_attendance(self):
        THRESHOLD = 0.5  # Add the threshold value

        # Load saved features
        saved_features = self.load_all_features()

        # Initialize Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Open a video feed
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]

                # Preprocess the face
                preprocessed_face = self.preprocess_image(face_img)

                # Extract features for the detected face
                face_features = self.extract_features(preprocessed_face)

                # Compare with saved features to recognize
                max_similarity = 0
                best_match = None
                for filename, features in saved_features.items():
                    similarity = self.calculate_similarity(face_features, features)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = filename

                # Check against threshold
                if max_similarity < THRESHOLD:
                    user_name = "Unknown"
                    match_score = "N/A"
                else:
                    # Extract user name from the filename
                    user_name = best_match.split("_")[0]
                    match_score = f"{round(max_similarity * 100, 2)}%"

                # Draw a rectangle around the detected face and display the user name and match score
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{user_name} ({match_score})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the frame with annotations
            cv2.imshow("Real-Time Face Recognition", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video feed and close windows
        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    # FaceMatching().record_attendance()
    FaceMatching().record_real_time_attendance()


