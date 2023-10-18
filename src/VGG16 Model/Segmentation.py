import os.path
import cv2
import face_recognition
import matplotlib.pyplot as plt


class Segmentation:
    """Segmentation class that provides utilities for image processing and facial landmarks' identification."""

    @staticmethod
    def put_landmarks(image_path, user_name, id):
        """
        Detects, marks, and saves the facial landmarks on the face in the given image.

        Parameters:
        :param image_path: str
            Path to the image where landmarks need to be put.
        :param user_name: str
            Name of the user associated with the image.
        :param id: int
            ID associated with the image or user.
        """
        db_image, directory_name, faces, user_name = Segmentation.make_image_grayscale(id, image_path, user_name)

        cropped_face = Segmentation.crop_image(db_image, directory_name, faces, id, user_name)

        # Detect facial landmarks
        # face_landmarks_list = face_recognition.face_landmarks(image)
        face_landmarks_list = face_recognition.face_landmarks(cropped_face)

        # Draw landmarks on the image
        for face_landmarks in face_landmarks_list:
            for landmark_type, points in face_landmarks.items():
                for point in points:
                    cv2.circle(cropped_face, point, 2, (0, 255, 0), -1)

        plt.imshow(cropped_face)
        plt.axis('off')
        plt.show()
        landmarks_directory = os.path.join(directory_name, "Landmarks Images")
        os.makedirs(landmarks_directory, exist_ok=True)
        landmarks_img_file_path = os.path.join(landmarks_directory, f"{user_name}_landmarks_image_{id}.jpg")
        cv2.imwrite(landmarks_img_file_path, cropped_face)

    @staticmethod
    def crop_image(db_image, directory_name, faces, id, user_name):
        """
        Crops the face from the provided image and saves it.

        Parameters:
        :param db_image: numpy.ndarray
            The input image array.
        :param directory_name: str
            Directory where the cropped image will be saved.
        :param faces: list of tuples
            List containing the coordinates of detected faces in the format (x, y, width, height).
        :param id: int
            ID associated with the image or user.
        :param user_name: str
            Name of the user associated with the image.

        Returns:
        :return: numpy.ndarray
            Cropped face from the image.
        """
        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped_face = db_image[y:y + h, x:x + w]
        else:
            cropped_face = db_image
        # Display the cropped face
        plt.imshow(cropped_face)
        plt.axis('off')
        plt.show()
        cropped_img_directory = os.path.join(directory_name, "Cropped Images")
        os.makedirs(cropped_img_directory, exist_ok=True)
        cropped_img_file_path = os.path.join(cropped_img_directory, f"{user_name}_cropped_image_{id}.jpg")
        cv2.imwrite(cropped_img_file_path, cropped_face)
        return cropped_face

    @staticmethod
    def make_image_grayscale(id, image_path, user_name):
        """
        Converts the input image to grayscale, detects faces, and saves the grayscale image.

        Parameters:
        :param id: int
            ID associated with the image or user.
        :param image_path: str
            Path to the input image.
        :param user_name: str
            Name of the user associated with the image.

        Returns:
        :return: tuple
            - db_image: numpy.ndarray - The original image array.
            - directory_name: str - Directory where the grayscale image is saved.
            - faces: list of tuples - List containing the coordinates of detected faces.
            - user_name: str - Processed user name (spaces replaced with underscores).
        """
        # Load an image using OpenCV
        # data_base_image_path = "./../Templates/Doruk Ozar/Doruk Ozar @16.jpg"
        db_image_path = image_path
        db_image = cv2.imread(db_image_path)
        # Convert the image to grayscale
        gray = cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY)
        # Load the cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(db_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.imshow(db_image)
        plt.axis('off')
        plt.show()
        # save grayscale image under ./../segmented_pictures folder
        segmented_img_dir = "../../segmented_pictures"
        os.makedirs(segmented_img_dir, exist_ok=True)
        directory_name = os.path.join(segmented_img_dir, user_name)
        os.makedirs(directory_name, exist_ok=True)
        gray_scale_directory = os.path.join(directory_name, "Gray Scale Images")
        os.makedirs(gray_scale_directory, exist_ok=True)
        user_name_list = user_name.split(" ")
        user_name = "_".join(user_name_list)
        gray_img_file_path = os.path.join(gray_scale_directory, f"{user_name}_gray_image_{id}.jpg")
        cv2.imwrite(gray_img_file_path, gray)
        return db_image, directory_name, faces, user_name


