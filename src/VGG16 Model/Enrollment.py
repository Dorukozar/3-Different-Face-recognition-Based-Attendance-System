import os
import cv2
import string
from Segmentation import Segmentation
from FaceMatching import FaceMatching
import shutil


class EnrollSystem:
    """EnrollSystem class that encapsulates methods for enrolling users, capturing images, and handling enrollment data."""

    @staticmethod
    def increase_ID(folder_path):
        """
        Calculates the next available ID by inspecting the given folder path.

        Parameters:
        :param folder_path: str
            Path to the folder where images or files are saved with ID-based naming.

        Returns:
        :return: str
            Next available ID for saving a new image or file.
        """
        elements_in_folder = os.listdir(folder_path)
        if len(elements_in_folder) == 0:
            return 1

        elements_in_folder.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        last_element = elements_in_folder[-1]
        id = last_element.split('_')[-1].split('.')[0]
        return str(int(id) + 1)

    def capture_and_save_face(self):
        """
        Initializes the webcam, captures a face image, and then saves it.

        Returns:
        :return: numpy.ndarray
            The captured image frame.
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

    def save_face_image(self, full_name, frame):
        """
        Saves the captured face image in the 'data_base' directory and invokes facial landmarks and
        features extraction methods.

        Parameters:
        :param full_name: str
            Full name of the person whose face has been captured.
        :param frame: numpy.ndarray
            Captured face image.
        """
        # Base directory
        # base_dir = "./../data_base"
        base_dir = os.path.join("../..", "data_base")

        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Create directory for the user under the base directory if it doesn't exist

        directory_name = os.path.join(base_dir, full_name)
        os.makedirs(directory_name, exist_ok=True)

        # Save the captured image
        user_name_list = full_name.split(" ")
        user_name = "_".join(user_name_list)
        file_path = os.path.join(directory_name,
                                 f"{user_name}_{EnrollSystem.increase_ID(os.path.join(base_dir, full_name))}.jpg")
        cv2.imwrite(file_path, frame)
        id = file_path.split("_")[-1].split(".")[0]
        Segmentation.put_landmarks(file_path, full_name, id)
        face_matcher_object = FaceMatching()
        preprocessed_img = face_matcher_object.preprocess_image(frame)
        features = face_matcher_object.extract_features(preprocessed_img)
        FaceMatching.save_features(full_name, features)

    def delete_data_base_images(self, full_name):
        """
        Prompts the user for a decision to delete their database images. If approved, deletes the images.

        Parameters:
        :param full_name: str
            Full name of the person whose database images may be deleted.

        Returns:
        :return: None
        """
        print(f"Image saved for {full_name}.")
        print("Enrollment successful!\nAll of your embeddings are saved.")
        while True:
            # print("Do you wish to delete your database images.")
            answer = input("Do you wish to delete your database images? (y/n):\n")
            if answer in ['yes', 'y']:
                shutil.rmtree(os.path.join("../../data_base", full_name))
                shutil.rmtree(os.path.join("../../segmented_pictures", full_name))
                print("Your database images will be deleted after the program terminates")
                return
            elif answer in ['no', 'n']:
                print("Your database images are not deleted.")
                return
            else:
                print("You entered wrong input.")

    def enrollment(self):
        """
        Enrolls a new user by:
        1. Asking for the user's full name.
        2. Capturing their face.
        3. Saving the captured face under the 'data_base' directory.
        4. Optionally deleting the saved images upon user's decision.

        Returns:
        :return: None
        """
        # Ask for the user's full name
        name = input("Enter your full name (first and last): ")
        user_info_list = name.split(" ")
        first_name = string.capwords(user_info_list[0])
        last_name = string.capwords(user_info_list[1])
        full_name = " ".join([first_name, last_name])
        # Capture the face
        frame = self.capture_and_save_face()

        # Save the face image under the 'data_base' directory
        self.save_face_image(full_name, frame)

        return self.delete_data_base_images(full_name)


