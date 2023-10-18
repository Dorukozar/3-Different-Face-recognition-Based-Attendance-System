import csv  # This module is responsible to enroll new users in the system
from TemplateManager import TemplateManager
from CameraCapture import CameraCapture
from User import User


class EnrollSystem:
    # def __init__(self, num_of_images_will_be_taken=10):
    #     self.num_of_images_will_be_taken = num_of_images_will_be_taken

    @staticmethod
    def enroll_user(num_of_images_will_be_taken=10):
        camera_capture_obj = CameraCapture()
        cap = camera_capture_obj.capture()
        TemplateManager.load_template_images()

        count = 0

        last_name, firs_name = User.ask_user_for_full_name()
        full_name = firs_name + ' ' + last_name
        randomNum = User.generate_random_numbers()
        # date_time = User.generate_date_and_time()
        CameraCapture.save_user_image(cap, count, full_name, last_name, num_of_images_will_be_taken, firs_name,
                                      randomNum)
        # CameraCapture.save_user_image(cap, count, full_name, last_name, self.num_of_images_will_be_taken, name,
        #                               date_time)

# obj = EnrollSystem()
# obj.enroll_user()

# o = User()
# o.generate_date_and_time()
