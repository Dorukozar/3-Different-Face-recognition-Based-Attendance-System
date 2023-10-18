import string
import random
from datetime import datetime


class User:
    @staticmethod
    def ask_user_for_full_name():
        firs_name = input("Enter Your First Name: ")
        last_name = input("Enter Your Last Name: ")
        firs_name = string.capwords(firs_name)
        last_name = string.capwords(last_name)
        return last_name, firs_name

    @staticmethod
    def generate_random_numbers():
        randomNum = random.randint(1, 500)
        randomNum = randomNum * 5
        randomNum = randomNum / 2
        return randomNum

    @staticmethod
    def generate_date_and_time():
        dt = datetime.now().isoformat().replace(":", "_").replace(".", "_").replace("-", "_").replace("T", "_T_")
        # print(dt)
        return dt
