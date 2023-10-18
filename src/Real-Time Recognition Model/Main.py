''' This module will be used to run the project.
You would import and use Enroll.py for enroll feature
and Record.py for recording the attendance
and Update.py for accomplishing the update model feature
'''

# from EnrollSystem import EnrollSystem
# from FaceRecognition import FaceRecognition
# from Plot import Plot
#
# boolean = True
# while boolean:
#     print("Welcome to the Facial Recognition Attendance System\n")
#     print("Your options are as follows:\n1. ENROLL In The System")
#     print("2. RECORD Attendance\n3. PLOT Attendance\n9. To quit\n")
#     #choice = int(input("Please Enter Your Choice (1,2, or 3)\n"))
#     choice = input("Please Enter Your Choice (1,2, or 3)\n")
#     if choice == str(1):
#         # print("ENROLL or UPDATE")
#         print("ENROLL")
#         EnrollSystem.enroll_user()
#     elif choice == str(2):
#         print("RECORD")
#         FaceRecognition.recognize_faces()
#     elif choice == str(3):
#         print("PLOT")
#         obj3 = Plot()
#         obj3.plot()
#
#     elif choice == str(9):
#         boolean = False
#         #exit()
#         #break
#     else:
#         print("Incorrect input provided, please try again")

from EnrollSystem import EnrollSystem
from FaceRecognition_copy import FaceRecognition
from Plot import Plot


def display_menu():
    """Display a graphical menu."""
    print("╔═════════════════════════════════════════════════════╗")
    print("║ Welcome to the Facial Recognition Attendance System ║")
    print("╚═════════════════════════════════════════════════════╝")
    print("\nYour options are as follows:")
    print("╔═══════════════════╗")
    print("║ 1. ENROLL         ║")
    print("║ 2. RECORD         ║")
    print("║ 3. PLOT           ║")
    print("║ 9. QUIT           ║")
    print("╚═══════════════════╝")
    return int(input("Please Enter Your Choice (1,2, 3, or 9): "))


if __name__ == "__main__":
    while True:
        choice = display_menu()

        if choice == 1:
            print("\nENROLL")
            EnrollSystem.enroll_user()
        elif choice == 2:
            print("\nRECORD")
            FaceRecognition.recognize_faces()
        elif choice == 3:
            print("\nPLOT")
            obj3 = Plot()
            obj3.plot()
        elif choice == 9:
            print("\nExiting the system. Goodbye!")
            break
        else:
            print("\nIncorrect input provided, please try again")



