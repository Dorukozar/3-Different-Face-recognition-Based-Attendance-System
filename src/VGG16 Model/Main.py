from Enrollment import EnrollSystem
from FaceMatching import FaceMatching


def main_menu():
    print("╔═════════════════════════════════════════════════════╗")
    print("║ Welcome to the Facial Recognition Attendance System ║")
    print("╚═════════════════════════════════════════════════════╝")
    while True:
        answer = input("Have you enrolled before? (yes/no/quit)\n")
        if answer in ['yes', 'y']:
            print("RECORDING ATTENDANCE")
            FaceMatching().record_attendance()
            # return
            while True:
                choice1 = input("Do you want to continue recording attendance? (yes/no)\n")
                if choice1 in ['yes', 'y']:
                    print("RECORDING ATTENDANCE")
                    FaceMatching().record_attendance()
                elif choice1 in ['no', 'n']:
                    print("QUITTING")
                    return
                else:
                    print("You entered wrong input. Please try again.")

        elif answer in ['no', 'n']:
            print("ENROLL NEW USER")
            EnrollSystem().enrollment()
            while True:
                choice = input("Do you want to record attendance? (yes/no)\n")
                if choice in ['yes', 'y']:
                    print("RECORDING ATTENDANCE")
                    FaceMatching().record_attendance()
                    return
                elif choice in ['no', 'n']:
                    print("QUITTING")
                    return
                else:
                    print("You entered wrong input. Please try again.")
        elif answer in ['quit', 'q']:
            print("QUITTING")
            return
        else:
            print("You entered wrong input. Please try again.")


if __name__ == "__main__":
    main_menu()









