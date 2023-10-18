from TemplateManager import TemplateManager
import cv2
import numpy as np
import face_recognition
from datetime import date
from AttendanceManager import AttendanceManager
from CameraCapture import CameraCapture
import time


class FaceRecognition:

    @staticmethod
    def find_encodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    @staticmethod
    def find_face_and_encode_frames(cap):
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        # There could be multiple faces in the image
        # You need to find the locations of faces first
        # Then pass that onto your encoding function
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        return encodesCurFrame, facesCurFrame, img

    @staticmethod
    def recognize_faces():
        TemplateManager.load_images_to_train()
        images, namesList = TemplateManager.load_images_one_by_one()
        encodeListKnown = FaceRecognition.find_encodings(images)
        print('Encoding complete')

        # cap = cv2.VideoCapture(0)
        camera_capture_obj = CameraCapture()
        cap = camera_capture_obj.capture()
        # t = time.time()
        while True:
            encodesCurFrame, facesCurFrame, img = FaceRecognition.find_face_and_encode_frames(cap)
            # print("printing encodes",encodesCurFrame)

            # one by one it will grab one face location and the encoding related to that face
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

                # because we are sending in a list of known encoded faces, we will get lists in return
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                # lowest distance is best match
                # print(faceDis)
                matchIndex = np.argmin(faceDis)  # will provide the index of the best match
                # length = len(classNames)
                if matches[matchIndex]:  # if it exists, the name will be found and printed
                    # name = classNames[matchIndex].upper()
                    name = namesList[matchIndex].upper()  # match the image back to the persons name
                    # print(name)
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    # print(namesList)
                    with open('../Output/Attendance.csv', 'r+') as f:
                        myDataList = f.readlines()

                    today1 = date.today()
                    dateToday1 = today1.strftime('%Y-%m-%d')
                    name = name.title()
                    # print("Name -> ",name)
                    # print(namesList)
                    AttendanceManager.mark_attendance(name)
                    # print(f"namelist -> {namesList}")
                    if name not in namesList:
                        # print(f"namelist -> {namesList}")
                        # print("not in namelist")
                        continue
                    else:
                        # print("we are at else")

                        for i in range(len(myDataList)):
                            # print("inside for loop")
                            list = myDataList[i].split(',')
                            # print("list", list)
                            # for j in range(len(list)):
                            full_date = list[len(list) - 1].split(' ')[0]
                            name_in_list = list[0]
                            # print("name in the list ->", name_in_list)
                            # print("name after configurations", name)
                            # print(full_date)
                            if full_date != dateToday1:
                                # print("-------inside if ")
                                name = name.upper()
                                if name == name_in_list:
                                    # name = name.upper()
                                    AttendanceManager.update_attendance(name)
                                    # update(name)
            # if time.time() - t >= 5:
            #     return
            cv2.imshow('Webcam', img)
            cv2.waitKey(1)

            # if the window for the webcam is closed, the program will stop
            if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
                exit()

# start_time = time.time()
# FaceRecognition.recognize_faces()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f" Time elapsed: {elapsed_time:.2f} seconds")
