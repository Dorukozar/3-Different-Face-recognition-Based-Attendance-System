import os
import cv2
from TemplateManager import TemplateManager
from Face import Face


class CameraCapture:

    # @staticmethod
    def capture(self):
        cap = cv2.VideoCapture(0)
        return cap

    @staticmethod
    def save_user_image(cap, count, fullName, last_name, n, name, randomNum):
        while count < n:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                if count >= 9:
                    if len(TemplateManager.classNames) == 0:
                        os.mkdir('./../Templates/' + fullName)
                        p = './../Templates/' + fullName
                        picture_name_path = f'{p}/{name} {last_name} @{Face.increase_ID(p)}.jpg'
                        # face_object = Face(picture_name_path, p)
                        # cv2.imwrite(f'{p}/{name} {last_name} @{randomNum}.jpg', frame)
                        cv2.imwrite(picture_name_path, frame)
                        Face.create_dictionary_with_face_objects(p)
                        # Face.create_face_objects(p)
                        break
                    for element in TemplateManager.classNames:
                        #print(f"element {element}\n{TemplateManager.classNames}")
                        if element.upper() == fullName.upper():
                            #print("inside if")
                            p = './../Templates/' + fullName
                            picture_name_path = f"{p}/{name} {last_name} @{Face.increase_ID(p)}.jpg"
                            # face_object = Face(picture_name_path, p)
                            # cv2.imwrite(f'{p}/{name} {last_name} @{randomNum}.jpg', frame)
                            cv2.imwrite(f"{picture_name_path}", frame)
                            Face.create_dictionary_with_face_objects(p)
                            # print(Face.face_dictionary)
                            # Face.create_face_objects(p)
                            # for i in face_object.face_object_dict.values():
                            #     print(len(i))
                            #print("face id", face_object.face_id)
                            break
                        else:
                            #print("inside else")
                            path1 = './../Templates/' + fullName
                            if os.path.exists(path1) == False:
                                os.mkdir('./../Templates/' + fullName)
                            pat = './../Templates/' + fullName
                            #print(f"frame {frame}")
                            picture_name_path = f'{pat}/{name} {last_name} @{Face.increase_ID(pat)}.jpg'
                            # face_object = Face(picture_name_path, pat)
                            # cv2.imwrite(f'{pat}/{name} {last_name} @{randomNum}.jpg', frame)
                            cv2.imwrite(picture_name_path, frame)
                            Face.create_dictionary_with_face_objects(pat)
                            # print(Face.face_dictionary)
                            # Face.create_face_objects(pat)
                            # for i in face_object.face_object_dict.values():
                            #     print(len(i))
                            #print("face id",face_object.face_id)
                            break
                count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the camera object and destroy all windows
        cap.release()
        cv2.destroyAllWindows()

















