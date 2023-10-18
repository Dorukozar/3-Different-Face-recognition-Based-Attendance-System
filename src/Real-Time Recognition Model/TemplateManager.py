import os
import cv2
import face_recognition
from Face import Face
from collections import defaultdict

class TemplateManager:
    template_path = './../Templates'
    cv2_read_images = []
    classNames = []
    images_with_indexes_stored = []
    template_folder_names_list = []  # stores the list of all files of classes of people
    index_names_list = []  # stores a list of names, with each index corresponding to an image and a persons name

    # def __init__(self):
    #     self.template_path = './../Templates'
    #     self.cv2_read_images = []
    #     self.classNames = []
    @staticmethod
    def load_template_images():
        myList = os.listdir(TemplateManager.template_path)
        for cl in myList:
            #print(f'the path {TemplateManager.template_path}/{cl}')
            curImg = cv2.imread(f'{TemplateManager.template_path}/{cl}')
            TemplateManager.cv2_read_images.append(curImg)
            TemplateManager.classNames.append(os.path.splitext(cl)[0])  # gets rid of .jpg at end


    @staticmethod
    def load_images_one_by_one():

        myList = os.listdir(TemplateManager.template_path)
        # print(myList)
        for cl in myList:
            # curImg = cv2.imread(f'{path}/{cl}')
            # images.append(curImg)
            TemplateManager.template_folder_names_list.append(os.path.splitext(cl)[0])  # gets rid of .jpg at end
        #print(TemplateManager.template_folder_names_list)
        for name in TemplateManager.template_folder_names_list:  # loop through all the folders to get images
            # print("name", name)
            pathForImg = f'{TemplateManager.template_path}/{name}'
            # mylist1 = []
            mylist1 = os.listdir(pathForImg)
            for cl in mylist1:  # loop through every image
                # print(f"path_for_image {pathForImg} cl {cl}")
                #print(f'path reading {pathForImg}/{cl}')
                curImg = cv2.imread(f'{pathForImg}/{cl}')
                TemplateManager.images_with_indexes_stored.append(curImg)
                TemplateManager.index_names_list.append(name)  # add the name of the person to the list for every image
        # print(f"namelist {index_names_list}")
        return TemplateManager.images_with_indexes_stored, TemplateManager.index_names_list

    @staticmethod
    def load_images_to_train():
        #result_dictionary = {}
        result_dictionary = defaultdict(list)
        face_object_dictionary = Face.face_object_dict
        if len(face_object_dictionary.items()) == 0:
            template_names_list = os.listdir(TemplateManager.template_path)
            for name in template_names_list:
                folder_path = os.path.join(TemplateManager.template_path, name)
                Face.create_dictionary_with_face_objects(folder_path)
        for name, object_list in face_object_dictionary.items():
            read_images_list = []
            for obj in object_list:
                current_image_name = obj.image
                image_path = os.path.join(TemplateManager.template_path, name, current_image_name)
                cur_read_img = cv2.imread(image_path)
                #read_images_list.append(cur_read_img)
                result_dictionary[name].append(cur_read_img)
            #result_dictionary[name] = read_images_list
        return result_dictionary

a = TemplateManager.load_images_to_train()
# for key, val in a.items():
#     print(val)
# print(list(a.keys()))
# TemplateManager.load_images_one_by_one()
# print(TemplateManager.images_with_indexes_stored)
    # @staticmethod
    # def find_encodings(images):
    #     encodeList = []
    #     for img in images:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         encode = face_recognition.face_encodings(img)[0]
    #         encodeList.append(encode)
    #     return encodeList


