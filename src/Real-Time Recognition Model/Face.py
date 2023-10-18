import os


class Face:
    face_object_dict = {}
    face_dictionary = {}
    def __init__(self, image, folder_path):
        self.folder_path = folder_path
        self.face_id = Face.get_face_id(image)
        #self.face_id_incrementor = Face.increase_ID(folder_path)
        self.image = image
        #self.face_dictionary = {}
        self.folder_name = os.path.basename(folder_path)
        # extract_template_path = folder_path.split('/')
        # extract_template_path.pop(-1)
        # self.template_folder_path = '/'.join(extract_template_path)
        #self.face_object_dict = {}
        #print(self.template_folder_path)

    @staticmethod
    def get_face_id(unread_image):
        id = unread_image.split('@')[-1].split('.')[0]
        return id

    @staticmethod
    def increase_ID(folder_path):
        elements_in_folder = os.listdir(folder_path)
        #elements_in_folder.sort()
        elements_in_folder.sort(key=lambda x: int(x.split("@")[1].split(".")[0]))
        #print(elements_in_folder)
        if len(elements_in_folder) == 0:
            return 1
        last_element = elements_in_folder[-1]
        print(f"last_element {last_element}")
        id = last_element.split('@')[-1].split('.')[0]
        print("last element id", id)
        # print(str(int(id) + 1))
        return str(int(id) + 1)
        # print(l1[-1])
    @staticmethod
    def create_dictionary_with_face_objects(folder_path):
        template_folder_path = folder_path.split('/')
        template_folder_path.pop(-1)
        template_folder_path = '/'.join(template_folder_path)
        template_list = os.listdir(template_folder_path)
        # print(template_folder_path)
        for folder in template_list:
            value_list = []
            face_object_list = []
            folder_list = os.listdir(template_folder_path + '/' + folder)
            for image in folder_list:
                value_list.append(image)
                face_object = Face(image, folder_path)
                face_object_list.append(face_object)
            value_list.sort(key=lambda x: int(x.split("@")[1].split(".")[0]))
            Face.face_dictionary[folder] = value_list
            Face.face_object_dict[folder] = face_object_list

        # print(len(Face.face_dictionary.values()))
        # print(len(Face.face_object_dict.values()))
        #print(self.face_dictionary)




# obj1 = Face("image", './../Templates/Doruk Ozar')
# #obj1.increase_ID('./../Templates/Doruk Ozar')
# obj1.update_dictionary('./../Templates/Doruk Ozar')
# obj1.create_face_objects()
# print(obj1.image)
# print(obj1.face_ID)
# obj2 = Face("image")
# print(obj2.face_ID)
# obj3 = Face("image")
# print(obj3.face_ID)
