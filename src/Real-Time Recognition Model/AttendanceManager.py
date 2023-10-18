from datetime import datetime, date
import os
import csv


class AttendanceManager:
    @staticmethod
    def mark_attendance(name):
        with open('../Output/Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []  # empty list to keep track of all names found
            name = name.upper()
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])  # entry[0] is just the name, leaving out date
            if name not in nameList:
                now = datetime.now()
                timeString = now.strftime('%H:%M:%S')
                today = date.today()
                dateToday = today.strftime('%Y-%m-%d')
                present = "YES"
                f.writelines(f'\n{name},{present},{dateToday} {timeString}')
                print("inside mark attendance")

    @staticmethod
    def update_attendance(name):
        path = '../Templates'
        # images = []
        classNames = []
        myList = os.listdir(path)
        print(myList)
        for cl in myList:
            # curImg = cv2.imread(f'{path}/{cl}')
            # images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])  # gets rid of .jpg at end
        print(classNames)

        # fullName = firstName + ' ' + lastName
        # print(fullName.upper())

        with open('../Output/Attendance.csv', 'r') as f:
            data = f.readlines()

        with open('../Output/Attendance.csv', 'r+') as file:

            location_idx = []
            now = datetime.now()
            timeString = now.strftime(' %H:%M:%S')
            today = date.today()
            dateToday = today.strftime('%Y-%m-%d')
            # for name in classNames:
            for index, row in enumerate(data):
                a = row.split(',')
                # print("row -> ", row)
                if (name.upper() == a[0]):
                    # print("index -> ", index)
                    location_idx.append(index)
                file.seek(0)
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                for j in range(len(location_idx)):
                    if i == location_idx[j]:
                        row = "YES," + dateToday + timeString
                        data[i] = data[i].replace('\n', '')
                        data[i] = data[i] + ',' + row + '\n'
            with open('../Output/Attendance.csv', 'w+') as csvfile:
                csvfile.writelines(data)
