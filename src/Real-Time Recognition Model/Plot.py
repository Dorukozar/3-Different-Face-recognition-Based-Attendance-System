# This module is responsible for plotting the data from attendance.csv
# Basically you read attendance.csv and plot the attendance of each student on a bar graph!
import os
import numpy as np
import matplotlib.pyplot as plt

class Plot:

    def plot(self):
        with open('../Output/Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = []
            attendanceList = []
            for line in myDataList:
                total = 0
                entries = line.split(',')
                nameList.append(entries[0])
                for entry in entries:
                    if entry == ("YES"):
                        total += 1
                attendanceList.append(total)

            maximum = max(attendanceList)
            #print(maximum)
            for i in range(len(attendanceList)):
                num = attendanceList[i]
                perc = num / maximum * 100
                attendanceList[i] = perc

            #width = np.diff(nameList).min()
            plt.bar(nameList, attendanceList, align='center')
            plt.xticks(range(len(nameList)), nameList, rotation=30)
            plt.title('Bargraph of Percentage of Attendance for Students')
            plt.xlabel('Students')
            plt.ylabel('Percentage of Classes Attended')
            plt.show()

#plot()
