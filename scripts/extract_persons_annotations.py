import os
import cv2
import csv
import numpy as np
from imutils.object_detection import non_max_suppression


def rects_from_image(frame):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    if isinstance(frame, str):
        frame = cv2.imread(frame)

    (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    return non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    
def get_all_rects_for_img_in_dir(dir_name, output):

    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
        return 

    with open(output, "w") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['path', 'posX', 'posY', 'maxX', 'maxY'])
        
        for filename in os.listdir(dir_name):
            path = os.path.join(dir_name.replace('/', '\\'), filename)

            print(path)

            if not os.path.isfile(path):
                return 
                
            result = rects_from_image(path)

            for (xA, yA, xB, yB) in result:
                writer.writerow([path, xA, yA, xB, yB])


# get_all_rects_for_img_in_dir('C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/src/data_sets/positive',
#                              'C:/Users/Otc_Chingy/PycharmProjects/AdvancedPython/ai_end_of_sem_projects/pedestrian_detection/src/data_sets/positive.csv')