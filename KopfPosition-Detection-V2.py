import cv2
import time
import os
import PoseModul as pm
import math

cap = cv2.VideoCapture(0)

folderPath = "FacialPhoto"
myList = os.listdir(folderPath)
print(myList)
overlaylist = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    #print(f'{folderPath}/{imPath}')
    overlaylist.append(image)

print(len(overlaylist))

pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:

        x1, y1 = lmList[12][1], lmList[12][2]  # Point 2
        x2, y2 = lmList[11][1], lmList[11][2]  # Point 3
        Ax, Ay = (x1 + x2) // 2, (y1 + y2) // 2  # Center 1 -Rechte seite

        length = math.hypot(x2 - x1, y2 - y1)
        # X-Axe
        #length1 = length+70 #Links
        #length2 = length-45 #Rechts

        length1 = length + (length // 3)  # Links
        length2 = length - (length // 7)  #Rechts

        if lmList[0][1] < length2:
            print("face Right")
        #else:
            #print("face")

        if lmList[0][1] > length1:
            print("face Left")
        else:
            print("face")

            #if lmList[0][1] == length:
                #print("face")


        #print(lmList[14])
        #cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (400, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)