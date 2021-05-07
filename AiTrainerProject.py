import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture('1v.mp4')

detector = pm.PoseDetector()
count = 0
dir = 0
pTime = 0

save_name = "PoseDetection.avi"
fps = 25
width = 1280
height = 720
output_size = (width, height)
out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'XVID'), fps, output_size)

while True:
    success, img = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or success == False:
        cap.release()
        cv2.destroyAllWindows()
        break

    img = cv2.resize(img, (1280, 720))
    # img = cv2.imread('1im.jpg')
    # img = cv2.resize(img, (int(img.shape[1]/9), int(img.shape[0]/9)))
    img = detector.find_pose(img, False)
    lmlist = detector.find_position(img, False)
    if len(lmlist) != 0:
        # # Right Arm
        # detector.find_angle(img, 12, 14, 16)
        # Left Arm
        angle = detector.find_angle(img, 11, 13, 15)
        per = np.interp(angle, (210, 325), (0, 100))
        # print(per)
        bar = np.interp(angle, (210, 325), (650, 100))
        # print(angle, per)

        # Check for the dumbbell curls
        color = (0, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)

        # Draw Bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                    color, 4)

        # Draw Curl Count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                    (0, 0, 255), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
    #             (255, 0, 0), 5)
    out.write(img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
