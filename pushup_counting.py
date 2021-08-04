import cv2
from src.keypoint_detector import PoseModule as pm
import time
import matplotlib.pyplot as plt

samples_path = "sample_videos"
video = "test9.mp4"
cap = cv2.VideoCapture(samples_path + "/" + video)

detector = pm.PoseDetector()
pTime = 0

angle_list = [160]
filter_list = [140]
up_list, down_list = [], []


frame_count = 0
frame_skip_rate = 6

T = 50
beta = 1 - frame_skip_rate / T

high = True
count = 0

target_frame, target_angle = None, 0
last_high, last_low = 160, 0

while True:
    success, org_frame = cap.read()

    if not success:
        break

    if org_frame.shape[0] > 720 or org_frame.shape[1] > 1080:
        org_frame = cv2.resize(org_frame, dsize=(1080, 720))

    frame = org_frame.copy()

    frame = detector.findPose(frame, draw=False)
    lmList = detector.findPosition(frame, draw=False)

    if lmList:
        cur_angle = None
        if detector.left():
            cur_angle = detector.findAngle(frame, 11, 13, 15, draw=True)
        else:
            cur_angle = detector.findAngle(frame, 12, 14, 16, draw=True)

        if (frame_count + 1) % frame_skip_rate == 0:
            # cur_angle = max(60, cur_angle)
            angle_list.append(cur_angle)

            if high and cur_angle > target_angle:
                target_frame, target_angle = org_frame, cur_angle
            if not high and cur_angle < target_angle:
                target_frame, target_angle = org_frame, cur_angle

            Fn = beta * filter_list[-1] + (1 - beta) * cur_angle
            filter_list.append(Fn)

            if high and Fn > cur_angle and abs(last_high - cur_angle) > 10:
                count += 0.5
                high = False
                up_list.append(target_frame)
                last_high = target_angle
                target_angle = 200
                print(f'last high: {last_high}')

            if not high and Fn < cur_angle and abs(cur_angle - last_low) > 10:
                count += 0.5
                high = True
                down_list.append(target_frame)
                last_low = target_angle
                target_angle = 0
                print(f'last low: {last_low}')

        frame_count += 1

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    cv2.putText(frame, str(int(count)), (300, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)

    cv2.imshow("push-up recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()


plt.figure(figsize=(10, 10))
plt.plot(angle_list)
plt.plot(filter_list)
plt.legend(['raw signal', 'low-pass-filter'])
plt.xlabel('frame count')
plt.ylabel('angle')
plt.show()
