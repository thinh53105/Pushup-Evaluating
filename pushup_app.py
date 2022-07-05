import time
import threading
import cv2
import numpy as np
import tkinter
from tkinter.filedialog import askopenfilename
import keras
import matplotlib.pyplot as plt
import src.keypoint_detector.PoseModule as pm
from src.utils.utils import Button, Label

root = tkinter.Tk()
root.withdraw()

# model_up = keras.models.load_model()
# model_down = keras.models.load_model("")

model_up = keras.models.load_model("src/pushup_predictor/models/eff_loss_0717_224x224_B1_1_up.h5")
model_down = keras.models.load_model("src/pushup_predictor/models/eff_loss_0717_224x224_B1_down.h5")


def preprocessing_image(img):
    img = cv2.resize(img, dsize=(224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(1, 224, 224, 3)
    return img


test_img = cv2.imread("sample_images/img001.png")
predict_test_image = preprocessing_image(test_img)

model_up.predict(predict_test_image)
model_down.predict(predict_test_image)

WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 720
VIDEO_X, VIDEO_Y = 50, 70
scale_rate = 1.5
VIDEO_WIDTH, VIDEO_HEIGHT = int(576 * scale_rate), int(320 * scale_rate)

bg = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype="uint8")*111
bg = cv2.rectangle(bg, (VIDEO_X-2, VIDEO_Y-2), (VIDEO_X + VIDEO_WIDTH+2, VIDEO_Y + VIDEO_HEIGHT+2), (0, 0, 0), 2)
bg[VIDEO_Y:VIDEO_Y+VIDEO_HEIGHT, VIDEO_X:VIDEO_X+VIDEO_WIDTH] = np.ones((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype="uint8")*177
lb_title = Label("PUSH-UP EVALUATION", 700, (45, 0), (700, 60), (255, 0, 0), 2, 5, None)

play_thread = None

# Variable to detect landmarks and predict
count, no_right, no_wrong = 0, 0, 0
fps = 0

up_list, down_list = [], []

angle_list, filter_list = [], []
eval = False

running = True


def play_video(video_path, x, y, width, height):
    global bg, running, btn_list
    global count, no_right, no_wrong, fps, up_list, down_list
    global angle_list, filter_list, eval

    count, no_right, no_wrong = 0, 0, 0
    detector = pm.PoseDetector()

    angle_list = [160]
    filter_list = [140]

    frame_count = 0
    frame_skip_rate = 6

    T = 50
    beta = 1 - frame_skip_rate / T

    high = True

    up_right = True
    no_right, no_wrong = 0, 0

    up_list = []
    down_list = []

    target_frame, target_angle = None, 0

    cap = cv2.VideoCapture(video_path)
    pTime = 0

    while running:
        success, org_frame = cap.read()
        if not success:
            break

        frame = cv2.resize(org_frame, dsize=(VIDEO_WIDTH, VIDEO_HEIGHT))
        frame = detector.findPose(frame, draw=False)
        lmList = detector.findPosition(frame, draw=False)
        if lmList:
            # frame, _ = detector.findBoundingBox(frame, draw=True)
            if not detector.left():
                cur_angle = detector.findAngle(frame, 12, 14, 16, draw=True)
            else:
                cur_angle = detector.findAngle(frame, 11, 13, 15, draw=True)

            if (frame_count + 1) % frame_skip_rate == 0:
                cur_angle = max(60, cur_angle)
                angle_list.append(cur_angle)

                if high and cur_angle > target_angle:
                    target_frame, target_angle = org_frame, cur_angle
                if not high and cur_angle < target_angle:
                    target_frame, target_angle = org_frame, cur_angle

                Fn = beta * filter_list[-1] + (1 - beta) * cur_angle
                filter_list.append(Fn)

                if high and Fn > cur_angle:
                    count += 0.5
                    high = False

                    predict_image = preprocessing_image(target_frame)
                    rate = model_up.predict(predict_image)[0][0]
                    if rate < 0.5:
                        up_right = True
                        print("up right")
                    else:
                        up_right = False
                        print("up wrong", rate)

                    up_list.append((target_frame, up_right, rate))

                    target_angle = 200

                if not high and Fn < cur_angle:
                    count += 0.5
                    high = True

                    predict_image = preprocessing_image(target_frame)
                    rate = model_down.predict(predict_image)[0][0]
                    if rate < 0.5:
                        down_right = True
                        print("down right")
                    else:
                        down_right = False
                        print("down wrong", rate)

                    down_list.append((target_frame, down_right, rate))

                    if up_right and down_right:
                        no_right += 1
                    else:
                        no_wrong += 1

                    target_angle = 0

            frame_count += 1
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        bg[y:y+height, x:x+width] = frame
        # cv2.waitKey(1)
    reset_btn(btn_list)
    eval = True


def evaluate(agl_list, fil_list, u_list, d_list):
    plt.figure(figsize=(10, 10))
    plt.plot(agl_list)
    plt.plot(fil_list)
    plt.show()

    l = len(d_list)
    color = {"RIGHT": (0, 255, 0),
             "WRONG": (0, 0, 255)}

    for i in range(l):
        up_img, up_right, up_rate = u_list[i]
        down_img, down_right, down_rate = d_list[i]

        up_img = cv2.resize(up_img, dsize=(600, 400))
        down_img = cv2.resize(down_img, dsize=(600, 400))
        up_str = "RIGHT" if up_right else "WRONG"
        down_str = "RIGHT" if down_right else "WRONG"

        ver = np.concatenate((up_img, down_img), axis=0)
        ver = cv2.putText(ver, up_str, (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, color[up_str], 3)
        ver = cv2.putText(ver, str(up_rate), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, color[up_str], 3)
        ver = cv2.putText(ver, down_str, (50, 450), cv2.FONT_HERSHEY_PLAIN, 3, color[down_str], 3)
        ver = cv2.putText(ver, str(down_rate), (50, 500), cv2.FONT_HERSHEY_PLAIN, 3, color[down_str], 3)

        lastname = f'{i}/{l}'
        if i != 0:
            cv2.destroyWindow(lastname)

        cur_name = f'{i+1}/{l}'
        cv2.namedWindow(cur_name)
        cv2.moveWindow(cur_name, 300, 50)
        cv2.imshow(cur_name, ver)
        cv2.waitKey(0)
    if l != 0:
        cv2.destroyWindow(f'{l}/{l}')


def video_player(file_name):
    global running, play_thread, eval
    if eval:
        return

    running = False
    if play_thread:
        play_thread.join()

    running = True
    play_thread = threading.Thread(target=play_video, args=(file_name, VIDEO_X, VIDEO_Y, VIDEO_WIDTH, VIDEO_HEIGHT))
    play_thread.setDaemon(True)
    play_thread.start()


def btn_file_action():
    file_name = askopenfilename(parent=root)
    file_name = file_name.replace("/", "//")
    if file_name:
        video_player(file_name)
    else:
        reset_btn(btn_list)


def btn_camera_action():
    video_player(0)


def btn_stop_action():
    global running
    running = False


quitt = False


def btn_quit_action():
    global running, play_thread, quitt
    running = False
    if play_thread:
        play_thread.join()
    quitt = True


btn_x = (WINDOW_WIDTH + VIDEO_X + VIDEO_WIDTH - 200) // 2
btn_file = Button("File (F)", 110, (btn_x, 25), (200, 80), 1, 2, btn_file_action)
btn_camera = Button("Camera (C)", 185, (btn_x, 125), (200, 80), 1, 2, btn_camera_action)
btn_stop = Button("Stop (S)", 120, (btn_x, 250), (200, 80), 1, 2, btn_stop_action)
btn_quit = Button("Quit (Q)", 100, (btn_x, 375), (200, 80), 1, 2, btn_quit_action)

lb_y = (WINDOW_HEIGHT + VIDEO_Y + VIDEO_HEIGHT - 25) // 2
lb_total = Label("TOTAL: 0", 100, (45, lb_y), (200, 50), (0, 255, 0), 1, 3, None)
lb_right = Label("RIGHT: 0", 100, (345, lb_y), (200, 50), (0, 255, 255), 1, 3, None)
lb_wrong = Label("WRONG: 0", 100, (545, lb_y), (200, 50), (0, 0, 255), 1, 3, None)

lb_fps = Label(f"fps: {str(int(fps))}", 100, (WINDOW_WIDTH-175, WINDOW_HEIGHT-50), (175, 50), (255, 255, 255), 1, 3, None)

btn_list = [btn_file, btn_camera, btn_stop, btn_quit]
utils = [lb_title, btn_file, btn_camera, btn_stop, btn_quit, lb_total, lb_right, lb_wrong, lb_fps]


def reset_btn(list_of_btn):
    for btn in list_of_btn:
        btn.set_state(0)


def mouse_click(event, x, y, flags, param):
    global btn_list

    # mouse hover
    for btn in btn_list:
        if btn.mouse_focus(x, y) and btn.get_state() != 2:
            btn.set_state(1)
        elif btn.get_state() == 1:
            btn.set_state(0)

    # mouse clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        for btn in btn_list:
            if btn.mouse_focus(x, y):
                reset_btn(btn_list)
                btn.set_state(2)
                btn.call_func()


cv2.namedWindow("Push-up Recognition")
cv2.setMouseCallback("Push-up Recognition", mouse_click)


while not quitt:
    bg[lb_y:, ] = np.ones((WINDOW_HEIGHT-lb_y, WINDOW_WIDTH, 3), dtype="uint8")*111

    for util in utils:
        bg = util.place(bg)

    cv2.imshow("Push-up Recognition", bg)

    lb_total.set_text(f"TOTAL: {str(int(count))}")
    lb_right.set_text(f"RIGHT: {str(no_right)}")
    lb_wrong.set_text(f"WRONG: {str(no_wrong)}")
    lb_fps.set_text(f"fps: {str(int(fps))}")

    key = cv2.waitKey(1) & 0xFF
    if eval:
        evaluate(angle_list, filter_list, up_list, down_list)
        eval = False

    if key == ord('q') or quitt:
        btn_quit.call_func()

    elif key == ord('f'):
        btn_file.set_state(2)
        btn_file.call_func()

    elif key == ord('c'):
        btn_camera.set_state(2)
        btn_camera.call_func()

    elif key == ord('s'):
        btn_stop.set_state(2)
        btn_stop.call_func()

running = False
cv2.destroyAllWindows()
