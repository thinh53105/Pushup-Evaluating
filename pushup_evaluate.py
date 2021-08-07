import cv2
import time
from threading import Thread, Lock
from src.utils.ui_drawer import UIDrawer
from src.utils.video_streamer import VideoStreamer
from src.keypoint_detector.keypoint_detector import KeypointDetector
import config

video_streamer = VideoStreamer().start()
keypoint_detector = KeypointDetector()


def btn_file_action():
    video_streamer.open_file()


def btn_camera_action():
    video_streamer.open_camera()


def btn_stop_action():
    video_streamer.stop()


def btn_quit_action():
    exit(0)


action_list = [btn_file_action, btn_camera_action, btn_stop_action, btn_quit_action]
ui = UIDrawer(action_list)


def mouse_click(event, x, y, flags, param):
    global ui
    btn_list = ui.get_btn_list()

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
                ui.reset_btn()
                btn.set_state(2)
                btn.call_func()


cv2.namedWindow("Push-up Recognition")
cv2.setMouseCallback("Push-up Recognition", mouse_click)


while True:
    video_frame, tup = video_streamer.get_frame()
    bg = ui.update_ui(video_frame, tup)

    cv2.imshow("Push-up Recognition", bg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        ui.btn_quit.call_func()

cv2.destroyAllWindows()
