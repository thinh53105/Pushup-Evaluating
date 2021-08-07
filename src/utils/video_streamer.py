import time
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
from threading import Thread, Lock
from src.keypoint_detector.keypoint_detector import KeypointDetector
from src.pushup_predictor.pushup_predictor import PushupPredictor
from src.utils.evaluator import Evaluator
import config
import keras
import matplotlib.pyplot as plt
import numpy as np

root = tk.Tk()
root.withdraw()


class VideoStreamer(object):
    def __init__(self):
        self.default_frame = cv2.imread('sample_images/img001.png')
        self.video_path = None
        self.is_con = True
        self.eval = False
        self.stream = None
        self.frame = None
        self.predictor = PushupPredictor()
        self.evaluator = Evaluator()
        self.angle = 0
        self.count = 0
        self.no_right = 0
        self.no_wrong = 0
        self.fps = 0
        self.lock = Lock()
        self.keypoint_detector = KeypointDetector()
        self.open_stream(None)

    def reset(self):
        self.angle, self.count, self.no_right, self.no_wrong = 0, 0, 0 , 0

    def open_stream(self, video_path):
        self.reset()
        self.is_con = True
        self.stream = cv2.VideoCapture(video_path)

    def open_file(self):
        filename = askopenfilename()
        if filename:
            self.open_stream(filename)

    def open_camera(self):
        self.open_stream(0)

    def start(self):
        t = Thread(target=self.get, args=())
        t.daemon = True
        t.start()
        return self

    def get(self):
        angle_list = [160]
        filter_list = [140]

        frame_count = 0

        high = True
        count = 0
        up_list, down_list = [], []

        target_frame, target_angle = None, 0
        p_time = 0
        while True:
            if not self.is_con:
                self.frame, self.angle = None, 0
                time.sleep(0.1)
                continue
            success, frame = self.stream.read()
            if frame is not None:
                frame, cur_angle = self.keypoint_detector.get_pose(frame)
                if not cur_angle:
                    if up_list and down_list:
                        self.lock.acquire()
                        self.evaluator.evaluate(up_list, down_list)
                        self.lock.release()
                    angle_list = [160]
                    filter_list = [140]
                    frame_count = 0
                    high = True
                    count = 0
                    self.no_right, self.no_wrong = 0, 0
                    up_list, down_list = [], []
                    time.sleep(0.1)
                    continue

                if (frame_count + 1) % config.FRAME_SKIP_RATE == 0:
                    cur_angle = max(60, cur_angle)
                    angle_list.append(cur_angle)

                    if high and cur_angle > target_angle:
                        target_frame, target_angle = frame, cur_angle
                    if not high and cur_angle < target_angle:
                        target_frame, target_angle = frame, cur_angle

                    Fn = config.BETA * filter_list[-1] + (1 - config.BETA) * cur_angle
                    filter_list.append(Fn)

                    if high and Fn > cur_angle:
                        count += 0.5
                        high = False

                        rate = self.predictor.predict(target_frame, 1)
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

                        rate = self.predictor.predict(target_frame, 0)
                        if rate < 0.5:
                            down_right = True
                            print("down right")
                        else:
                            down_right = False
                            print("down wrong", rate)

                        down_list.append((target_frame, down_right, rate))

                        if up_right and down_right:
                            self.no_right += 1
                        else:
                            self.no_wrong += 1

                        target_angle = 0

                frame_count += 1
                self.frame = frame
                self.count = count
                c_time = time.time()
                self.fps = 1 / (c_time - p_time + 1 ** (-9))
                p_time = c_time

            else:
                self.frame = None

    def get_frame(self):
        frame = self.frame.copy() if self.frame is not None else self.default_frame
        return frame, (self.count, self.no_right, self.no_wrong, self.fps)

    def stop(self):
        self.is_con = False
        self.eval = True

    def is_stopping(self):
        return not self.is_con

