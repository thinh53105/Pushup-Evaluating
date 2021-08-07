import time
import cv2
import tkinter as tk
from tkinter.filedialog import askopenfilename
from threading import Thread, Lock
from src.keypoint_detector.keypoint_detector import KeypointDetector
from src.counter.signal_processor import LPF
from src.pushup_predictor.predictor import Predictor
from src.utils.evaluator import Evaluator
import config

root = tk.Tk()
root.withdraw()


class VideoStreamer(object):
    def __init__(self):
        self.default_frame = cv2.imread('sample_images/img001.png')
        self.video_path = None
        self.is_con = True
        self.stream = None
        self.frame = None
        self.angle, self.count, self.no_right, self.no_wrong, self.fps, self.frame_count = 0, 0, 0, 0, 0, 0
        self.lock = Lock()
        self.predictor = Predictor(model_paths=(config.MODEL_UP_PATH, config.MODEL_DOWN_PATH), default_frame=self.default_frame)
        self.evaluator = Evaluator(up_list=[], down_list=[])
        self.keypoint_detector = KeypointDetector()
        self.lpf = LPF((160, 140))
        self.open_stream(None)

    def reset(self):
        self.angle, self.count, self.no_right, self.no_wrong = 0, 0, 0, 0

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

        target_frame, target_angle = None, 0
        p_time = 0
        while True:
            if not self.is_con:
                self.lock.acquire()
                self.frame, self.angle = None, 0
                self.lock.release()
                time.sleep(0.1)
                continue
            success, frame = self.stream.read()
            if frame is not None:
                frame, cur_angle = self.keypoint_detector.get_pose(frame)
                if not cur_angle:
                    if self.evaluator.not_empty():
                        self.lock.acquire()
                        self.evaluator.evaluate(config.EVAL_IMG_SIZE)
                        self.lock.release()
                    self.lpf.reset()
                    self.frame_count = 0
                    self.no_right, self.no_wrong = 0, 0
                    self.evaluator.reset()
                    time.sleep(0.1)
                    continue

                if (self.frame_count + 1) % config.FRAME_SKIP_RATE == 0:
                    cur_angle = max(60, cur_angle)
                    if self.lpf.high and cur_angle > target_angle:
                        target_frame, target_angle = frame, cur_angle
                    if not self.lpf.high and cur_angle < target_angle:
                        target_frame, target_angle = frame, cur_angle

                    Fn, state = self.lpf.cal_next(cur_angle)

                    if state == 1:
                        rate = self.predictor.predict(target_frame, 1)
                        if rate < 0.5:
                            up_right = True
                            print("up right")
                        else:
                            up_right = False
                            print("up wrong", rate)

                        self.evaluator.up_list.append((target_frame, up_right, rate))

                        target_angle = 200

                    if state == 0:
                        rate = self.predictor.predict(target_frame, 0)
                        if rate < 0.5:
                            down_right = True
                            print("down right")
                        else:
                            down_right = False
                            print("down wrong", rate)

                        self.evaluator.down_list.append((target_frame, down_right, rate))

                        if up_right and down_right:
                            self.no_right += 1
                        else:
                            self.no_wrong += 1

                        target_angle = 0

                self.frame_count += 1
                self.lock.acquire()
                self.frame = frame
                self.lock.release()
                c_time = time.time()
                self.fps = 1 / (c_time - p_time + 1 ** (-9))
                p_time = c_time

            else:
                self.frame = None

    def get_frame(self):
        frame = self.default_frame
        self.lock.acquire()
        if self.frame is not None:
            frame = self.frame.copy()
        self.lock.release()
        return frame, (self.lpf.count, self.no_right, self.no_wrong, self.fps)

    def stop(self):
        self.is_con = False
        self.lock.acquire()
        self.evaluator.evaluate(config.EVAL_IMG_SIZE)
        self.lock.release()
        self.lpf.reset()
        self.frame_count = 0
        self.no_right, self.no_wrong = 0, 0
        self.evaluator.reset()
        time.sleep(0.1)

