import cv2
import config
import numpy as np
from src.utils.utils import Label, Button


class UIDrawer(object):

    def __init__(self, action_list):
        self.bg = np.ones((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype="uint8") * 111
        self.bg = cv2.rectangle(self.bg, (config.VIDEO_X - 2, config.VIDEO_Y - 2),
                           (config.VIDEO_X + config.VIDEO_WIDTH + 2, config.VIDEO_Y + config.VIDEO_HEIGHT + 2),
                           (0, 0, 0), 2)
        self.bg[config.VIDEO_Y:config.VIDEO_Y + config.VIDEO_HEIGHT,
        config.VIDEO_X:config.VIDEO_X + config.VIDEO_WIDTH] = np.ones((config.VIDEO_HEIGHT, config.VIDEO_WIDTH, 3),
                                                                      dtype="uint8") * 177
        self.lb_title = Label("PUSH-UP EVALUATION", 700, (45, 0), (700, 60), (255, 0, 0), 2, 5, None)
        self.btn_x = (config.WINDOW_WIDTH + config.VIDEO_X + config.VIDEO_WIDTH - 200) // 2
        self.btn_file = Button("File (F)", 110, (self.btn_x, 25), (200, 80), 1, 2, action_list[0])
        self.btn_camera = Button("Camera (C)", 185, (self.btn_x, 125), (200, 80), 1, 2, action_list[1])
        self.btn_stop = Button("Stop (S)", 120, (self.btn_x, 250), (200, 80), 1, 2, action_list[2])
        self.btn_quit = Button("Quit (Q)", 100, (self.btn_x, 375), (200, 80), 1, 2, action_list[3])

        self.lb_y = (config.WINDOW_HEIGHT + config.VIDEO_Y + config.VIDEO_HEIGHT - 25) // 2
        self.lb_total = Label("TOTAL: 0", 100, (45, self.lb_y), (200, 50), (0, 255, 0), 1, 3, None)
        self.lb_right = Label("RIGHT: 0", 100, (345, self.lb_y), (200, 50), (0, 255, 255), 1, 3, None)
        self.lb_wrong = Label("WRONG: 0", 100, (545, self.lb_y), (200, 50), (0, 0, 255), 1, 3, None)

        self.lb_fps = Label(f"fps: 0", 100, (config.WINDOW_WIDTH - 175, config.WINDOW_HEIGHT - 50), (175, 50),
                       (255, 255, 255), 1, 3, None)

        self.btn_list = [self.btn_file, self.btn_camera, self.btn_stop, self.btn_quit]
        self.utils = [self.lb_title, self.btn_file, self.btn_camera, self.btn_stop, self.btn_quit, self.lb_total, self.lb_right, self.lb_wrong, self.lb_fps]

    def reset_btn(self):
        for btn in self.btn_list:
            btn.set_state(0)

    def get_btn_list(self):
        return self.btn_list

    def update_fps(self, fps):
        self.lb_fps.set_text(f"fps: {str(int(self.fps))}")

    def update_ui(self, video_frame, l):
        count, no_right, no_wrong, fps = l
        self.lb_total.set_text(f"TOTAL: {int(count)}")
        self.lb_right.set_text(f"RIGHT: {no_right}")
        self.lb_wrong.set_text(f"RIGHT: {no_wrong}")
        self.lb_fps.set_text(f"fps: {(fps)}")
        x, y, w, h = config.VIDEO_X, config.VIDEO_Y, config.VIDEO_WIDTH, config.VIDEO_HEIGHT
        video_frame = cv2.resize(video_frame, dsize=(w, h))
        self.bg[self.lb_y:, ] = np.ones((config.WINDOW_HEIGHT - self.lb_y, config.WINDOW_WIDTH, 3), dtype="uint8") * 111
        self.bg[y:y+h, x:x+w] = video_frame
        for util in self.utils:
            self.bg = util.place(self.bg)
        return self.bg
