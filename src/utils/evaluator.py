import cv2
import numpy as np


class Evaluator(object):

    def __init__(self, up_list, down_list):
        self.up_list = up_list
        self.down_list = down_list
        self.color = {"RIGHT": (0, 255, 0), "WRONG": (0, 0, 255)}

    def reset(self):
        self.up_list, self.down_list = [], []

    def not_empty(self):
        return self.up_list and self.down_list

    def evaluate(self, img_size):
        l = len(self.down_list)

        for i in range(l):
            up_img, up_right, up_rate = self.up_list[i]
            down_img, down_right, down_rate = self.down_list[i]

            up_img = cv2.resize(up_img, dsize=img_size)
            down_img = cv2.resize(down_img, dsize=img_size)
            up_str = "RIGHT" if up_right else "WRONG"
            down_str = "RIGHT" if down_right else "WRONG"

            ver = np.concatenate((up_img, down_img), axis=0)
            ver = cv2.putText(ver, up_str, (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, self.color[up_str], 3)
            ver = cv2.putText(ver, str(up_rate), (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, self.color[up_str], 3)
            ver = cv2.putText(ver, down_str, (50, 450), cv2.FONT_HERSHEY_PLAIN, 3, self.color[down_str], 3)
            ver = cv2.putText(ver, str(down_rate), (50, 500), cv2.FONT_HERSHEY_PLAIN, 3, self.color[down_str], 3)

            lastname = f'{i}/{l}'
            if i != 0:
                cv2.destroyWindow(lastname)

            cur_name = f'{i + 1}/{l}'
            cv2.namedWindow(cur_name)
            cv2.moveWindow(cur_name, 300, 0)
            cv2.imshow(cur_name, ver)
            cv2.waitKey(0)
        if l != 0:
            cv2.destroyWindow(f'{l}/{l}')