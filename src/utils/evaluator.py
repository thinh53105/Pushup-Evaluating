import cv2
import numpy as np


class Evaluator(object):

    @staticmethod
    def evaluate(u_list, d_list):
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

            cur_name = f'{i + 1}/{l}'
            cv2.namedWindow(cur_name)
            cv2.moveWindow(cur_name, 300, 0)
            cv2.imshow(cur_name, ver)
            cv2.waitKey(0)
        if l != 0:
            cv2.destroyWindow(f'{l}/{l}')