WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 720
VIDEO_X, VIDEO_Y = 50, 70
scale_rate = 1.5
VIDEO_WIDTH, VIDEO_HEIGHT = int(576 * scale_rate), int(320 * scale_rate)
FRAME_SKIP_RATE = 6
T = 50
BETA = 1 - FRAME_SKIP_RATE / T
MODEL_UP_PATH = "src/pushup_predictor/models/eff_loss_up_8623.h5"
MODEL_DOWN_PATH = "src/pushup_predictor/models/eff_loss_0717_224x224_B1_down.h5"
EVAL_IMG_SIZE = (525, 350)
