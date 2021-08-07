import cv2
import keras


class PushupPredictor(object):

    def __init__(self):
        self.default_frame = cv2.imread('sample_images/img001.png')
        self.model_up = keras.models.load_model("src/pushup_predictor/models/eff_loss_up_8623.h5")
        self.model_down = keras.models.load_model("src/pushup_predictor/models/eff_acc_down.h5")
        self.test_predict()

    def test_predict(self):
        test_img = self.preprocessing_image(self.default_frame)
        self.model_up.predict(test_img)
        self.model_down.predict(test_img)

    @staticmethod
    def preprocessing_image(img):
        img = cv2.resize(img, dsize=(224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, 224, 224, 3)
        return img

    def predict(self, image, ty=1):
        img = self.preprocessing_image(image)
        if ty == 1:
            return self.model_up.predict(img)[0][0]
        else:
            return self.model_down.predict(img)[0][0]
