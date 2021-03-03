import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from math import floor
from skimage.transform import resize
import numpy as np
from scipy.linalg import sqrtm
from math import ceil

class Inception_score(tf.keras.Model):
    def __init__(self):
        super(Inception_score, self).__init__()
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=[150, 150, 3], classes=10)

    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            new_image = resize(image, new_shape, 0)
            images_list.append(new_image)
        return np.asarray(images_list)

    def inception_score(self, p_yx, eps):
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        return ceil(is_score)

    def process_data(self, X):
        resize_data = self.scale_images(X, (150, 150, 3))
        dataset = tf.cast(resize_data, tf.float32)
        return preprocess_input(dataset)

    def calculate_fid(self, real, fake):
        mu1, sigma1 = real.mean(axis=0), np.cov(real, rowvar=False)
        mu2, sigma2 = fake.mean(axis=0), np.cov(fake, rowvar=False)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def compute_score(self, X, Y, n_split=10, eps=1E-16):
        data_X = self.process_data(X)
        data_Y = self.process_data(Y)
        prediction = self.model.predict(data_X)
        actual = self.model.predict(data_Y)
        fid = self.calculate_fid(prediction, actual)
        score_list = []
        n_part = floor(prediction.shape[0] / n_split)
        for i in range(n_split):
            ix_start, ix_end = i * n_part, (i + 1) * n_part
            subset_X = prediction[ix_start:ix_end, :]
            score_list.append(self.inception_score(subset_X, eps))
        return fid, score_list