from model import Classifier
from dataset import preprocess_images
from tensorflow_addons.image import rotate
import numpy as np
import tensorflow as tf


(train_set, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = preprocess_images(test_dataset)
classifier = Classifier(shape=(28, 28, 1))
c_t = test_images
c_l = test_labels
for d in range(0, 180, 10):
    degree = np.radians(d)
    r_t = rotate(test_images, degree)
    c_t = np.concatenate((c_t, r_t))
    c_l = np.concatenate((c_l, test_labels))
classifier.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
classifier.fit(c_t, c_l, epochs=30, verbose=0)
filePath = "./classifier"
checkpoint_path = "./checkpoints/" + filePath
ckpt = tf.train.Checkpoint(classifier=classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path)
ckpt_save_path = ckpt_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(1,
                                                    ckpt_save_path))