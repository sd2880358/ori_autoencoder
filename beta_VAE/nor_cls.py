from model import Classifier
from dataset import preprocess_images
from tensorflow_addons.image import rotate
import numpy as np
import tensorflow as tf


(train_set, train_labels), (test_dataset, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = preprocess_images(train_set)
test_images = preprocess_images(test_dataset)
classifier = Classifier(shape=(28, 28, 1))
classifier_path = checkpoint_path = "./checkpoints/cls_nor"
cls = tf.train.Checkpoint(classifier=classifier)
cls_manager = tf.train.CheckpointManager(cls, classifier_path, max_to_keep=5)
if cls_manager.latest_checkpoint:
    cls.restore(cls_manager.latest_checkpoint)
    print('classifier checkpoint restored!!')
c_t = train_images
c_l = train_labels
classifier.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
classifier.fit(c_t, c_l, epochs=30, verbose=2)
test_loss, test_acc = classifier.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
filePath = "./cls_nor"
checkpoint_path = "./checkpoints/" + filePath
ckpt_save_path = cls_manager.save()
print('Saving checkpoint for epoch {} at {}'.format(1,
                                                    ckpt_save_path))