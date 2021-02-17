from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import model
from dataset import preprocess_images
from tensorflow_addons.image import rotate
import random
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.losses.MeanSquaredError()



def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test


def ori_cross_loss(model, x, z, d):
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)
    l_q = mse(x, phi_x)
    return l_q


def rota_cross_loss(model, x, z, d):
    c, s = np.cos(d), np.sin(d)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    phi_z = rotate_vector(z, r_m)
    phi_q = model.decode(phi_z)
    phi_x = rotate(x, -d)
    l_q = mse(phi_q, phi_x)
    return l_q


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    d = random.randint(0, 360)
    r_x = rotate(x, -d)
    ori_loss = ori_cross_loss(model, r_x, z, d)
    rotate_loss = rota_cross_loss(model, x, z, d)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x) + rotate_loss + ori_loss


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    file_dir = './image'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig('./image/image_at_epoch_{:04d}.png'.format(epoch))




def start_train(epochs, model, train_dataset, test_dataset, date, filePath):
    @tf.function
    def train_step(model, x, optimizer):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    checkpoint_path = "./checkpoints/"+ date + filePath
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    generate_and_save_images(model, 0, test_sample)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(model, train_x, optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(compute_loss(model, test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                    .format(epoch, elbo, end_time - start_time))
            generate_and_save_images(model, epoch, test_sample)
        if (epoch + 1) % 10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                        ckpt_save_path))




if __name__ == '__main__':
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_size = 60000
    batch_size = 32
    test_size = 10000
    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))
    epochs = 100
    num_examples_to_generate = 16
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, 10])
    model = model.CVAE(latent_dim=10)
    date = '2/17'
    file_path = 'method1'
    start_train(epochs, model, train_dataset, test_dataset, date, file_path)

