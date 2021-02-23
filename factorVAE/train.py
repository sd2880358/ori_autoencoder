from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from model import CVAE, Discriminator
from beta_VAE.dataset import preprocess_images
from tensorflow_addons.image import rotate
import random
import time
from tensorflow.linalg import matvec
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display


mbs = tf.losses.MeanAbsoluteError()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, permuted):
    real_sample_loss = tf.reduce_mean(tf.math.log(real[:, 0]))
    permuted_sample_loss = tf.reduce_mean(tf.math.log(permuted[:, 1]))
    return 0.5 * (real_sample_loss + permuted_sample_loss)

def reconstruction_loss(model, X):
    mean, logvar = model.encode(X)
    Z = model.reparameterize(mean, logvar)
    X_pred = model.decode(Z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=X_pred, labels=X)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return logx_z


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def rotate_vector(vector, matrix):
    matrix = tf.cast(matrix, tf.float32)
    test = matvec(matrix, vector)
    return test


def ori_cross_loss(model, x, d):
    r_x = rotate(x, -d)
    mean, logvar = model.encode(r_x)
    r_z = model.reparameterize(mean, logvar)
    angle = np.radians(d)
    c, s = np.cos(angle), np.sin(angle)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, -s], [s, c]
    phi_z = rotate_vector(r_z, r_m)
    phi_x = model.decode(phi_z)
    # logx_z = cross_entropy(phi_x, x)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=x)
    logx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    return tf.reduce_mean(logx_z)


def rota_cross_loss(model, x, d):
    angle = np.radians(d)
    r_x = rotate(x, -d)
    c, s = np.cos(angle), np.sin(angle)
    latent = model.latent_dim
    r_m = np.identity(latent)
    r_m[0, [0, 1]], r_m[1, [0, 1]] = [c, s], [-s, c]
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    phi_z = rotate_vector(z, r_m)
    phi_x = model.decode(phi_z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=phi_x, labels=r_x)
    logx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    # logx_z = cross_entropy(phi_x, r_x)
    return tf.reduce_mean(logx_z)



def compute_loss(x):
    beta = model.beta
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    p_z = model.permuted(z)
    real_logit, real_pro = discriminator(z, trainning=True)
    fake_logit, fake_pro = discriminator(p_z, trainning=True)
    tc_regulariser = discriminator.gamma * tf.reduce_mean(real_logit[:, 0] - real_logit[:, 1], axis=0)
    x_logit = model.decode(z)
    '''
    reco_loss = reconstruction_loss(x_logit, x)
    kl_loss = kl_divergence(logvar, mean)
    beta_loss = reco_loss + kl_loss * beta
    '''
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    vae_loss = -tf.reduce_mean(logx_z + beta * (logpz - logqz_x)) + tc_regulariser
    disc_loss = discriminator_loss(real_pro, fake_pro)
    return vae_loss, disc_loss

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    file_dir = './image/' + date + file_path
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_dir + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def start_train(epochs, train_dataset, test_dataset, date, filePath):
    @tf.function
    def train_step(x, optimizer):
        d = random.randint(30, 90)
        with tf.GradientTape() as tape:
            r_x = rotate(x, -d)
            ori_loss, ori_disc_loss = compute_loss(x)
            rota_loss, rota_disc_loss = compute_loss(r_x)
            ori_cross_l = ori_cross_loss(model, x, d)
            rota_cross_l = rota_cross_loss(model, x, d)
            total_loss = ori_loss + rota_loss + ori_cross_l + rota_cross_l
            total_disc_loss = ori_disc_loss + rota_disc_loss
        vae_gradients = tape.gradient(total_loss, model.trainable_variables)
        disc_gradients = tape.gradient(total_disc_loss, discriminator.trainable_variables)
        vae_optimizer.apply_gradients(zip(vae_gradients, model.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        '''
        with tf.GradientTape() as tape:
            r_x = rotate(x, -d)
            rota_loss = compute_loss(model, r_x)
        gradients = tape.gradient(rota_loss, model.trainable_variables)  
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        with tf.GradientTape() as tape:
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        '''

    checkpoint_path = "./checkpoints/" + date + filePath
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=vae_optimizer,
                               discriminator=discriminator,
                               dis_optimizer=disc_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    generate_and_save_images(model, 0, test_sample)
    display.clear_output(wait=False)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            train_step(train_x, vae_optimizer, )
            end_time = time.time()
        loss = tf.keras.metrics.Mean()
        d = random.randint(30, 90)
        for test_x in test_dataset:
            r_x = rotate(test_x, d)
            total_loss = rota_cross_loss(model, test_x, d) \
                         + ori_cross_loss(model, test_x, d) \
                         + compute_loss(test_x) \
                         + compute_loss(r_x)
            loss(total_loss)
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
    vae_optimizer = tf.keras.optimizers.Adam(1e-4)
    disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))
    epochs = 100
    num_examples_to_generate = 16
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, 10])
    for i in range(1, 6):
        model = CVAE(latent_dim=16, beta=i)
        discriminator = Discriminator(latent_dim=16, beta=i)
        date = '2_22/'
        str_i = str(i)
        file_path = 'method' + str_i
        start_train(epochs, train_dataset, test_dataset, date, file_path)

