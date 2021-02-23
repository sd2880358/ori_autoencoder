import tensorflow as tf

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, beta=4, gamma=1):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )



    @tf.function
    def sample(self, eps=None):
        if eps is None:
          eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar


    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean


    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
          probs = tf.sigmoid(logits)
          return probs
        return logits

class Discriminator(tf.keras.Model):
    def __init__(self, latent_dim, beta=4, gamma=1):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.d1 = tf.keras.layers.Dense(1000, activation='relu')
        self.d2 = tf.keras.layers.Dense(1000, activation='relu')
        self.d3 = tf.keras.layers.Dense(1000, activation='relu')
        self.d4 = tf.keras.layers.Dense(1000, activation='relu')
        self.l = tf.keras.layers.Dense(2)
        self.p = tf.keras.layers.Dense(2, activation='softmax')
    def call(self, inputs, trainning=True):
        X = self.d1(inputs)
        X = self.d2(X)
        X = self.d3(X)
        X = self.d4(X)

        logits = self.l(X)
        probability = self.p(logits)
        return logits, probability

