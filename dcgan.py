from tensorflow.keras import Sequential, Model, Input, backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def accuracy(y_true, y_pred):
    # Custom accuracy metric for use with smooth labelling
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))


def flip_labels(labels, prob=0.05):
    mask = tf.random.uniform(labels.shape) <= prob
    indices = tf.cast(tf.where(mask), tf.int32)
    ones = tf.ones(indices.shape[0])
    vals = tf.scatter_nd(indices, ones, shape=labels.shape)
    return tf.abs(vals - labels)


class DCGAN():
    def __init__(self, dataset, img_dim, num_channels,
                 batch_size=32, lr=2e-4, momentum=0.5, lrelu_alpha=0.2,
                 model_dir='./dcgan/models/', img_dir='./dcgan/images/'):
        self.img_dim = img_dim
        self.img_shape = (img_dim, img_dim, num_channels)
        self.batch_size = batch_size
        self.noise_dim = 100
        self.flip_prob = 0.05  # 5% chance of flipping labels
        self.lrelu_alpha = lrelu_alpha
        self.dataset = dataset.cache().batch(
            batch_size // 2, drop_remainder=True)  # Sample half batch
        self.model_dir = model_dir
        self.img_dir = img_dir

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(
                                       learning_rate=lr, beta_1=momentum),
                                   metrics=[accuracy])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=Adam(learning_rate=lr, beta_1=momentum))

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.noise_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=Adam(learning_rate=lr, beta_1=momentum))

        self.seed = tf.random.normal([self.batch_size // 2, self.noise_dim])

    def build_generator(self):
        noise_shape = (self.noise_dim,)

        weight_init = RandomNormal(mean=0.0, stddev=0.02)

        model = Sequential()

        dim = self.img_dim // 4
        model.add(Dense(dim * dim * 256,
                        use_bias=False,
                        input_shape=noise_shape,
                        kernel_initializer=weight_init,
                        name='gen_dense'))
        model.add(BatchNormalization(name='gen_bn1'))
        model.add(LeakyReLU(name='gen_lrelu1'))
        # Use dropout of 50%. See https://github.com/soumith/ganhacks#17-use-dropouts-in-g-in-both-train-and-test-phase
        model.add(Dropout(0.5, name='gen_drop1'))

        model.add(Reshape((dim, dim, 256), name='gen_reshape'))
        # Note: None is the batch size
        assert model.output_shape == (None, dim, dim, 256)

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same',
                                  use_bias=False, kernel_initializer=weight_init,
                                  name='gen_conv2dt1'))
        assert model.output_shape == (None, dim, dim, 128), model.output_shape
        model.add(BatchNormalization(name='gen_bn2'))
        model.add(LeakyReLU(name='gen_lrelu2'))
        model.add(Dropout(0.5, name='gen_drop2'))

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same',
                                  use_bias=False, kernel_initializer=weight_init,
                                  name='gen_conv2dt2'))
        assert model.output_shape == (None, dim * 2, dim * 2, 64)
        model.add(BatchNormalization(name='gen_bn3'))
        model.add(LeakyReLU(name='gen_lrelu3'))
        model.add(Dropout(0.5, name='gen_drop3'))

        model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',
                                  use_bias=False, activation='tanh',
                                  kernel_initializer=weight_init,
                                  name='gen_conv2dt3'))
        assert model.output_shape == (
            None, *self.img_shape), model.output_shape

        return model

    def build_discriminator(self):

        weight_init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                         input_shape=self.img_shape,
                         kernel_initializer=weight_init,
                         name='dis_conv2d1'))
        model.add(LeakyReLU(alpha=self.lrelu_alpha, name='dis_lrelu1'))
        model.add(Dropout(0.3, name='dis_drop1'))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                         kernel_initializer=weight_init,
                         name='dis_conv2d2'))
        model.add(LeakyReLU(alpha=self.lrelu_alpha, name='dis_lrelu2'))
        model.add(Dropout(0.3, name='dis_drop2'))

        model.add(Flatten(name='dis_flatten'))
        model.add(Dense(1, activation='sigmoid',
                        kernel_initializer=weight_init, name='dis_dense'))

        return model

    def train_step(self, images):

        half_batch = self.batch_size // 2

        # ---------------------
        #  Train Discriminator
        # ---------------------
        noise = tf.random.normal([half_batch, self.noise_dim])

        # Generate a half batch of new images
        gen_imgs = self.generator(noise, training=False)

        # Use one-sided label smoothing. See https://arxiv.org/abs/1701.00160
        # and noisy labelling. See https://github.com/soumith/ganhacks#6-use-soft-and-noisy-labels
        real_targets = tf.random.uniform((half_batch, 1), 0.7, 1.0, tf.float32)
        fake_targets = tf.random.uniform((half_batch, 1), 0.0, 0.3, tf.float32)

        # Flip labels with a 5% chance
        real_targets = flip_labels(real_targets, self.flip_prob)
        fake_targets = flip_labels(fake_targets, self.flip_prob)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(images, real_targets)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_targets)
        d_loss, acc = 0.5 * tf.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([1] * self.batch_size)

        # Train the generator
        g_loss = self.combined.train_on_batch(noise, valid_y)

        return d_loss, g_loss, acc

    def train(self, epochs, save_interval=50, img_interval=10):

        for epoch in range(epochs):
            start = time.time()
            acc_arr = []
            d_loss_arr = []
            g_loss_arr = []
            for img_batch in self.dataset:
                d_loss, g_loss, acc = self.train_step(img_batch)

                d_loss_arr.append(d_loss)
                g_loss_arr.append(g_loss)
                acc_arr.append(acc)

            # Plot the progress
            acc = 100 * np.mean(acc_arr)
            d_loss = np.mean(d_loss_arr)
            g_loss = np.mean(g_loss_arr)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                  (epoch, d_loss, acc, g_loss))
            print('Time for epoch {} is {} sec'.format(
                epoch + 1, time.time() - start))

            # Produce images for the GIF as we go
            if (epoch + 1) % img_interval == 0:
                self.generate_and_save_images(epoch + 1)

            # Save the model every few epochs
            if (epoch + 1) % save_interval == 0:
                self.save(self.model_dir)

        # Generate after the final epoch
        self.generate_and_save_images(epochs)

    def generate_and_save_images(self, epoch):
        r, c = 4, 4
        predictions = self.generator.predict(self.seed)

        # Rescale to [0, 255]
        predictions = (127.5 * predictions + 127.5).astype(np.uint8)

        fig = plt.figure(figsize=(r, c))

        for i in range(r * c):
            plt.subplot(r, c, i + 1)
            plt.imshow(predictions[i])
            plt.axis('off')

        plt.savefig(self.img_dir + 'epoch_{:04d}.png'.format(epoch))

    def save(self, path='./dcgan/models/'):
        self.generator.save(path + 'generator.h5')
        self.discriminator.save(path + 'discriminator.h5')
        self.combined.save(path + 'combined.h5')
        print("Saved models to " + path)

    def load(self, path='./dcgan/models/'):
        self.generator = tf.keras.models.load_model(path + 'generator.h5')
        self.discriminator = tf.keras.models.load_model(
            path + 'discriminator.h5')
        self.combined = tf.keras.models.load_model(path + 'combined.h5')
        print("Loaded models from " + path)
