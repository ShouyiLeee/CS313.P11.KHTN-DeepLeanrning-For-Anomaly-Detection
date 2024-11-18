import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Reshape, Dense, Dropout, Input, BatchNormalization, LeakyReLU

import tensorflow as tf
from tensorflow.keras import layers, ops


class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
  def __str__(self):
    return "Autoencoder"

class VAE(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_encoder_decoder()
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Flatten both target and reconstruction for loss calculation
            data_flat = tf.keras.layers.Flatten()(data)
            reconstruction_flat = tf.keras.layers.Flatten()(reconstruction)
            
            reconstruction_loss = ops.mean(
                tf.keras.losses.binary_crossentropy(data_flat, reconstruction_flat)
            )
            
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = 0.8 * reconstruction_loss + 0.2 * kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def build_encoder_decoder(self):
        @tf.keras.utils.register_keras_serializable(package="Custom", name="Sampling")
        class Sampling(layers.Layer):
            """Sampling layer using (z_mean, z_log_var) to sample z."""
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = ops.shape(z_mean)[0]
                dim = ops.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + ops.exp(0.5 * z_log_var) * epsilon

            def get_config(self):
                return super().get_config()
                
        # Define encoder
        latent_dim = 2
        encoder_inputs = tf.keras.Input(shape=(140, 1))
        x = tf.keras.layers.Flatten()(encoder_inputs)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Define decoder
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(16, activation="relu")(latent_inputs)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        decoder_outputs = tf.keras.layers.Dense(140, activation="sigmoid")(x)
        decoder_outputs = tf.keras.layers.Reshape((140, 1))(decoder_outputs)
        self.decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        return reconstruction

    def calculate_loss(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Flatten both target and reconstruction for loss calculation
        data_flat = tf.keras.layers.Flatten()(data)
        reconstruction_flat = tf.keras.layers.Flatten()(reconstruction)
        
        # Calculate reconstruction loss for each sample
        reconstruction_loss = tf.keras.losses.binary_crossentropy(
            data_flat, 
            reconstruction_flat,
            from_logits=False
        )  # This returns loss per sample
        
        # Calculate KL loss for each sample without taking the mean
        kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1)
        
        # Calculate total loss for each sample
        individual_losses = reconstruction_loss + kl_loss
        
        return individual_losses

    def get_config(self):
        # Save custom attributes if any
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BIGAN(Model):
    def __init__(self):
        super(BIGAN, self).__init__()
        
        self.latent_dim = 2
        self.img_shape = (140, 1)
        lr = 2e-4
        # Create separate optimizers for each component
        self.encoder_optimizer = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.generator_optimizer = Adam(lr, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = Adam(lr, beta_1=0.5, beta_2=0.999)

        # Build all components
        self.encoder = self.build_encoder()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=self.discriminator_optimizer, metrics=['accuracy'])
        self.encoder.compile(optimizer=self.encoder_optimizer)
        self.generator.compile(optimizer=self.generator_optimizer)
        
        # Metrics
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.d_accuracy_metric = tf.keras.metrics.BinaryAccuracy(name="d_accuracy")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")
        self.e_loss_metric = tf.keras.metrics.Mean(name="e_loss")

    def build_encoder(self):
        model = Sequential([
            Flatten(input_shape=(140, )),
            Dense(256),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(64),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(self.latent_dim)
        ])
        
        img = Input(shape=(140,))
        z = model(img)
        return Model(img, z)

    def build_generator(self):
        model = Sequential([
            Dense(64, input_dim=self.latent_dim),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(256),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(140, activation='tanh')
        ])
        
        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)
        return Model(z, gen_img)

    def build_discriminator(self):
        z = Input(shape=(self.latent_dim,))
        img = Input(shape=(140,))
        d_in = tf.keras.layers.Concatenate()([z, img])

        # x = Dense(256)(d_in)
        x = Dense(256)(d_in)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        validity = Dense(1, activation="sigmoid")(x)

        return Model([z, img], validity)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.d_accuracy_metric, self.g_loss_metric, self.e_loss_metric]

    def D_loss(self, DG, DE, eps=1e-6):
        """Custom Discriminator Loss"""
        return -tf.reduce_mean(tf.math.log(DE + eps) + tf.math.log(1 - DG + eps))
    
    def EG_loss(self, DG, DE, eps=1e-6):
        """Custom Encoder-Generator Loss"""
        return -tf.reduce_mean(tf.math.log(DG + eps) + tf.math.log(1 - DE + eps))
    
    def train_step(self, data):
        img = data
        batch_size = tf.shape(img)[0]

        # Sample noise
        z = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train Discriminator
        with tf.GradientTape() as tape:
            # Generate fake images and encode real images
            img_ = self.generator(z, training=True)  # G(z)
            z_ = self.encoder(img, training=True)    # E(x)
            
            # Real pairs [E(x), x] should be classified as valid
            d_real = self.discriminator([z_, img], training=True)
            # Fake pairs [z, G(z)] should be classified as fake
            d_fake = self.discriminator([z, img_], training=True)
            
            # Custom Discriminator loss
            d_loss = self.D_loss(d_fake, d_real)

        # Apply discriminator gradients
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

        # Train Generator and Encoder together
        with tf.GradientTape() as tape:
            # Generate fake images and encode real images
            img_ = self.generator(z, training=True)  # G(z)
            z_ = self.encoder(img, training=True)    # E(x)
            
            # Get discriminator predictions
            fake_pairs = self.discriminator([z, img_], training=True)   # D(z, G(z))
            real_pairs = self.discriminator([z_, img], training=True)   # D(E(x), x)
            
            # Custom Generator and Encoder loss
            ge_loss = self.EG_loss(fake_pairs, real_pairs)
        
        # Get gradients for both generator and encoder
        ge_variables = self.generator.trainable_variables + self.encoder.trainable_variables
        ge_gradients = tape.gradient(ge_loss, ge_variables)
        
        # Split gradients between generator and encoder
        g_gradients = ge_gradients[:len(self.generator.trainable_variables)]
        e_gradients = ge_gradients[len(self.generator.trainable_variables):]
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        self.encoder_optimizer.apply_gradients(
            zip(e_gradients, self.encoder.trainable_variables)
        )

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(ge_loss)
        self.e_loss_metric.update_state(ge_loss)
        self.d_accuracy_metric.update_state(tf.ones((batch_size, 1)), d_real)
        self.d_accuracy_metric.update_state(tf.zeros((batch_size, 1)), d_fake)

        return {m.name: m.result() for m in self.metrics}
    
    def calculate_loss(self, data):
        # Encode data and reconstruct
        z_encoded = self.encoder(data)
        reconstruction = self.generator(z_encoded)
        
        # Flatten data and reconstruction
        data_flat = tf.keras.layers.Flatten()(data)
        reconstruction_flat = tf.keras.layers.Flatten()(reconstruction)

        # Reconstruction loss per item
        reconstruction_loss = tf.keras.losses.binary_crossentropy(
            data_flat,
            reconstruction_flat
        )

        # Adversarial loss per item
        real_labels = tf.ones((tf.shape(data)[0], 1))
        fake_labels = tf.zeros((tf.shape(data)[0], 1))

        real_predictions = self.discriminator([data_flat, z_encoded])
        fake_predictions = self.discriminator([reconstruction_flat, z_encoded])

        d_loss_real = tf.keras.losses.binary_crossentropy(
            real_labels, real_predictions
        )
        d_loss_fake = tf.keras.losses.binary_crossentropy(
            fake_labels, fake_predictions
        )

        # Combine losses without reduction (per item)
        total_loss = reconstruction_loss + 0.5 * (d_loss_real + d_loss_fake)

        return total_loss

    def call(self, data):
        encoded = self.encoder(data)
        
        # Decode the latent representation back to original space
        reconstructed = self.generator(encoded)
        
        return reconstructed