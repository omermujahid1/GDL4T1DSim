#Omer Mujahid ©
# BG generation conditioned on Insulin and Carbs using the S2S GAN.
#This code implements the S2S GAN along with the training loop.
# This code is not used for data generation.

#Importing Libraries
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import xlsxwriter
from keras import backend
from keras.constraints import Constraint
from keras.initializers import RandomNormal
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Conv1DTranspose
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.models import Model
from matplotlib import pyplot
from numpy import load
from numpy import ones
from numpy import zeros
from numpy.random import randint
from numpy.random import randn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import RMSprop
from openpyxl import Workbook
from numpy import savez_compressed
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Input

#The following code imports insulin, carbs, and BG data from excel sheets.
#It then normalizes the imported data.
#A shift of 90 minutes is introduced in the normalized data.

in_seq1 = pd.read_excel('insulin.xlsx')
in_seq2 = pd.read_excel('carbs.xlsx')
out_seq = pd.read_excel('bg.xlsx')
in_seq1 = pd.DataFrame.to_numpy(in_seq1)
in_seq2 = pd.DataFrame.to_numpy(in_seq2)
out_seq = pd.DataFrame.to_numpy(out_seq)
# reshape series
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
# define generator
scaler = MinMaxScaler()
# scaler = StandardScaler()
# fit and transform in one step
normalized_insulin = scaler.fit_transform(in_seq1)
normalized_carbs = scaler.fit_transform(in_seq2)
normalized_bg = scaler.fit_transform(out_seq)


#Introducing a shift of 90 minutes between input and output.
# i.e. 1 insulin and 1 carbs value is equal to 18 BG values. 

in_signal = np.hstack((normalized_insulin, normalized_carbs))
n_features = in_seq1.shape[1]
tar_signal = np.concatenate((np.roll(normalized_bg, -1, axis=0),
                             np.roll(normalized_bg, -2, axis=0),
                             np.roll(normalized_bg, -3, axis=0),
                             np.roll(normalized_bg, -4, axis=0),
                             np.roll(normalized_bg, -5, axis=0),
                             np.roll(normalized_bg, -6, axis=0),
                             np.roll(normalized_bg, -7, axis=0),
                             np.roll(normalized_bg, -8, axis=0),
                             np.roll(normalized_bg, -9, axis=0),
                             np.roll(normalized_bg, -10, axis=0),
                             np.roll(normalized_bg, -11, axis=0),
                             np.roll(normalized_bg, -12, axis=0),
                             np.roll(normalized_bg, -13, axis=0),
                             np.roll(normalized_bg, -14, axis=0),
                             np.roll(normalized_bg, -15, axis=0),
                             np.roll(normalized_bg, -16, axis=0),
                             np.roll(normalized_bg, -17, axis=0),
                             np.roll(normalized_bg, -18, axis=0),
                             ), axis=1)
n_input = 2
generator = TimeseriesGenerator(in_signal, tar_signal, length=n_input, batch_size=1)

for i in range(len(generator)):
    x, y = generator[i]
    print('%s => %s' % (x, y))

normalized_insulin = normalized_insulin[:, :, None]
normalized_carbs = normalized_carbs[:, :, None]
print(normalized_insulin.shape)
tar_signal = tar_signal[:, :, None]
tar_signal = tf.keras.layers.Reshape((1, 18))(tar_signal)
print(tar_signal.shape)
in_src_sig1 = Input(shape=(1, 1))
in_src_sig2 = Input(shape=(1, 1))
# target BG input
in_target_sig = Input(shape=(1, 18))
merged = Concatenate()([in_src_sig1, in_src_sig2, in_target_sig])
print(merged.shape)
print('Loaded: ', normalized_insulin.shape, normalized_carbs.shape, normalized_bg.shape)
filename = 'glucose_18.npz'
savez_compressed(filename, normalized_insulin, normalized_carbs, tar_signal)

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)
    
#Defining discriminator
def define_discriminator(input_data_shape):
    init = RandomNormal(stddev=0.02)
    const = ClipConstraint(0.01)
    # conditioning insulin and carbs inputs
    in_src_sig1 = Input(shape=in_data_shape)
    in_src_sig2 = Input(shape=in_data_shape)
    # target BG input
    in_target_sig = Input(shape=out_data_shape)
    # concatenate images channel-wise
    merged = Concatenate()([in_src_sig1, in_src_sig2, in_target_sig])
    # C64
    d = Conv1D(64, 9, padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=(1, 20))(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv1D(128, 5, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv1D(256, 5, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv1D(512, 5, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv1D(1024, 5, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv1D(1, 3, padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    patch_out = Activation('linear')(d)
    # define model
    model = Model([in_src_image1, in_src_image2, in_target_image], patch_out)
    # compile model
    # compile model
    opt = RMSprop(learning_rate=0.00001)
    model.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return model

# define the standalone generator model
def define_generator(latent_dim, in_data_shape):
    init = RandomNormal(stddev=0.02)
    # label inputs
    in_sig1 = Input(shape=(in_data_shape))
    in_sig2 = Input(shape=(in_data_shape))
    n_nodes = 1 * 1
    li_1 = Dense(n_nodes)(in_sig1)
    li_2 = Reshape((1, 1))(li_1)
    li_3 = Dense(n_nodes)(in_sig2)
    li_4 = Reshape((1, 1))(li_3)
    # latent space input
    in_lat = Input(shape=(latent_dim,))
    gen_1 = Dense(n_nodes)(in_lat)
    gen_2 = LeakyReLU(alpha=0.2)(gen_1)
    gen_3 = Reshape((1, 1))(gen_2)
    # merge latent and label inputs
    merge = Concatenate()([gen_3, li_2, li_4])
    #CNN operations
    gen_4 = Conv1DTranspose(512, 1, padding='same')(merge)
    gen_5 = LeakyReLU(alpha=0.2)(gen_4)
    gen_6 = Conv1DTranspose(256, 1, padding='same')(gen_5)
    gen_7 = LeakyReLU(alpha=0.2)(gen_6)
    gen_8 = Conv1DTranspose(128, 1, padding='same')(gen_7)
    gen_9 = LeakyReLU(alpha=0.2)(gen_8)
    # output
    out_layer = Conv1DTranspose(18, 1, padding='same', kernel_initializer=init)(gen_9)
    out_sig_1 = Activation('relu')(out_layer)
    out_sig_2 = Reshape((1, 18))(out_sig_1)
    # defining model
    model = Model([in_sig1, in_sig2, in_lat], out_sig_2)
    print(model.summary())
    return model
    
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, in_data_shape, out_data_shape, latent_dim):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src1 = Input(shape=in_data_shape)  # (1,1)
    in_src2 = Input(shape=in_data_shape)  # (1,1)
    gen_label1, gen_label2, gen_noise = g_model.input
    gen_output = g_model([in_src1, in_src2, gen_noise])
    dis_out = d_model([in_src1, in_src2, gen_output])
    # src image as input, generated image and classification output
    model = Model([in_src1, in_src2, gen_noise], [dis_out, gen_output])
    # compile model
    opt = RMSprop(learning_rate=0.00005)
    model.compile(loss=[wasserstein_loss, 'mae'], optimizer=opt)
    print(model.summary())
    return model

# load and prepare training data
def load_real_samples(filename):
    data = load(filename)
    X1, X2, X3 = data['arr_0'], data['arr_1'], data['arr_2']
    return [X1, X2, X3]


# select a batch of random samples, returns insulin, carbs and BG
def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB, trainC = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2, X3 = trainA[ix], trainB[ix], trainC[ix]
    y = ones((n_samples, patch_shape, 1, 1))
    return [X1, X2, X3], y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=360):
    # generate points in the latent space
    x_input = randn(1)
    t_input = np.asarray(x_input)
    z_input = t_input.reshape(1, 1)
    return [z_input]


# generate a batch of data, returns insulin, carbs and target BG
def generate_fake_samples(g_model, sample_a, sample_b, latent_dim, patch_shape):
    # generate fake instance
    z_input = generate_latent_points(latent_dim, sample_a)
    X = g_model.predict([sample_a, sample_b, z_input])
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, 1, 1))
    return X, y
    
# generate samples and save the model
def summarize_performance(step, g_model, dataset, latent_dim, n_samples=1):
    # select a sample of input images
    [X_realA, X_realB, XrealC], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, X_realB, latent_dim, 1)
    # save the generator model
    filename2 = 'Two_inputs_90min%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
    # plot loss
    fig, axs = pyplot.subplots(5)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(d1_hist, label='d-real')
    axs[1].plot(d2_hist, label='d-fake')
    axs[2].plot(g_hist, label='gen')
    axs[3].plot(a1_hist, label='acc_real')
    axs[4].plot(a2_hist, label='acc_fake')
    
def train(d_model, g_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=1):
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB, trainC = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    for i in range(n_steps):
        [X_realA, X_realB, X_realC], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeC, y_fake = generate_fake_samples(g_model, X_realA, X_realB, latent_dim, n_patch)
        d_loss1, d_acc1 = d_model.train_on_batch([X_realA, X_realB, X_realC], y_real)
        d_loss2, d_acc2 = d_model.train_on_batch([X_realA, X_realB, X_fakeC], y_fake)
        [z_input] = generate_latent_points(latent_dim, n_batch)
        g_loss, _, _ = gan_model.train_on_batch([z_input, X_realA, X_realB], [y_real, X_realC])
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        a1_hist.append(d_acc1)
        a2_hist.append(d_acc2)
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo * 1) == 1:
            summarize_performance(i, g_model, dataset, latent_dim)
            plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)

# loading training data
latent_dim = 1
dataset = load_real_samples('glucose_72.npz')
print('Loaded', dataset[0].shape, dataset[1].shape, dataset[2].shape)
# define input shape based on the loaded dataset
in_data_shape = (1, 1)
out_data_shape = (1, 18)
data_shape = (1, 20)
# define the models
d_model = define_discriminator(data_shape)
g_model = define_generator(latent_dim, in_data_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, in_data_shape, out_data_shape, latent_dim)
# train model
train(d_model, g_model, gan_model, dataset, latent_dim)
