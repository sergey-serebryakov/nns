# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# A number of simple models from some of the projects related to detecting anomalies in sensor data.
import typing
import tensorflow as tf
from nns.model import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import (InputLayer, Dense, Input, LSTM, RepeatVector, Reshape, Lambda, Concatenate,
                                     TimeDistributed, Conv2D, Layer, BatchNormalization, Activation, Conv2DTranspose)

__ALL__ = ['FullyConnectedAutoencoder', 'LSTMForecaster', 'LSTMAutoencoder', 'SMDAnomalyDetection']


class FullyConnectedAutoencoder(Model):
    """   369 -> 128 -> 64 -> 16 -> 64 -> 128 -> 369
    In some implementations decoder reconstructs only one window, for instance, in the center.
    """
    def __init__(self, num_features: int = 41, window_size: int = 9) -> None:
        super().__init__('FullyConnectedAutoencoder')
        self.num_features = num_features
        self.window_size = window_size

    def create(self) -> tf.keras.models.Model:
        input_size = self.num_features * self.window_size
        return tf.keras.models.Sequential([
            InputLayer((input_size,), name='input'),
            Dense(128, activation='relu', name='enc/dense01'),
            Dense(64, activation='relu', name='enc/dense02'),
            Dense(16, activation='relu', name='enc/dense03', activity_regularizer=l1(10e-6)),
            Dense(64, activation='relu', name='dec/dense01'),
            Dense(128, activation='relu', name='dec/dense02'),
            Dense(input_size, activation=None, name='dec/dense03')
        ], name=self.name)


class LSTMForecaster(Model):
    def __init__(self, num_features: int = 41, length: int = 9) -> None:
        super().__init__('LSTMForecaster')
        self.num_features = num_features
        self.length = length

    def create(self) -> tf.keras.models.Model:
        inputs = Input(shape=(self.length, self.num_features), name='input')
        X = LSTM(128, return_sequences=True, unroll=True, name='lstm01')(inputs)
        X = LSTM(128, return_sequences=False, unroll=True, name='lstm02')(X)
        outputs = Dense(self.num_features, activation=None, name='output')(X)

        model = tf.keras.models.Model(inputs, outputs, name=self.name)
        return model


class LSTMAutoencoder(Model):
    def __init__(self, num_features: int = 41, length: int = 100, transfer_hs: bool = False, version: int = 1) -> None:
        super().__init__('LSTMAutoencoder')
        self.num_features = num_features
        self.length = length
        self.transfer_hs = transfer_hs
        self.version = version

    def create(self) -> tf.keras.models.Model:
        # Build encoder except for the last layer (the one that computes code)
        inputs = Input(shape=(self.length, self.num_features), name='input')
        X = LSTM(64, return_sequences=True, unroll=True, name='enc/lstm01')(inputs)
        X = LSTM(32, return_sequences=True, unroll=True, name='enc/lstm02')(X)

        # Build last encoder layer
        if self.transfer_hs is True:
            X, state_h, state_c = LSTM(16, return_sequences=False, unroll=True, return_state=True, name='enc/lstm03')(X)
            encoder_state = [state_h, state_c]
        else:
            X = LSTM(16, return_sequences=False, unroll=True, name='enc/lstm03')(X)
            encoder_state = None
        X = Dense(16, activation=None, name='enc/dense04')(X)

        if self.version == 1:
            # Repeat vector pattern
            X = RepeatVector(self.length, name='enc/repeat')(X)
        elif self.version == 2:
            # Teacher forcing pattern
            # Concatenate along time dimension -> Batch, Length +1, features
            # First decoder input is encoder output
            first_step = Reshape((1, self.num_features), name='enc/reshape')(X)
            # Everything else is input except last time
            other_steps = Lambda(lambda x: x[:, :-1, :], name='enc/lambda')(inputs)
            X = Concatenate(axis=1, name='enc/concatenate')([first_step, other_steps])

        # Build decoder
        if self.transfer_hs is True:
            X = LSTM(16, return_sequences=True, unroll=True, name='dec/lstm01')(X, initial_state=encoder_state)
        else:
            X = LSTM(16, return_sequences=True, unroll=True, name='dec/lstm01')(X)
        X = LSTM(32, return_sequences=True, unroll=True, name='dec/lstm02')(X)
        X = LSTM(64, return_sequences=True, unroll=True, name='dec/lstm03')(X)
        outputs = TimeDistributed(Dense(self.num_features, name='dec/dense04'), name='dec/dense_wrapper04')(X)

        model = tf.keras.models.Model(inputs, outputs, name='LSTMAutoencoder')
        return model


class SMDAnomalyDetection(Model):
    """
    https://github.com/DongYuls/SMD_Anomaly_Detection/blob/master/model.py
    TODO: refactor conv_block and conv_transpose_block.

    Implementation is here: https://github.com/DongYuls/SMD_Anomaly_Detection
    Paper is here: https://www.mdpi.com/1424-8220/18/5/1308
    The problem being solved is the anomaly detection based on environment noise. In particular, authors uses sound
    produced by Surface-Mounted Device machine (SMD). Authors propose the following approach. It's hard to work with
    sound in time domain. But going to a frequency domain may be a benefit. Thus, input to an anomaly detection model
    is a spectrogram, or in other words, an image. Authors apply STFT (short-time Fourier transform) with window size
    2048 and stride equal to 512 points.

    Sound time series -> STFT -> Spectrogram -> Image -> CNN-based autoencoder.
    """
    def __init__(self):
        super().__init__('SMDAnomalyDetection')

    @staticmethod
    def conv_block(input_: Layer, num_filters: int, kernel, strides=(1, 1), padding: str = 'same',
                   activation: typing.Optional[str] = 'relu', name: typing.Optional[str] = None,
                   bn: bool = True) -> Layer:
        x = Conv2D(num_filters, kernel, strides=strides, padding=padding, use_bias=False, name=name + '/conv')(input_)
        if bn:
            x = BatchNormalization(scale=False, name=name + '/bn')(x)
        if activation is not None:
            x = Activation(activation=activation, name=name + '/' + activation)(x)
        return x

    @staticmethod
    def conv_transpose_block(input_: Layer, num_filters: int, kernel, strides=(1, 1), padding: str = 'same',
                             activation: typing.Optional[str] = 'relu', name: typing.Optional[str] = None,
                             bn: bool = True) -> Layer:
        x = Conv2DTranspose(num_filters, kernel, strides=strides, padding=padding, use_bias=False,
                            name=name + '/convt')(input_)
        if bn:
            x = BatchNormalization(scale=False, name=name + '/bn')(x)
        if activation is not None:
            x = Activation(activation=activation, name=name + '/' + activation)(x)
        return x

    def create(self) -> tf.keras.models.Model:
        input_ = Input((1024, 32, 1), name='input')
        x = input_
        # Encoder
        ConvBlock = SMDAnomalyDetection.conv_block
        for idx, num_channels in enumerate([64, 64, 96, 96, 128]):
            x = ConvBlock(x, num_channels, (5, 5), (2, 1), name='enc/conv0{}'.format(idx+1))
        for idx, num_channels in enumerate([128, 160, 160]):
            x = ConvBlock(x, num_channels, (4, 4), (2, 2), name='enc/conv0{}'.format(idx+6))
        x = ConvBlock(x, 192, 3, (2, 2), name='enc/conv09')
        x = ConvBlock(x, 192, 3, (2, 2), name='enc/conv10', activation=None, bn=False)

        # Decoder
        ConvTBlock = SMDAnomalyDetection.conv_transpose_block
        x = ConvTBlock(x, 192, (3, 3), (2, 2), name='dec/convt01')
        x = ConvTBlock(x, 160, (3, 3), (2, 2), name='dec/convt02')
        for idx, num_channels in enumerate([160, 128, 128]):
            x = ConvTBlock(x, num_channels, (4, 4), (2, 2), name='dec/convt0{}'.format(idx+3))
        for idx, num_channels in enumerate([96, 96, 64, 64]):
            x = ConvTBlock(x, num_channels, (5, 5), (2, 1), name='dec/convt0{}'.format(idx+6))
        x = ConvTBlock(x, 1, (5, 5), (2, 1), name='dec/conv10'.format(idx+6), bn=False, activation=None)

        return tf.keras.models.Model(input_, x, name=self.name)
