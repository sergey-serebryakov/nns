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
import tensorflow as tf
from nns.model import Model
from tensorflow.keras.layers import (Dropout, Dense, Input, LSTM, Permute, Conv1D, BatchNormalization, Concatenate,
                                     Activation, GlobalAveragePooling1D)

__ALL__ = ['LSTM_FCN']


class LSTM_FCN(Model):
    """
    Models work OK for univariate time series.
        Project on GitHub: https://github.com/titu1994/LSTM-FCN
        Paper: https://ieeexplore.ieee.org/document/8141873/

    Problems are defined here:
        https://github.com/titu1994/LSTM-FCN/blob/master/utils/constants.py

    Problem 118: http://www.timeseriesclassification.com/description.php?Dataset=PigCVP
    Train size: 104, test size: 208, length: 2000, num_classes: 52
    """
    def __init__(self, max_sequence_length: int = 2000, nb_classes: int = 52, num_cells: int = 8) -> None:
        super().__init__('LSTM_FCN')
        self.max_sequence_length = max_sequence_length
        self.nb_classes = nb_classes
        self.num_cells = num_cells

    def create(self) -> tf.keras.models.Model:
        ip = Input((1, self.max_sequence_length), name='input')

        x = LSTM(self.num_cells)(ip)
        x = Dropout(0.8)(x)

        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = Concatenate()([x, y])

        out = Dense(self.nb_classes, activation='softmax')(x)

        model = tf.keras.models.Model(ip, out, name=self.name)

        return model
