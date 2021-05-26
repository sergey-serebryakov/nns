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
from tensorflow.keras.layers import (InputLayer, Dense,  LSTM, Reshape, Conv2D)

__ALL__ = ['DeepConvLSTMModel']


class DeepConvLSTMModel(Model):
    """
    Seems like they describe exactly this model here:
        https://stackoverflow.com/questions/53144762/convert-lasagne-to-keras-code-cnn-lstm
    """

    def __init__(self, **kwargs) -> None:
        super().__init__('DeepConvLSTMModel')
        self.nb_sensor_channels = kwargs.get('nb_sensor_channels', 113)
        self.num_classes = kwargs.get('num_classes', 18)
        self.sliding_window_length = kwargs.get('sliding_window_length', 24)
        self.final_sequence_length = kwargs.get('final_sequence_length', 8)
        self.num_filters = kwargs.get('num_filters', 64)
        self.filter_size = kwargs.get('filter_size', 5)
        self.num_units_lstm = kwargs.get('num_units_lstm', 128)

    def create(self) -> tf.keras.models.Model:
        return tf.keras.models.Sequential([
            InputLayer((self.sliding_window_length, self.nb_sensor_channels, 1)),                    # (24, 113, 1)
            Conv2D(self.num_filters, (self.filter_size, 1), activation='relu', name='conv1/5x1'),    # (20, 113, 64)
            Conv2D(self.num_filters, (self.filter_size, 1), activation='relu', name='conv2/5x1'),    # (16, 113, 64)
            Conv2D(self.num_filters, (self.filter_size, 1), activation='relu', name='conv3/5x1'),    # (12, 113, 64)
            Conv2D(self.num_filters, (self.filter_size, 1), activation='relu', name='conv4/5x1'),    # (8,  113, 64)
            Reshape((self.final_sequence_length, 113*64)),                                           # (8,  113*64)
            LSTM(self.num_units_lstm, return_sequences=True, name='lstm1'),
            LSTM(self.num_units_lstm, return_sequences=False, name='lstm2'),
            Dense(self.num_classes, activation='softmax', name='prob')
        ], name=self.name)
