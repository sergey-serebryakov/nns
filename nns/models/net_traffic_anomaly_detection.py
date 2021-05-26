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

# Time Series Anomaly Detection: Detection of Anomalous Drops with Limited Features and Sparse Examples in Noisy Highly
# Periodic Data
#
# Paper: https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/dfd834facc9460163438b94d53b36f51bb5ea952.pdf
# Additional information: https://nau-datascience.github.io/Time-Series-Anomaly-Detection
#
# Anomaly detection in network traffic. Only two attributes - time stamp and bytes per second sampled at 5 minute
# interval. In total - 14 different data streams. Each data stream gets its own model.
# Three models are considered - DNN, RNN and LSTM. Not clear how many input features they used, I will use 50 because
# this is what is used in examples on the page with additional info.
# **Models**
# 1. DNN model - DNN Regressor - fully-connected model. In total, 10 layers of 200 neurons each, batch size 200,
#    trained for 1200 steps, single linear output layer
# 2. RNN model - 10 layer RNN with 75 units in hidden size, batch size 200 at 2500 steps.
# 3. LSTM - 10 layer model with 70 neurons, single linear output layer.
import tensorflow as tf
from nns.model import Model
from tensorflow.keras.layers import (InputLayer, Dense, LSTM, SimpleRNN)

__ALL__ = ['DNNModel', 'RNNModel', 'LSTMModel']


class DNNModel(Model):
    def __init__(self):
        super().__init__('DNNModel')

    def create(self) -> tf.keras.models.Model:
        model = tf.keras.models.Sequential([InputLayer((50, ), name='input')], name=self.name)
        for i in range(10):
            model.add(Dense(200, activation='relu', name='dense{}'.format(i+1)))
        model.add(Dense(1, activation=None, name='output'))
        return model


class RNNModel(Model):
    def __init__(self):
        super().__init__('RNNModel')

    def create(self) -> tf.keras.models.Model:
        model = tf.keras.models.Sequential([InputLayer((50, 1), name='input')], name=self.name)
        for i in range(9):
            model.add(SimpleRNN(75, return_sequences=True, name='rnn0{}'.format(i+1)))
        model.add(SimpleRNN(75, return_sequences=False, name='rnn10'))
        model.add(Dense(1, activation=None, name='output'))
        return model


class LSTMModel(Model):
    def __init__(self):
        super().__init__('LSTMModel')

    def create(self) -> tf.keras.models.Model:
        model = tf.keras.models.Sequential([InputLayer((50, 1), name='input')], name=self.name)
        for i in range(9):
            model.add(LSTM(70, return_sequences=True, name='lstm0{}'.format(i+1)))
        model.add(LSTM(70, return_sequences=False, name='lstm10'))
        model.add(Dense(1, activation=None, name='output'))
        return model
