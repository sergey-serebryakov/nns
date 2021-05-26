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
from tensorflow.keras.layers import (InputLayer, Dense, Dropout, LSTM, Bidirectional)

__ALL__ = ['LSTMAnomalyDetect', 'KerasAnomalyDetection01', 'KerasAnomalyDetection02']


class LSTMAnomalyDetect(Model):
    """
    Super simple and not particularly useful:
        https://github.com/aurotripathy/lstm-anomaly-detect
    """
    def __init__(self) -> None:
        super().__init__('LSTMAnomalyDetect')

    def create(self) -> tf.keras.models.Model:
        return tf.keras.models.Sequential([
            InputLayer((100,1), name='input'),
            LSTM(64, return_sequences=True, name='lstm1'),
            Dropout(0.2),
            LSTM(256, return_sequences=True, name='lstm2'),
            Dropout(0.2),
            LSTM(100, return_sequences=False, name='lstm3'),
            Dropout(0.2),
            Dense(1, activation='linear', name='output')
        ], name=self.name)


class KerasAnomalyDetection01(Model):
    """
    https://github.com/chen0040/keras-anomaly-detection/blob/master/keras_anomaly_detection/library/recurrent.py#L7
    """
    def __init__(self) -> None:
        super().__init__('KerasAnomalyDetection01')

    def create(self) -> tf.keras.models.Model:
        # Sequence length = 210, number of features = 1, batch size = ?
        return tf.keras.models.Sequential([
            InputLayer((210, 1), name='input'),
            LSTM(128, return_sequences=False, name='lstm1'),
            Dense(128, activation='linear', name='output')
        ], name=self.name)


class KerasAnomalyDetection02(Model):
    """
    https://github.com/chen0040/keras-anomaly-detection/blob/master/keras_anomaly_detection/library/recurrent.py#L218
    """
    def __init__(self):
        super().__init__('KerasAnomalyDetection02')

    def create(self) -> tf.keras.models.Model:
        # Sequence length = 210, number of features = 1, batch size = ?
        return tf.keras.models.Sequential([
            InputLayer((210, 1), name='input'),
            Bidirectional(LSTM(128, return_sequences=False), name='bLSTM'),
            Dense(128, activation='linear', name='output')
        ], name=self.name)
