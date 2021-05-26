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
import tensorflow as tf
from . import ModelTest
from nns.models.anomaly_detection import (FullyConnectedAutoencoder, LSTMForecaster, LSTMAutoencoder,
                                          SMDAnomalyDetection)


class TestAnomalyDetection(ModelTest):
    def tearDown(self) -> None:
        tf.keras.backend.clear_session()

    def test_FullyConnectedAutoencoder(self) -> None:
        _ = self.get_summary(FullyConnectedAutoencoder(), 'FullyConnectedAutoencoder', (7+1, 7))

    def test_LSTMForecaster(self) -> None:
        _ = self.get_summary(LSTMForecaster(), 'LSTMForecaster', (4+1, 7))

    def test_LSTMAutoencoder(self) -> None:
        _ = self.get_summary(LSTMAutoencoder(), 'LSTMAutoencoder', (10+1, 7))

    def test_SMDAnomalyDetection(self) -> None:
        _ = self.get_summary(SMDAnomalyDetection(), 'SMDAnomalyDetection', (29+28+1, 7))
