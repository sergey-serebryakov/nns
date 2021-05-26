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
from nns.models.deep_ts import (LSTMAnomalyDetect, KerasAnomalyDetection01, KerasAnomalyDetection02)


class TestDeepTS(ModelTest):
    def tearDown(self) -> None:
        tf.keras.backend.clear_session()

    def test_LSTMAnomalyDetect(self) -> None:
        _ = self.get_summary(LSTMAnomalyDetect(), 'LSTMAnomalyDetect', (8+1, 7))

    def test_KerasAnomalyDetection01(self) -> None:
        _ = self.get_summary(KerasAnomalyDetection01(), 'KerasAnomalyDetection01', (3+1, 7))

    def test_KerasAnomalyDetection02(self) -> None:
        _ = self.get_summary(KerasAnomalyDetection02(), 'KerasAnomalyDetection02', (3+1, 7))
