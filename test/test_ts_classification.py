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
from nns.models.ts_classification import LSTM_FCN


class TestTimeSeriesClassification(ModelTest):
    def tearDown(self) -> None:
        tf.keras.backend.clear_session()

    def test_DeepConvLSTMModel(self) -> None:
        _ = self.get_summary(LSTM_FCN(), 'LSTM_FCN', (16+1, 7))
