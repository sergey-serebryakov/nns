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

from unittest import TestCase
from nns.nns import ModelTest
from nns.model import Model
import pandas as pd
from tensorflow.python.keras import models
from tensorflow.python.keras import layers


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class LstmRegressor(Model):
    def __init__(self):
        super().__init__('LstmRegressor')

    def create(self):
        return models.Sequential([
            Model.Input((100, 41)),
            layers.LSTM(128, return_sequences=True, unroll=True, use_bias=True, name='lstm01'),
            layers.LSTM(128, return_sequences=False, unroll=True, use_bias=True, name='lstm02'),
            layers.Dense(41, activation=None, name='dense03')
        ], name=self.name)


class TestLstmRegressor(TestCase, ModelTest):
    MUT = LstmRegressor()
    # TODO: Num activations for LSTM layers are not tested.
    EXPECTED_LAYERS = [
        # name             out_shape          flops                         num_params                num_activations
        ['input',          (100, 41),         0,                            0,                        100*41],
        ['lstm01 (LSTM)',  (100, 128),        4*(100*(41*128+128*128)),     4*(41*128+128*128+128),   179200],
        ['lstm02 (LSTM)',  (128, ),           4*(100*(128*128+128*128)),    4*(128*128+128*128+128),  179200],
        ['dense03',        (41, ),            128*41,                       128*41+41,                41],
    ]
