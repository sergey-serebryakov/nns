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
from nns.dlbs_models import DeepMNIST
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TestDeepMNIST(TestCase, ModelTest):
    """
        Model.Input((28, 28, 1)),
        Flatten(),
        Model.Dense(2500, name='dense1'),
        Model.Dense(2000, name='dense2'),
        Model.Dense(1500, name='dense3'),
        Model.Dense(1000, name='dense4'),
        Model.Dense(500, name='dense5'),
        Model.Dense(10, 'softmax', name='dense6')

    The tool may support computations for fused implementations (dense + activations). This test is for non-fused
    implementation (that's why you see x2 in dense layers for number of activations).
    """
    MUT = DeepMNIST()
    EXPECTED_LAYERS = [
        # name      out_shape    flops      num_params       num_activations
        ['input',   (28, 28, 1), 0,         0,               28*28*1],
        ['flatten', (784,),      0,         0,               784],
        ['dense1',  (2500,),     784*2500,  784*2500+2500,   2500*2],
        ['dense2',  (2000,),     2500*2000, 2500*2000+2000,  2000*2],
        ['dense3',  (1500,),     2000*1500, 2000*1500+1500,  1500*2],
        ['dense4',  (1000,),     1500*1000, 1500*1000+1000,  1000*2],
        ['dense5',  (500,),      1000*500,  1000*500+500,    500*2],
        ['dense6',  (10,),       500*10,    500*10+10,       10*2]
    ]
