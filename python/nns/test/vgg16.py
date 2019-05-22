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
from nns.dlbs_models import VGG
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TestVGG16(TestCase, ModelTest):
    """
    Output is consistent with this description: https://dgschwend.github.io/netscope/#/preset/vgg-16
    Except number of operations - so ignore it. Look at the `Details` table, also they use batch = 10.
    This implementation also uses conv with relu, so #params is 2x of what they have.
    The tool may support computations for fused implementations (dense/conv2d + activations). This test is for
    non-fused implementation (that's why you see x2 in dense/conv2d layers for number of activations).
    """
    MUT = VGG(version='vgg16')
    EXPECTED_LAYERS = [
        # name       out_shape         flops                    num_params          num_activations
        ['input',    (224, 224, 3),    0,                       0,                  224*224*3],
        ['conv1_1',  (224, 224, 64),   (3*3*3)*(224*224*64),    64*(3*3*3)+64,      2*(224*224*64)],
        ['conv1_2',  (224, 224, 64),   (3*3*64)*(224*224*64),   64*(3*3*64)+64,     2*(224*224*64)],
        ['pool1',    (112, 112, 64),   0,                       0,                  (112*112*64)],
        ['conv2_1',  (112, 112, 128),  (3*3*64)*(112*112*128),  128*(3*3*64)+128,   2*(112*112*128)],
        ['conv2_2',  (112, 112, 128),  (3*3*128)*(112*112*128), 128*(3*3*128)+128,  2*(112*112*128)],
        ['pool2',    (56, 56, 128),    0,                       0,                  (56*56*128)],
        ['conv3_1',  (56, 56, 256),    (3*3*128)*(56*56*256),   256*(3*3*128)+256,  2*(56*56*256)],
        ['conv3_2',  (56, 56, 256),    (3*3*256)*(56*56*256),   256*(3*3*256)+256,  2*(56*56*256)],
        ['conv3_3',  (56, 56, 256),    (3*3*256)*(56*56*256),   256*(3*3*256)+256,  2*(56*56*256)],
        ['pool3',    (28, 28, 256),    0,                       0,                  (28*28*256)],
        ['conv4_1',  (28, 28, 512),    (3*3*256)*(28*28*512),   512*(3*3*256)+512,  2*(28*28*512)],
        ['conv4_2',  (28, 28, 512),    (3*3*512)*(28*28*512),   512*(3*3*512)+512,  2*(28*28*512)],
        ['conv4_3',  (28, 28, 512),    (3*3*512)*(28*28*512),   512*(3*3*512)+512,  2*(28*28*512)],
        ['pool4',    (14, 14, 512),    0,                       0,                  (14*14*512)],
        ['conv5_1',  (14, 14, 512),    (3*3*512)*(14*14*512),   512*(3*3*512)+512,  2*(14*14*512)],
        ['conv5_2',  (14, 14, 512),    (3*3*512)*(14*14*512),   512*(3*3*512)+512,  2*(14*14*512)],
        ['conv5_3',  (14, 14, 512),    (3*3*512)*(14*14*512),   512*(3*3*512)+512,  2*(14*14*512)],
        ['pool5',    (7, 7, 512),      0,                       0,                  (7*7*512)],
        ['flatten',  (7*7*512,),       0,                       0,                  (7*7*512)],
        ['fc6',      (4096,),          7*7*512*4096,            7*7*512*4096+4096,  2*4096],
        ['dropout6', (4096,),          0,                       0,                  4096],
        ['fc7',      (4096,),          4096*4096,               4096*4096+4096,     2*4096],
        ['dropout7', (4096,),          0,                       0,                  4096],
        ['fc8',      (1000,),          4096*1000,               4096*1000+1000,     2*1000],
    ]
