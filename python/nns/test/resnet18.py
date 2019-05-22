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
from nns.dlbs_models import ResNet
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TestResNet18(TestCase, ModelTest):
    """
    Output is consistent with this description: https://dgschwend.github.io/netscope/#/preset/vgg-16
    Except number of operations - so ignore it. Look at the `Details` table, also they use batch = 10.
    This implementation also uses conv with relu, so #params is 2x of what they have.
    The tool may support computations for fused implementations (dense/conv2d + activations). This test is for
    non-fused implementation (that's why you see x2 in dense/conv2d layers for number of activations).
    """
    MUT = ResNet(version='resnet18')
    EXPECTED_LAYERS = [
        # name                    out_shape          flops                    num_params      num_activations
        ['input',                 (224, 224, 3),     0,                       0,              224*224*3],
        ['conv1/conv',            (112, 112, 64),    (7*7*3)*(112*112*64),    64*(7*7*3),     112*112*64],
        ['conv1/bn',              (112, 112, 64),    0,                       2*64,           112*112*64],
        ['conv1/relu',            (112, 112, 64),    0,                       0,              112*112*64],
        ['conv1/mpool',           (56, 56, 64),      0,                       0,              56*56*64],
        ['res1a/branch2a/conv',   (56, 56, 64),      (3*3*64)*(56*56*64),     64*(3*3*64),    56*56*64],
        ['res1a/branch2a/bn',     (56, 56, 64),      0,                       2*64,           56*56*64],
        ['res1a/branch2a/relu',   (56, 56, 64),      0,                       0,              56*56*64],
        ['res1a/branch1/conv',    (56, 56, 64),      (1*1*64)*(56*56*64),     64*(1*1*64),    56*56*64],
        ['res1a/branch2b/conv',   (56, 56, 64),      (3*3*64)*(56*56*64),     64*(3*3*64),    56*56*64],
        ['res1a/branch1/bn',      (56, 56, 64),      0,                       2*64,           56*56*64],
        ['res1a/branch2b/bn',     (56, 56, 64),      0,                       2*64,           56*56*64],
        ['res1a/sum',             (56, 56, 64),      0,                       0,              56*56*64],
        ['res1a/relu',            (56, 56, 64),      0,                       0,              56*56*64],

        ['res1b1/branch2a/conv',  (56, 56, 64),      (3*3*64)*(56*56*64),     64*(3*3*64),    56*56*64],
        ['res1b1/branch2a/bn',    (56, 56, 64),      0,                       2*64,           56*56*64],
        ['res1b1/branch2a/relu',  (56, 56, 64),      0,                       0,              56*56*64],
        ['res1b1/branch2b/conv',  (56, 56, 64),      (3*3*64)*(56*56*64),     64*(3*3*64),    56*56*64],
        ['res1b1/branch2b/bn',    (56, 56, 64),      0,                       2*64,           56*56*64],
        ['res1b1/sum',            (56, 56, 64),      0,                       0,              56*56*64],
        ['res1b1/relu',           (56, 56, 64),      0,                       0,              56*56*64],

        ['res2a/branch2a/conv',   (28, 28, 128),     (3*3*64)*(28*28*128),    128*(3*3*64),   28*28*128],
        ['res2a/branch2a/bn',     (28, 28, 128),     0,                       2*128,          28*28*128],
        ['res2a/branch2a/relu',   (28, 28, 128),     0,                       0,              28*28*128],
        ['res2a/branch1/conv',    (28, 28, 128),     (1*1*64)*(28*28*128),    128*(1*1*64),   28*28*128],
        ['res2a/branch2b/conv',   (28, 28, 128),     (3*3*128)*(28*28*128),   128*(3*3*128),  28*28*128],
        ['res2a/branch1/bn',      (28, 28, 128),     0,                       2*128,          28*28*128],
        ['res2a/branch2b/bn',     (28, 28, 128),     0,                       2*128,          28*28*128],
        ['res2a/sum',             (28, 28, 128),     0,                       0,              28*28*128],
        ['res2a/relu',            (28, 28, 128),     0,                       0,              28*28*128],

        ['res2b1/branch2a/conv',  (28, 28, 128),     (3*3*128)*(28*28*128),   128*(3*3*128),  28*28*128],
        ['res2b1/branch2a/bn',    (28, 28, 128),     0,                       2*128,          28*28*128],
        ['res2b1/branch2a/relu',  (28, 28, 128),     0,                       0,              28*28*128],
        ['res2b1/branch2b/conv',  (28, 28, 128),     (3*3*128)*(28*28*128),   128*(3*3*128),  28*28*128],
        ['res2b1/branch2b/bn',    (28, 28, 128),     0,                       2*128,          28*28*128],
        ['res2b1/sum',            (28, 28, 128),     0,                       0,              28*28*128],
        ['res2b1/relu',           (28, 28, 128),     0,                       0,              28*28*128],

        ['res3a/branch2a/conv',   (14, 14, 256),     (3*3*128)*(14*14*256),   256*(3*3*128),  14*14*256],
        ['res3a/branch2a/bn',     (14, 14, 256),     0,                       2*256,          14*14*256],
        ['res3a/branch2a/relu',   (14, 14, 256),     0,                       0,              14*14*256],
        ['res3a/branch1/conv',    (14, 14, 256),     (1*1*128)*(14*14*256),   256*(1*1*128),  14*14*256],
        ['res3a/branch2b/conv',   (14, 14, 256),     (3*3*256)*(14*14*256),   256*(3*3*256),  14*14*256],
        ['res3a/branch1/bn',      (14, 14, 256),     0,                       2*256,          14*14*256],
        ['res3a/branch2b/bn',     (14, 14, 256),     0,                       2*256,          14*14*256],
        ['res3a/sum',             (14, 14, 256),     0,                       0,              14*14*256],
        ['res3a/relu',            (14, 14, 256),     0,                       0,              14*14*256],

        ['res3b1/branch2a/conv',  (14, 14, 256),     (3*3*256)*(14*14*256),   256*(3*3*256),  14*14*256],
        ['res3b1/branch2a/bn',    (14, 14, 256),     0,                       2*256,          14*14*256],
        ['res3b1/branch2a/relu',  (14, 14, 256),     0,                       0,              14*14*256],
        ['res3b1/branch2b/conv',  (14, 14, 256),     (3*3*256)*(14*14*256),   256*(3*3*256),  14*14*256],
        ['res3b1/branch2b/bn',    (14, 14, 256),     0,                       2*256,          14*14*256],
        ['res3b1/sum',            (14, 14, 256),     0,                       0,              14*14*256],
        ['res3b1/relu',           (14, 14, 256),     0,                       0,              14*14*256],

        ['res4a/branch2a/conv',   (7, 7, 512),       (3*3*256)*(7*7*512),     512*(3*3*256),  7*7*512],
        ['res4a/branch2a/bn',     (7, 7, 512),       0,                       2*512,          7*7*512],
        ['res4a/branch2a/relu',   (7, 7, 512),       0,                       0,              7*7*512],
        ['res4a/branch1/conv',    (7, 7, 512),       (1*1*256)*(7*7*512),     512*(1*1*256),  7*7*512],
        ['res4a/branch2b/conv',   (7, 7, 512),       (3*3*512)*(7*7*512),     512*(3*3*512),  7*7*512],
        ['res4a/branch1/bn',      (7, 7, 512),       0,                       2*512,          7*7*512],
        ['res4a/branch2b/bn',     (7, 7, 512),       0,                       2*512,          7*7*512],
        ['res4a/sum',             (7, 7, 512),       0,                       0,              7*7*512],
        ['res4a/relu',            (7, 7, 512),       0,                       0,              7*7*512],

        ['res4b1/branch2a/conv',  (7, 7, 512),       (3*3*512)*(7*7*512),     512*(3*3*512),  7*7*512],
        ['res4b1/branch2a/bn',    (7, 7, 512),       0,                       2*512,          7*7*512],
        ['res4b1/branch2a/relu',  (7, 7, 512),       0,                       0,              7*7*512],
        ['res4b1/branch2b/conv',  (7, 7, 512),       (3*3*512)*(7*7*512),     512*(3*3*512),  7*7*512],
        ['res4b1/branch2b/bn',    (7, 7, 512),       0,                       2*512,          7*7*512],
        ['res4b1/sum',            (7, 7, 512),       0,                       0,              7*7*512],
        ['res4b1/relu',           (7, 7, 512),       0,                       0,              7*7*512],

        ['apool',                 (1, 1, 512),       0,                       0,              512],
        ['flatten',               (512,),            0,                       0,              512],
        ['output',                (1000,),           512*1000,                512*1000+1000,  2*1000],
    ]
