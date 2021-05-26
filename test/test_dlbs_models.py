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
import pandas as pd
import tensorflow as tf
from . import ModelTest
from nns.models.dlbs import (EnglishAcousticModel, AlexNet, DeepMNIST, VGG, Overfeat, ResNet)


class TestDLBSModels(ModelTest):
    def tearDown(self) -> None:
        tf.keras.backend.clear_session()

    def test_EnglishAcousticModel(self) -> None:
        _ = self.get_summary(EnglishAcousticModel(), 'EnglishAcousticModel', (7+1, 7))

    def test_AlexNet(self) -> None:
        _ = self.get_summary(AlexNet(), 'AlexNet', (15+1, 7))

    def test_AlexNetOWT(self) -> None:
        _ = self.get_summary(AlexNet(version='owt'), 'AlexNetOWT', (15+1, 7))

    def test_DeepMNIST(self) -> None:
        expected_layers = [
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
        layers: pd.DataFrame = self.get_summary(DeepMNIST(), 'DeepMNIST', (8+1, 7))
        self.check_model(layers, expected_layers)

    def test_VGG11(self) -> None:
        _ = self.get_summary(VGG(version='vgg11'), 'VGG11', (8+12+1, 7))

    def test_VGG13(self) -> None:
        _ = self.get_summary(VGG(version='vgg13'), 'VGG13', (10 + 12 + 1, 7))

    def test_VGG19(self) -> None:
        _ = self.get_summary(VGG(version='vgg19'), 'VGG19', (16 + 12 + 1, 7))

    def test_VGG16(self) -> None:
        expected_layers = [
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
        layers: pd.DataFrame = self.get_summary(VGG(version='vgg16'), 'VGG16', (13 + 12 + 1, 7))
        self.check_model(layers, expected_layers)

    def test_Overfeat(self) -> None:
        _ = self.get_summary(Overfeat(), 'Overfeat', (15 + 1, 7))

    def test_ResNet18(self) -> None:
        expected_layers = [
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
        layers = self.get_summary(ResNet(version='resnet18'), 'ResNet18', (72+1, 7))
        self.check_model(layers, expected_layers)
