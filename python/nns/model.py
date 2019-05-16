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


class Model(object):
    """ A base class for all NN models. """
    def __init__(self, name):
        self.name = name

    def create(self):
        """ This method must return an instance of `tensorflow.python.keras.models.Model`. """
        raise NotImplementedError('Implement me in a derived class.')

    @staticmethod
    def Dense(size, activation='relu', **kwargs):
        return tf.keras.layers.Dense(size, activation=activation, **kwargs)

    @staticmethod
    def Input(shape, name='input'):
        return tf.keras.layers.Input(shape, name=name)
