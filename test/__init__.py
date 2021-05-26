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
import typing
from unittest import TestCase

import pandas as pd

from nns.nns import ModelSummary
from nns.model import Model as NNSModel


class ModelTest(TestCase):
    """ A base class to test model summary class.

    Child classes derive from ModelTest. Them unittest framework calls test_model.
    """
    PARAMS = {'name': 0, 'out_shape': 1, 'flops': 2, 'num_params': 3, 'num_activations': 4,
              'params_mem': 5, 'activations_mem': 6}

    def get_summary(self, model: NNSModel, name: str, shape: typing.Tuple) -> pd.DataFrame:
        self.assertIsInstance(model, NNSModel)
        self.assertEqual(name, model.name)

        layers: pd.DataFrame = ModelSummary(model.create()).detailed_summary()
        self.assertIsInstance(layers, pd.DataFrame)
        self.assertEqual(shape, layers.shape)

        return layers

    def check_layer(self, actual_layer, expected_layer) -> None:
        """
        Verify that computed layer parameters match expected values.

        :param actual_layer: pandas row with actual (computed) layer parameters
        :param expected_layer: python list with expected (true) layer parameters.
        """
        for param_name in ModelTest.PARAMS:
            index = ModelTest.PARAMS[param_name]
            # noinspection PyUnresolvedReferences
            self.assertEqual(expected_layer[index], actual_layer[param_name],
                             "Failed to match parameter '{}' for layer '{}'".format(param_name, actual_layer['name']))

    def check_model(self, model_or_layers: typing.Union[NNSModel, pd.DataFrame],
                    expected_layers: typing.List[typing.List]) -> None:
        """ Test model and model summary object.
        """
        # Update table of expected layers. Add memory requirements for parameters and activations.
        # noinspection PyUnresolvedReferences
        for i in range(len(expected_layers)):
            expected_layers[i].append(expected_layers[i][ModelTest.PARAMS['num_params']] * 4)
            expected_layers[i].append(expected_layers[i][ModelTest.PARAMS['num_activations']] * 4)

        # noinspection PyUnresolvedReferences
        if isinstance(model_or_layers, NNSModel):
            summary = ModelSummary(model_or_layers.create())
            actual_layers = summary.detailed_summary()
        else:
            actual_layers = model_or_layers

        for i in range(len(expected_layers)):
            self.check_layer(actual_layers.iloc[i], expected_layers[i])
