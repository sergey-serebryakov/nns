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

import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Conv2D, RNN, SimpleRNNCell, GRUCell, Conv2DTranspose
from tensorflow.python.keras.layers import LSTMCell, Bidirectional, Dense, TimeDistributed


class ModelSummary(object):
    """
    FLOPs are computed for batch size = 1. Only Dense and Conv2D layers are taken into account.
    FLOPs are forward FLOPs (inference) and should generally be used to compare models and not
        estimating times.
    Supported layers/wrappers:
        Dense
        Conv2D
        Conv2DTranspose
        RNNs/bidirectional RNNs with the following cells: SimpleRNNCell, LSTMCell and GRUCell
        TimeDistributed with Dense, Conv2D and Conv2DTranspose

    What if batch size > 1 and need backward/training FLOPs?
       - For batch size N, multiple result by N.
       - For backward FLOPs, multiply results by 2.
       - For training FLOPs, multiply result by 3.
    """

    def __init__(self, model, verbose=False):
        """
            TODO: What if user reuses layers with functional API?
        """
        self.name = model.name
        self.layers = []  # Per-layer statistics
        self.gflops = 0.0  # Total model gFLOPs
        self.nparams = model.count_params()  # Total model parameters
        self.params_mb = 0  # Total size in MB for model parameters
        self.verbose = verbose  # If true, print layers used in computations

        for layer in model.layers:
            repeat_count = 1
            if isinstance(layer, TimeDistributed):
                repeat_count = layer.input_shape[1]  # Batch, Length, FeatureDim1, FeatureDim2, ...
                layer = layer.layer
            if isinstance(layer, Dense):
                self.compute_dense_layer_stats(layer, repeat_count)
            elif isinstance(layer, Conv2D):
                self.compute_conv2d_layer_stats(layer, repeat_count)
            elif isinstance(layer, Conv2DTranspose):
                self.compute_conv2dtranspose_layer_stats(layer, repeat_count)
            elif isinstance(layer, (RNN, Bidirectional)):
                self.compute_rnn_layer_stats(layer)

        for layer in self.layers:
            layer['gflops'] /= 1e9
            layer['params_mb'] = layer['nparams'] * 4 / (1024 * 1024)
            self.gflops += layer['gflops']
            self.params_mb += layer['params_mb']

    def add_layer(self, **kwargs):
        self.layers.append(kwargs)

    def compute_rnn_layer_stats(self, layer):
        num_steps = layer.input_shape[1]      # Number of time steps (sequence length)
        input_size = layer.input_shape[2]     # Number of input features
        output_size = layer.output_shape[-1]  # Number of output features
        layer_params = layer.count_params()   # Number of layer parameters
        layer_name = layer.name               # Name of a layer, will be qualified with cell type

        repeat_count = 1                      # If bidirectional, this will be 2
        if isinstance(layer, Bidirectional):
            if layer.merge_mode == 'concat':
                output_size = output_size // 2
            elif layer.merge_mode is None:
                raise NotImplementedError("Implement this!")
            repeat_count = 2
            layer = layer.layer

        # By default, number of later FLOPs is equal to RNN FLOPs
        layer_flops = repeat_count * (num_steps * (input_size * output_size + output_size * output_size))
        rnn_cell = None
        if isinstance(layer.cell, SimpleRNNCell):
            rnn_cell = 'SimpleRNN'
        elif isinstance(layer.cell, LSTMCell):
            rnn_cell = 'LSTM'
            layer_flops *= 4
        elif isinstance(layer.cell, GRUCell):
            rnn_cell = 'GRU'
            layer_flops *= 3

        if rnn_cell:
            self.add_layer(
                name="{} ({})".format(layer_name, rnn_cell),
                gflops=layer_flops,
                nparams=layer_params)
        if self.verbose:
            print("Found supported layer: type={}, num_steps={}, input_size={}, "
                  "output_size={}.".format(rnn_cell, num_steps, input_size, output_size))

    def compute_dense_layer_stats(self, layer, repeat_count=1):
        # Ignoring biases
        if self.verbose:
            print("Found supported layer: type=Dense, repeat_count={}.".format(repeat_count))
        self.add_layer(
            name=layer.name,
            gflops=repeat_count * (np.prod(layer.weights[0].shape)),
            nparams=layer.count_params())

    def compute_conv2d_layer_stats(self, layer, repeat_count=1):
        """
            layer.weights[0].shape  :  Filter Shape [FilterDim, FilterDim, InChannels, OutChannels]
            layer.output_shape      :  [Batch, SpatialDim, SpatialDim, OutChannels]
        """
        if self.verbose:
            print("Found supported layer: type=Conv2D, repeat_count={}.".format(repeat_count))
        # Number of flops per one output feature
        filter_flops = np.prod(layer.weights[0].shape)
        self.add_layer(
            name=layer.name,
            gflops=repeat_count * (filter_flops * layer.output_shape[1] * layer.output_shape[2]),
            nparams=layer.count_params())

    def compute_conv2dtranspose_layer_stats(self, layer, repeat_count=1):
        """
            Double check this implementation. Consider this layer as Conv2D reversing forward/backward
            passes.
        """
        if self.verbose:
            print("Found supported layer: type=Conv2DTranspose, repeat_count={}.".format(repeat_count))
        # Number of flops per one output feature for 'depth' column
        filter_flops = np.prod(layer.weights[0].shape)
        self.add_layer(
            name=layer.name,
            gflops=repeat_count * (filter_flops * layer.input_shape[1] * layer.input_shape[2]),
            nparams=layer.count_params())

    def __str__(self):
        return "batch_size=1, forward_gflops={:.4f}, nparams={:,}".format(self.gflops, self.nparams)

    def summary(self):
        df = pd.DataFrame(self.layers + [{'name': 'TOTAL', 'gflops': self.gflops, 'nparams': self.nparams,
                                          'params_mb': self.params_mb}],
                          columns=['name', 'gflops', 'nparams', 'params_mb'])
        print(self.name)
        print(df)
