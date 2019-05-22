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
import copy
from tensorflow.python.keras.layers import Dense, Conv1D, Conv2D, Conv2DTranspose, Dropout, BatchNormalization
from tensorflow.python.keras.layers import RNN, SimpleRNNCell, GRUCell, LSTMCell, Bidirectional, TimeDistributed
from tensorflow.python.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
from tensorflow.python.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from tensorflow.python.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.python.keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from tensorflow.python.keras.layers import Flatten, Reshape, RepeatVector, Lambda
from tensorflow.python.keras.layers import Activation, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU
from tensorflow.python.keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate      # no Dot
from tensorflow.python.keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from tensorflow.python.keras.layers import ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.activations import linear

from tensorflow.python.keras import backend
from tensorflow.python import ConfigProto, Session
import gc

from tensorflow.python.keras import models
from .model import Model


def reset_keras(model):
    """Reset Keras Session.
        https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/18
        https://github.com/keras-team/keras/issues/12625
    """
    sess = backend.get_session()
    backend.clear_session()
    sess.close()
    try:
        del model
    except:
        pass
    gc.collect()
    backend.set_session(Session(config=ConfigProto()))


def printable_dataframe(data):
    """ Refactor this somehow. """
    column_oder = ['name', 'input_shape', 'num_parameters', 'param_memory', 'flops', 'activation_memory']
    columns = {'name': 'Model', 'input_shape': 'Input shape', 'num_parameters': '#Parameters',
               'param_memory': 'Model size (MB) FP32', 'flops': 'GFLOPs (multiply-add)',
               'activation_memory': 'Activation size (MB) FP32'}
    df = pd.DataFrame(data, columns=column_oder)
    df.rename(columns=columns, inplace=True)
    return df


class Summary(object):
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.out_shape = kwargs.get('out_shape', None)
        self.num_params = kwargs.get('num_params', 0)
        self.flops = kwargs.get('flops', 0)
        self.num_activations = kwargs.get('num_activations', 0)

    def to_dict(self, **overrides):
        dict_repr = {'name': self.name, 'out_shape': self.out_shape, 'flops': self.flops,
                     'num_params': self.num_params, 'num_activations': self.num_activations}
        dict_repr.update(overrides)
        return dict_repr


class ModelSummary(object):
    """
    FLOPs are computed for batch size = 1. Only Dense and Conv2D layers are taken into account.
    FLOPs are forward FLOPs (inference) and should generally be used to compare models and not
        estimating times.
    Supported layers/wrappers:
        Conv1D, Conv2D, Dense
        Conv2DTranspose
        RNNs/bidirectional RNNs with the following cells: SimpleRNNCell, LSTMCell and GRUCell
        TimeDistributed with Dense, Conv2D and Conv2DTranspose
        All other layers defined in GENERIC_LAYERS

    What if batch size > 1 and need backward/training FLOPs?
       - For batch size N, multiple result by N.
       - For backward FLOPs, multiply results by 2.
       - For training FLOPs, multiply result by 3.
    """
    SCALERS = {'B': 1, 'k': 1e3, 'M': 1e6, 'G': 1e9}
    UNITS = {'B': 'Bytes', 'k': 'KB', 'M': 'MB', 'G': 'GB'}
    FLOPS = {'B': 'FLOPs', 'k': 'kFLOPs', 'M': 'mFLOPs', 'G': 'gFLOPs'}
    # For Generic Layers, FLOPs not taken into account, only activation memory. These layers are not trainable.
    GENERIC_LAYERS = (Dropout,
                      GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,
                      GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D,
                      AveragePooling1D, AveragePooling2D, AveragePooling3D,
                      MaxPooling1D, MaxPooling2D, MaxPooling3D,
                      Flatten, Reshape, RepeatVector, Lambda,
                      Activation, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU,
                      Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate,  # without Dot
                      UpSampling1D, UpSampling2D, UpSampling3D,
                      ZeroPadding1D, ZeroPadding2D, ZeroPadding3D)

    def debug(self, message, *kwargs):
        if self.verbose:
            print(message.format(*kwargs))

    def add(self, summary):
        if not isinstance(summary, Summary):
            raise ValueError("Invalid argument type: '{}' (must be 'Summary').".format(type(summary)))
        self.layers.append(summary)
        self.model.flops += summary.flops
        self.model.num_activations += summary.num_activations

    def add_layer(self, layer, flops=0, repeat_count=1, num_params=None, output_shape=None):
        if not isinstance(layer, Layer):
            raise ValueError("Invalid argument type: '{}' (must be 'Layer').".format(type(layer)))
        try:
            out_shape = (output_shape or layer.output_shape)[1:]
        except AttributeError as e:
            print("Cannot get shape for layer '{}'.".format(layer.name))
            raise e
        num_activations = repeat_count * np.prod(out_shape)
        if isinstance(layer, (Conv1D, Conv2D, Dense)):
            if layer.activation != linear:
                num_activations *= 2
        num_params = num_params or layer.count_params()
        self.add(Summary(name=layer.name, out_shape=out_shape, flops=repeat_count*flops,
                         num_params=num_params, num_activations=num_activations))

    def add_input_layer(self, output_shape):
        self.debug("Found supported layer: type=Input, shape={}.", output_shape)
        output_shape = output_shape[1:]
        self.add(Summary(name='input', out_shape=output_shape, num_activations=np.prod(output_shape)))

    def add_conv_layer(self, layer, repeat_count=1, output_shape=None):
        """      WeightsShape                                      OutputShape
        Conv1D   [FilterDim, InChannels, OutChannels]              [Batch, Length, OutChannels]
        Conv2D   [FilterDim, FilterDim, InChannels, OutChannels]   [Batch, Height, Width, OutChannels]
        """
        output_shape = output_shape or layer.output_shape
        filter_flops = np.prod(layer.weights[0].shape[:-1])  # Last dim is number of filters )
        flops = filter_flops * np.prod(output_shape[1:])     # First dim is batch dim
        self.add_layer(layer, flops=flops, repeat_count=repeat_count, output_shape=output_shape)

    def detailed_summary(self, **column_scalers):
        df = pd.DataFrame(
            [layer.to_dict() for layer in self.layers] + [self.model.to_dict(name='TOTAL')],
            columns=['name', 'out_shape', 'flops', 'num_params', 'num_activations']
        )
        df['params_mem'] = df['num_params'] * 4
        df['activations_mem'] = df['num_activations'] * 4

        for column in column_scalers:
            scaler_unit = column_scalers[column]
            df[column] /= ModelSummary.SCALERS[scaler_unit]
            if column == 'flops':
                df.rename(columns={'flops': ModelSummary.FLOPS[scaler_unit]}, inplace=True)
            elif column in ('params_mem', 'activations_mem'):
                df.rename(columns={column: '{} ({})'.format(column, ModelSummary.UNITS[scaler_unit])}, inplace=True)
        return df

    def summary(self, phase='inference', batch_size=1, flops_units='G', memory_units='M'):
        flops_units = flops_units or 'B'
        memory_units = memory_units or 'B'
        flops = batch_size * self.model.flops / ModelSummary.SCALERS[flops_units]
        if phase == 'training':
            # FLOPs(backward) = 2 * FLOPs(forward)
            flops = flops * 3

        activation_memory = batch_size * self.model.num_activations
        param_memory = self.model.num_params
        if phase == 'training':
            # Add gradients for activations and parameters
            activation_memory *= 2
            # This is confusing to account for gradients in this parameter, so it is commented now.
            # param_memory += self.model.num_params

        bytes_per_param = 4
        activation_memory = activation_memory * bytes_per_param / ModelSummary.SCALERS[memory_units]
        param_memory = param_memory * bytes_per_param / ModelSummary.SCALERS[memory_units]

        return {'name': self.model.name, 'batch': batch_size, 'phase': phase, 'flops': flops,
                'param_memory': param_memory, 'activation_memory': activation_memory, 'input_shape': self.input_shape,
                'num_parameters': self.model.num_params}

    def __init__(self, model, verbose=False):
        """
            TODO: What if user reuses layers with functional API?
        """
        self.layers = []  # Per-layer statistics
        self.input_shape = model.input_shape[1:]
        self.model = Summary(name=model.name, num_params=model.count_params())
        self.model.out_shape = model.output_shape[1:]
        self.verbose = verbose  # If true, print layers used in computations

        self.add_input_layer(model.input_shape)
        for layer in model.layers:
            repeat_count = 1      # A layer is applied this number of times (e.g. TimeDistributed).
            output_shape = None
            if isinstance(layer, TimeDistributed):
                # Shape of this layer is (Batch, Time, Feature1, Feature2, ...). We need to infer shape of the wrapped
                # layer since it does not provide it. But to keep things consistent, batch dimension must be there.
                repeat_count = layer.input_shape[1]  # Batch, Length, FeatureDim1, FeatureDim2, ...
                output_shape = (layer.output_shape[0],) + layer.output_shape[2:]
                layer = layer.layer

            if isinstance(layer, Dense):
                self.add_layer(layer, flops=int(np.prod(layer.weights[0].shape)), repeat_count=repeat_count,
                               output_shape=output_shape)
            elif isinstance(layer, (Conv1D, Conv2D, Conv2DTranspose)):
                self.add_conv_layer(layer, repeat_count, output_shape=output_shape)
            elif isinstance(layer, (RNN, Bidirectional)):
                self.add_rnn_layer(layer)
            elif isinstance(layer, BatchNormalization):
                # Num params will most likely differ from what Keras reports
                # https://stackoverflow.com/questions/42521005/how-the-number-of-parameters-associated-with-batchnormalization-layer-is-2048
                output_shape = output_shape or layer.output_shape
                self.add_layer(layer, flops=0, repeat_count=repeat_count, num_params=2*output_shape[-1],
                               output_shape=output_shape)
            elif isinstance(layer, ModelSummary.GENERIC_LAYERS):
                self.add_layer(layer, flops=0, repeat_count=repeat_count, output_shape=output_shape)
            else:
                # Keep it here
                print("Layer not recognized (type={}, name={})".format(str(type(layer)), layer.name))

    def add_rnn_layer(self, layer):
        # TODO: Add memory estimations. What's the best conservative strategy?
        num_steps = layer.input_shape[1]  # Number of time steps (sequence length).
        input_size = layer.input_shape[2]  # Number of input features.
        output_size = layer.output_shape[-1]  # Number of output features of this `layer`.
        layer_params = layer.count_params()  # Number of layer parameters.
        layer_name = layer.name  # Name of a layer, will be qualified with cell type.
        layer_output_shape = layer.output_shape[1:]
        num_output_activations = 0

        repeat_count = 1  # If bidirectional, this will be 2
        if isinstance(layer, Bidirectional):
            # - layer.merge_mode:  {'sum', 'mul', 'concat', 'ave', None}
            # - if None, the outputs will not be combined, they will be returned as a list.
            if layer.merge_mode == 'concat':
                # Number of output features in one bRNN branch.
                output_size = output_size // 2
            elif layer.merge_mode is None:
                raise NotImplementedError("Implement me.")
            repeat_count = 2
            layer = layer.layer
            # In bidirectional case, we need to aggregate outputs of two branches and thus need to have this (base impl)
            num_output_activations += np.prod(layer_output_shape)   # [Length, Features]

        # By default, number of layer FLOPs is equal to RNN FLOPs
        layer_flops = repeat_count * (num_steps * (input_size * output_size + output_size * output_size))
        if isinstance(layer.cell, SimpleRNNCell):
            # 1. Assumed implementation (all output shapes below: [output_size,]):
            #    a.  h = x*W_xh + b
            #    b.  output = h + prev_output*W_hh
            #    c.  output = activation(output)
            rnn_cell = 'SimpleRNN'
            num_cell_activations = (2 if layer.activation == linear else 3) * output_size
        elif isinstance(layer.cell, LSTMCell):
            # 1. Assumed implementation (all output shapes below: [output_size,]):
            #    x_i = x*W_xi + b_xi                             1
            #    x_f = x*W_xf + b_xf                             1
            #    x_c = x*W_xc + b_xc                             1
            #    x_o = x*W_xo + b_xo                             1
            #    i = recurrent_activation(x_i + h*W_hi)          1/2
            #    f = recurrent_activation(x_f + h*W_hf)          1/2
            #    c = f.*h + i.*activation(x_c + h*W_hc)          1/2
            #    o = recurrent_activation(x_o + h*W_ho)          1/2
            #    h = o.*activation(c)                            1/2
            rnn_cell = 'LSTM'
            layer_flops *= 4
            num_internal_activations = 9
            if layer.recurrent_activation != linear:
                num_internal_activations += 4
            if layer.activation != linear:
                num_internal_activations += 1
            num_cell_activations = num_internal_activations * output_size
        elif isinstance(layer.cell, GRUCell):
            # 1. Assumed implementation (all output shapes below: [output_size,]):
            #    a.  x_z = x*W_xz + b_xz                                1
            #    b.  x_r = x*W_xr + b_xr                                1
            #    c.  x_h = x*W_xh + b_xh                                1
            #    d.  r_z = h*W_hz + b_hz                                1
            #    e.  r_r = h*W_hr + b_hr                                1
            #    f.  r_h = h*W_hh + b_hh                                1
            #    g.  z = recurrent_activation(x_z + r_z)                1 or 2
            #    h.  r = recurrent_activation(x_r + r_r)                1 or 2
            #    i.  r_h = r.*r_h                                       1
            #    j.  hh = activation(x_h + r_h)                         1 or 2
            rnn_cell = 'GRU'
            layer_flops *= 3
            num_internal_activations = 10
            if layer.recurrent_activation != linear:
                num_internal_activations += 2
            if layer.activation != linear:
                num_internal_activations += 1
            num_cell_activations = num_internal_activations * output_size
        else:
            raise NotImplementedError("Unknown RNN Cell '{}'.".format(str(type(layer.cell))))

        num_activations = num_output_activations + repeat_count * num_steps * num_cell_activations
        self.add(Summary(name="{} ({})".format(layer_name, rnn_cell),
                         out_shape=layer_output_shape,
                         flops=layer_flops,
                         num_params=layer_params,
                         num_activations=num_activations))
        if self.verbose:
            print("Found supported layer: type={}, num_steps={}, input_size={}, "
                  "output_size={}.".format(rnn_cell, num_steps, input_size, output_size))


def estimate(model, inference, training):
    """ Helper function. """
    if isinstance(model, Model):
        model = model.create()
    if not isinstance(model, models.Model):
        raise ValueError('Unknown model format')

    summary = ModelSummary(model)
    reset_keras(model)

    if inference is not None:
        inference.append(summary.summary(phase='inference'))
    if training is not None:
        training.append(summary.summary(phase='training'))
    return summary.detailed_summary(flops='G', params_mem='M', activations_mem='M')


class ModelTest:
    """ A base class to test model summary class.

    Child classes derive from TestCase and ModelTest. Them unittest framework calls test_model.
    """
    PARAMS = {'name': 0, 'out_shape': 1, 'flops': 2, 'num_params': 3, 'num_activations': 4,
              'params_mem': 5, 'activations_mem': 6}

    def verify(self, actual_layer, expected_layer):
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

    def test_model(self):
        """ Test model and model summary object.
        """
        # Update table of expected layers. Add memory requirements for parameters and activations.
        # noinspection PyUnresolvedReferences
        expected_layers = copy.deepcopy(self.EXPECTED_LAYERS)
        for i in range(len(expected_layers)):
            expected_layers[i].append(expected_layers[i][ModelTest.PARAMS['num_params']] * 4)
            expected_layers[i].append(expected_layers[i][ModelTest.PARAMS['num_activations']] * 4)

        # noinspection PyUnresolvedReferences
        summary = ModelSummary(self.MUT.create())
        actual_layers = summary.detailed_summary()
        print(actual_layers)

        for i in range(len(expected_layers)):
            self.verify(actual_layers.iloc[i], expected_layers[i])
