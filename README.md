> __This repository moved to a new location - [https://github.com/mlperf-deepts/nns](https://github.com/mlperf-deepts/nns)__

# Neural network runtime characteristics - FLOPS and memory requirements


The `nns` (Neural Network Summary) package implements a simple class `ModelSummary` that estimates compute and memory requirements for neural networks:

1. Compute requirements are estimated in terms of `FLOPs` where a `FLOP` is a floating point __multiply-add__ operation. The implemented algorithm follows standard approach that takes into account operators that dominate computations: `matrix multiply` and `convoutional` operators. Backward pass is considered to be twice as compute intensive as forward pass:
  ```
  FLOPs(backward) = 2 * FLOPs(forward)
  FLOPs(training) = FLOPs(forward) + FLOPs(backward) = 3 * FLOPs(forward)
  ```
2. Memory requirements are estimated based on memory required to store activations. Current implementation is quite naive and may only be used to compare different models.
   ```
   MEMORY(backward) = MEMORY(forward)
   MEMORY(training) = MEMORY(forward) + MEMORY(backward) = 2 * MEMORY(forward)  
   ```  
 
 While these estimations are not accurate for computing, for instance, batch times, they nevertheless can be used for comparing different models with each other.  

> Assumption: computations in deep NNs are dominated by multiply-adds in dense and convolutional layers. Other operators such as non-linearity, dropout, normalization and others are ignored. The description on this page may be outdated, so, study the source files, in particular, [nns.py](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py).


The class `ModelSummary` computes FLOPs by iterating over neural network layers. A model itself is defined as Keras model. Only those layers that are supported are taken into account, so, make sure your model does not contain non-supported compute intensive layers. The class reports approximate FLOPs count for one input instance (batch size is 1). The following layers are supported (bias is not taken into account):

## Operators  

### Generic operators
`Dropout, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D, AveragePooling1D, AveragePooling2D, AveragePooling3D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten, Reshape, RepeatVector, Lambda, Permute, Activation, LeakyReLU, PReLU, ELU, ThresholdedReLU, Softmax, ReLU, Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate,  UpSampling1D, UpSampling2D, UpSampling3D, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, BatchNormalization`  
These operators do not contribute to FLOPs. Memory is estimated based on shape of the output tensors. 

### Dense  
`Forward pass`  
    1. Matrix-matrix multiply: _Y = X * W_, _FLOPs = W.nrows * W.ncols_  
`Backward pass`  
    1. Matrix-matrix multiply: _dX = dY * W.T_, _FLOPs = W.nrows * W.ncols_
    2. Matrix-matrix multiply: _dW = X.T * dY_, _FLOPs = W.nrows * W.ncols_  
  Both forward and backward passes linearly depend on the batch size. Also, _FLOPs(backward) = 2 * FLOPs(forward)_.

Memory requirements are computed based on the size of an output tensor plus the weight tensor. NNS uses simple strategy to count number of activation tensors. Keras API for Dense layers supports _activation_ parameter. If activation is not linear, NNS does not assume fused implementation. So, number of activation tensors are doubled and so are memory requirements: `MEMORY(Dense) = MEMORY(Activations_Dense) + MEMORY(Activations_Activation) = 2 * MEMORY(Activations_Dense)`. This is also the case for other layers that accept _activation_ parameter such as _Conv1D_ and _Conv2D_. 
  
### Conv1D, Conv2D, Conv2DTranspose
`Forward pass`  
 FLOPs = NumFilters * (Output.H * Output.W) * (Filter.D * Filter.H * Filter.W)  OR, assuming `im2col` implementation [[1](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/), [2](https://arxiv.org/pdf/1410.0759.pdf)]:
    1. Matrix-matrix multiply: Y[K, NPQ] = F[K, CRS] * D[CRS, NPQ], FLOPS = KCRSNPQ, removing (N - batch size), it's the same as expression above.
`Backward pass`  
    1. Matrix-matrix multiple: dX = F.T[CRS, K] * dY[K, NPQ], FLOPs = CRSKNPQ
    2. Matrix-matrix multiple: dW = dY[K, NPQ] * D.T[NPQ, CRS], FLOPs = KNPQCRS
  Both forward and backward passes linearly depend on a batch size. Also, _FLOPs(backward) = 2 * FLOPs(forward)_, same as for `Dense` layer.

Memory requirements are computed based on the size of an output tensor plus weight tensor. If one of these layers is coupled with non-linear layer, memory doubles - same as for _Dense_ layer.

### RNN / Bidirectional layers
Bidirectional models double number of FLOPs. The following cells are supported: `SimpleRNNCell`, `LSTMCell` and `GRUCell`. RNNs use matrix multiply, so forward/backward FLOPs are similar to those for `Dense` layer.  
Also, _FLOPs(LSTM) ~ 4 * FLOPs(RNN)_ and _FLOPs(GRU) ~ 3 * FLOPs(RNN)_
1. `RNN` Two matrix multiplications for each time step: _hidden[t] = x[t]*Wxh + hidden[t-1]*Whh_. Memory estimation algorithm is [here](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L264).
2. `LSTM` Hidden and cell sizes are equal. In total, 4 matrix multiplications with input X and 4 matrix multiplications with hidden state H. Plus a bunch of element wise multiplications, sums and activations that we do not take into account. Memory estimation is [here](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L271).
3. `GRU ` Update/reset/hidden, each has 1 matrix multiply with X and one with H, so in total 3 matrix multiplies with X and 3 with H. Memory estimation is [here](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L290).

### TimeDistributed
Time distributed [layer](https://keras.io/layers/wrappers/) applies a base layer to every temporal slice of an input. The base layer can be any layer described above. Number of FLOPS of the base layer is multiplied by the sequence length. Memory requirements are computed based on memory of a base layer times number of time steps. See [source code](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L211) for more details.

## Examples 
Examples are in [notebooks](./notebooks) folder. Before, install Python virtual enviroment:
```bash
virtualenv -p python3 ./.tf2
source ./.tf2/bin/activate
pip install -r ./requirements.txt
jupyter notebook
```

## License
[Apache License 2.0](./LICENSE.md)
