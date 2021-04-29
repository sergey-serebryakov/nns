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
These operators do not contribute to FLOPs. Memory is estimated based on shape of the output tensors. List of operators: `Dropout`, `GlobalMaxPooling1D`, `GlobalMaxPooling2D`, `GlobalMaxPooling3D`, `GlobalAveragePooling1D`, `GlobalAveragePooling2D`, `GlobalAveragePooling3D`, `AveragePooling1D`, `AveragePooling2D`, `AveragePooling3D`, `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `Flatten`, `Reshape`, `RepeatVector`, `Lambda`, `Permute`, `Activation`, `LeakyReLU`, `PReLU`, `ELU`, `ThresholdedReLU`, `Softmax`, `ReLU`, `Add`, `Subtract`, `Multiply`, `Average`, `Maximum`, `Minimum`, `Concatenate`, `UpSampling1D`, `UpSampling2D`, `UpSampling3D`, `ZeroPadding1D`, `ZeroPadding2D`, `ZeroPadding3D`, `BatchNormalization`.  


### Dense  
- `X` is the rank 2 input tensor.
- `Y` is the rank 2 output tensor.
- `W` is the rank 2 weight tensor.
#### Forward pass  
1. Matrix-matrix multiply: _Y = X * W_, _FLOPs = W.nrows * W.ncols_

#### Backward pass  
1. Matrix-matrix multiply: _dX = dY * W.T_, _FLOPs = W.nrows * W.ncols_  
2. Matrix-matrix multiply: _dW = X.T * dY_, _FLOPs = W.nrows * W.ncols_    

Both forward and backward passes linearly depend on the batch size. Also (see expressions for backward pass above): `FLOPs(backward) = 2 * FLOPs(forward)`.
Memory requirements are computed based on the size of an output tensor plus the weight tensor. NNS uses simple strategy to count number of activation tensors. Keras API for Dense layers supports _activation_ parameter. If activation is not linear, NNS does not assume fused implementation. So, number of activation tensors are doubled and so are memory requirements: `MEMORY(Dense) = MEMORY(Activations_Dense) + MEMORY(Activations_Activation) = 2 * MEMORY(Activations_Dense)`. This is also the case for other layers that accept _activation_ parameter such as _Conv1D_ and _Conv2D_. 
  
### Conv1D, Conv2D, Conv2DTranspose
We can think of Conv2D operation as a bunch of dot products between filters and elements of input feature map.
- `Filter` is a rank 3 tensor with filter weights. Shape is `[Depth, Height, Width]` where `Depth` equals to depth of input feature map, and `[Height, Width]` is the filter's receptive field.
- `Output` is a rank 2 output tensor that is the convolution result of input feature map with **one** filter. Output's spatial shape is `[Height, Weight]`.
- `NumFilters` is the number of filters. In a more common problem definition, the output feature map is a rank 3 tensor of the following shape: `[NumFilters, Height, Weight]`.

Given the above definitions, we have:
```shell
FLOPs = NumFilters * (Output.H * Output.W) * (Filter.D * Filter.H * Filter.W)
```
which is equal to a product between number of elements in the output feature map and number of FLOPs for one element. To simplify computations (primarily, for backward pass), the NNS assumes `im2col` implementation [[1](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/), [2](https://arxiv.org/pdf/1410.0759.pdf)]. Below (forward/backward passes) we use notation from [2](https://arxiv.org/pdf/1410.0759.pdf):
- `K` is the number of output feature maps (number of filters).
- `N` is the batch size. It's assumed that total FLOPs are proportional to batch size, so we can safely assume it equals to 1.
- `P` is the height of an output feature map.
- `Q` is the width of an output feature map.
- `C` is the number of input feature maps (depth of input rank 3 tensor, same as `Filter.Depth`).
- `R` is the filter height (`Filter.Height`).
- `S` is the filter width (`Filter.Width`).


- `Y` is the output feature map (`Output`).
- `F` is the filter tensor (`Filter`).
- `D` is the input data (`X`)

#### Forward pass  
1. Matrix-matrix multiply: _Y[K, NPQ] = F[K, CRS] * D[CRS, NPQ]_, _FLOPS = KCRSNPQ_. If we remove _N_ from this equation (_N_ is the batch size), then this expression equals exactly to the one presented above (the one based on computing FLOPs by counting number of dot products).

#### Backward pass  
1. Matrix-matrix multiple: _dX = F.T[CRS, K] * dY[K, NPQ]_, _FLOPs = CRSKNPQ_
2. Matrix-matrix multiple: _dW = dY[K, NPQ] * D.T[NPQ, CRS]_, _FLOPs = KNPQCRS = CRSKNPQ_

Both forward and backward passes linearly depend on a batch size. Also `FLOPs(backward) = 2 * FLOPs(forward)`.
Memory requirements are computed based on the size of an output tensor plus weight tensor. If one of these layers is coupled with non-linear layer, memory doubles - same as for _Dense_ layer.

### RNN / Bidirectional layers
Bidirectional models double number of FLOPs. The following cells are supported: `SimpleRNNCell`, `LSTMCell` and `GRUCell`. RNNs use matrix multiply, so forward/backward FLOPs are similar to those for `Dense` layer.  
Also: `FLOPs(LSTM) ~ 4 * FLOPs(RNN)` and `FLOPs(GRU) ~ 3 * FLOPs(RNN)`.
1. `RNN` Two matrix multiplications for each time step: _hidden[t] = x[t]*Wxh + hidden[t-1]*Whh_. Memory estimation algorithm is [here](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L264).
2. `LSTM` Hidden and cell sizes are equal. In total, 4 matrix multiplications with input X and 4 matrix multiplications with hidden state H. Plus a bunch of element wise multiplications, sums and activations that we do not take into account. Memory estimation is [here](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L271).
3. `GRU ` Update/reset/hidden, each has 1 matrix multiply with X and one with H, so in total 3 matrix multiplies with X and 3 with H. Memory estimation is [here](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L290).

### TimeDistributed
Time distributed [layer](https://keras.io/layers/wrappers/) applies a base layer to every temporal slice of an input. The base layer can be any layer described above. Number of FLOPS of the base layer is multiplied by the sequence length. Memory requirements are computed based on memory of a base layer times number of time steps. See [source code](https://github.com/sergey-serebryakov/nns/blob/master/python/nns/nns.py#L211) for more details.

## Examples 
Examples are in [notebooks](./notebooks) folder. Before, install Python virtual environment:
```bash
virtualenv -p python3 ./.tf2
source ./.tf2/bin/activate
pip install -r ./requirements.txt
jupyter notebook
```

## License
[Apache License 2.0](./LICENSE.md)


## Questions?
[Contact me](serebryakov.sergey@gmail.com).

## References
1. [Convnet: Implementing Convolution Layer with Numpy](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/)
2. [cuDNN: Efficient Primitives for Deep Learning](https://arxiv.org/pdf/1410.0759.pdf)
3. [Memory usage and computational considerations](http://imatge-upc.github.io/telecombcn-2016-dlcv/slides/D2L1-memory.pdf)