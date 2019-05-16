# Estimating neural network runtime characteristics


The `nns` (Neural Network Summary) package implements a simple class `ModelSummary` that computes FLOPs requirements for neural networks (for both forward and backward passes). While these estimations may not be accurate to compute, for instance, batch times, FLOPs estimations can be used for comparing different models with each other.

The algorithm follows "standard" approach that only takes into account operators that dominate computations, in particular `matrix multiply` and `convoutional` operations.

> Assumption: computations in deep NNs are dominated by multiply-adds in dense and convolutional layers.
> Other operations such as non-linearity, dropout and normalization layers are ignored.

> The package estimates **multiply-add** operations.

The class `ModelSummary` computes FLOPs by iterating over neural netowk layers. A model itself is defined as Keras model. Only those layers that are supported are taken into account, so, make sure your model does not contain non-supported compute intensive layers. The class reports approximate FLOPs count for one instance (batch size is 1). The following layers are supported (bias is not taken into account):  
### Dense  
`Forward pass`  
    1. Matrix-matrix multiply: Y = X * W, FLOPs = W.nrows * W.ncols
`Backward pass`  
    1. Matrix-matrix multiple: dX = dY * W.T, FLOPs = W.nrows * W.ncols
    2. Matrix-matrix multiple: dW = X.T * dY, FLOPs = W.nrows * W.ncols
  Both forward and backward passes linearly depend on a batch sise. Moreover, _FLOPs(backward) = 2 * FLOPs(forward)_.
  
### Conv2D
`Forward pass`  
 FLOPs = NumFilters * (Output.H * Output.W) * (Filter.D * Filter.H * Filter.W)  OR, assuming `im2col` implementation [[1](https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/), [2](https://arxiv.org/pdf/1410.0759.pdf)]:
    1. Matrix-matrix multiply: Y[K, NPQ] = F[K, CRS] * D[CRS, NPQ], FLOPS = KCRSNPQ, removing (N - batch size), it's the same as expression above.
`Backward pass`  
    1. Matrix-matrix multiple: dX = F.T[CRS, K] * dY[K, NPQ], FLOPs = CRSKNPQ
    2. Matrix-matrix multiple: dW = dY[K, NPQ] * D.T[NPQ, CRS], FLOPs = KNPQCRS
  Both forward and backward passes linearly depend on a batch sise. Moreover, _FLOPs(backward) = 2 * FLOPs(forward)_, same as for `Dense` layer.

### Conv2DTranspose
Same as Conv2D (reversed forward/backward). **This needs to be checked.**

> Current implementation assumes it's a Conv2D layer with inputs/outputs replaced i.e. instead of considering _Y = Conv2DTranspose(X)_ the computations are done for _X = Conv2D(Y)_.

### RNN including Bidirectional models
Bidirectonal models multiples number of FLOPs. The following cells are supported: `SimpleRNNCell`, `LSTMCell` and `GRUCell`. RNNs uses matrix multiply, so forward/backward GLOPs are similar to those for `Dense` layer.  
Moreover, _FLOPs(LSTM) ~ 4 * FLOPs(RNN)_ and _FLOPs(GRU) ~ 3 * FLOPs(RNN)_
1. `RNN` Two matrix multiplications for each time step: _hidden[t] = x[t]*Wxh + hidden[t-1]*Whh_
2. `LSTM` Hidden and cell sizes are equal. In total, 4 matrix multiplications with input X and 4 matrix multiplications with hidden state H. Plus a bunch of element wise multiplications, sums and activations that we do not take into account.
3. `GRU ` Update/resete/hidden  each has 1 matrix multiply with X and one with H, so in total 3 mat multiplies with X and 3 with H.

### TimeDistributed
Time distributed layer supports `Dense`, `Conv2D` and `Conv2DTranspose` layers. Number of FLOPs is multiplied by a sequence length.

# Examples 
Examples are in [notebooks](./notebooks) folder. Before, install Python virtual enviroment:
```bash
virtualenv -p python3 ./.tf2
source ./.tf2/bin/activate
pip install -r ./requirements.txt
jupyter notebook
```