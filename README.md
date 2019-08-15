# GLMHMM

Instructions on using the GLM-HMM codebase:

1. Clone the repository and add the matlab_code directory to your MATLAB path

2. To train a GLM-HMM model, call the function HMMGLMtrain9. This function takes the following form:

```HMMGLMtrain9(symb, emit_w, trans_w, stim, analog_emit_w, analog_symb, outfilename, options)```

For each trial, we assume the following things:
- The trial is of some length T
- There are N possible outputs of which only one is emitted per time point
- There are M possible states the model can enter
- There are K regressors that we are fitting

`symb`: The output symbols that we are trying to fit. This variable should be a (set) of cells. Each cell represents a trial and contains a vector that is of length T. The elements of the vector should be an integer in the set 0, ..., N-1.

`emit_w`: The initial weights that predict the emissions at each time point. This should be a matrix of size (number of states, number of emissions - 1, number of regressors) or (M, N-1, K)

`trans_w`: The initial weights that predict the state transitions at each time point. This should be a matrix of size (number of states, number of states, number of regressors) or (M, M, K)

`stim`: The stimulus at each time point that we are regressing against. Each cell represents a trial that matches up with the corresponding cell for the symb variable. This matrix should be of size {(number of regressors, number of time points)}, or {(K, T)}

`analog_emit_w` and `analog_symb`: Optional. These are for a (somewhat untested) version with gaussian outputs. Use this at your own risk!

`outfilename`: Optional. If you want to save each iteration of the output, pass this as a string. Leave blank ('').

`options`: Optional. Many fitting options to be documented soon.


## Some notes on running these models
Note that if you set the number of states to be 1 this is equivalent to fitting a standard multinomial GLM and if you fit this with only constant regressors (eg, the `stim` parameter is all 1s) this is equivalent to an HMM.

The more variables you include as regressors, the more data that you need to fit on. These models are really, really good at overfitting on your data! Be sure to regularize (otherwise you will heavily overfit), but note that it is adviseable to regularize the transition and emission GLMs with different values if you believe the operate on different timescales.

Beware false minima: it often seems like the model can be stuck at a minima during training. However, these are often just plateaus that require many iterations for the model to train against to find the true minima.

## TODO

Add tutorial, add helper function to format data for fitting GLM HMM.