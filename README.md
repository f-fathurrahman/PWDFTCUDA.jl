# PWDFTCUDA.jl

An attempt to make a CUDA version of `PWDFT.jl`.

Not all features of `PWDFT.jl` are implemented. Most notably
use of k-points and symmetries are not yet implemented.

This package was first written during the time when `CUDA.jl` was
fragmented into various separate package such as `CuArrays`
and `CUDAnative`, so there are possibly several unnecessary codes left.

No attempt to tune the execution configuration parameters of CUDA
kernels.

Several examples can be found in the directory `test` (should be
renamed `examples`).

I also have tried to run `PWDFTCUDA.jl` on Google Colab.
Please see the notebook `test/Test_PWDFT.ipynb`.
