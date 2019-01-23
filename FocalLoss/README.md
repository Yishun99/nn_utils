# FocalLoss modules at the Lua and C or CUDA level
Corresponding to "Focal Loss for Dense Object Detection".


## Lua level
Just copy *FocalLoss.lua* to your project.

**Note**:
- The naive implementation may cause **memory leak**, which is a known bug(or defect) of torch. But you can collect garbage by yourself.
- Memory consumption is higher than C/CUDA level countpart.(The memory optimization of torch is not very well. And I have tried my best to use inplace operator as possible.)



## C and CUDA level
**Re-compile torch is needed.**

### Installation


1. Assuming torch installed at `~/torch`. If you have compiled torch, run `~/torch/clean.sh`.

2. There are 4 **source files** and 4 **registries**. You need **copy** or **modify** corresponding files.


    Current folder structure:
    ```shell
    $ tree
    .
    ├── cunn
    │   └── lib
    │       └── THCUNN
    │           ├── FLCriterion.cu
    │           └── generic
    │               ├── FLCriterion.cu
    │               └── THCUNN.h
    └── nn
        ├── FLCriterion.lua
        ├── lib
        │   └── THNN
        │       ├── generic
        │       │   ├── FLCriterion.c
        │       │   └── THNN.h
        │       └── init.c
        └── THNN_h.lua
    ```

    `cunn` and `nn` all under `~/torch/extra/`.


3. In file `~/torch/install.sh`, comment out  `git submodule update --init --recursive`  at 50th line.

4. Run `~/torch/install.sh`.



### Usage

**Use the CUDA version whenever possible.**

`nn.FLCriterion(alpha, gamma, sizeAverage)`


Parameters:
- `alpha` (double)
- `gamma` (double)
- `sizeAverage` (bool) (option, default `true`)


```lua
--CPU:
require 'nn'
criterion = nn.FLCriterion(0.25, 2)
...

--GPU:
require 'nn'
require 'cunn'
criterion = nn.FLCriterion(0.25, 2):cuda()
...
```

### Test

You can test **loss** and **gradient** of *FLCriterion* on `demo_torch_loss.ipynb`. For comparision, python counterpart also provided.

