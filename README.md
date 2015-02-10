Deep Networks on Cifar10/Cifar100 using Torch
=============================================
This is a complete training example for Cifar10/100 tasks (http://www.cs.toronto.edu/~kriz/cifar.html)

##Data
You can get the needed data using @soumith's repo: https://github.com/soumith/cifar.torch.git

##Dependencies
Requires "eladtools" (https://github.com/eladhoffer/eladtools.git) for DataProvider class and optimizer.

##Training
You can start training using:
```lua
th Main.lua -dataset Cifar10 -network Cifar10_Model -LR 0.1
```
##Additional flags
|Flag             | Default Value        |Description
|:----------------|:--------------------:|:----------------------------------------------
|modelsFolder     |  ./Models/           | Models Folder
|network          |  Model.lua           | Model file - must return valid network.
|LR               |  0.1                 | learning rate
|LRDecay          |  0                   | learning rate decay (in # samples
|weightDecay      |  1e-4                | L2 penalty on the weights
|momentum         |  0.9                 | momentum
|batchSize        |  128                 | batch size
|optimization     |  sgd                 | optimization method
|epoch            |  -1                  | number of epochs to train (-1 for unbounded)
|threads          |  8                   | number of threads
|type             |  cuda                | float or cuda
|devid            |  1                   | device ID (if using CUDA)
|load             |  none                |  load existing net weights
|save             |  time-identifier     | save directory
|dataset          |  Cifar10             | Dataset - Cifar10 or Cifar100
|whiten           |  false               | whiten data
|augment          |  false               | Augment training data
|preProcDir       |  ./PreProcData/      | Data for pre-processing (means,Pinv,P)
