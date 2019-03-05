# Vehicle Classifier Optimizations

'inference/' contains the OpenCL implementation for the separable convolution version of the network. The program does a performance measurement for each value of K from 1 to 16.

'tf/' contains the tensorflow scripts for finding the separable kernels for separable convolution and the script for pruning, retraining and fine tuning the sparse network. The weights used are in 'weights/'.

For the Binarized vehicle classifier, refer to BinCNN repository: github.com/Valentin4869/BinCNN
