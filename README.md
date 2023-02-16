# Large deformation measurement method of speckle image based on deep learning

In this paper, a displacement field measurement method for speckle image with complex large deformation is proposed. This method uses convolutional block attention module and depthwise separable convolution module to improve an existing convolutional neural network model for measuring large deformation displacement fields. In order to train the model, a dataset containing multiple types of speckle images and complex large deformation displacement field is constructed and a new loss function is proposed. This method is compared with the traditional method and the latest deep learning method on the self-built dataset and the open dataset respectively. The results show that the method proposed in this paper achieves the highest average accuracy with the minimum number of model parameters, and the displacement field measurement speed is much faster than the traditional method, which can meet the actual real-time measurement requirements of large deformation displacement field.



## Dependencies

DICNet is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu 20.04, please install PyTorch first following the official instruction.

- Python 3.8
- PyTorch
- Torchvision
- Pillow 
- numpy
- CUDA

## Datasets

training dataset, validation dataset and test dataset.

https://pan.baidu.com/s/1KzC9g_GIkvMnGFumDYGyBA?pwd=fd5x     password：fd5x 

