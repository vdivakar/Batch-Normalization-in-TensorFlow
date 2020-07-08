# Batch-Normalization-in-TensorFlow
Experimental code setup to demonstrate how to correctly use batch normalization in TensorFlow using tf.nn.batch_normalization()


__Usage:__ python BN_nn.py<br>
__Requiremens:__ TF1.x, OpenCV <br>

Network is a fully convolutional 2 layer network with a skip connection.<br>
Model input: 512 x 512 colored images (rgb 3-channeled).<br>
Model output: A [512,512,3] tensor which can be saved as 512x512 sized image for visualization.
![Model](https://github.com/vdivakar/Batch-Normalization-in-TensorFlow/blob/master/network_image.png)<br>

### How to use tf.nn.batch_normalization()<br>
__In detail:__ checkout this blog post:
<br>__In short:__ add control_dependencies on moving average & moving variance to update them during Training time.



<br>This repository also contains a dummy dataset of 5 training image pairs & 1 test image. So, you are ready to run it directly.<br>
<br>Things to try:
<br>Show dummy dataset->
Images were taken from here: [Link](https://homepages.cae.wisc.edu/~ece533/images/) <br>
```
Batch-Normalization-in-TensorFlow/dataset
├── test_input
│   └── sails.png
├── train_input
│   ├── baboon.png
│   ├── fruits.png
│   ├── lena.png
│   └── peppers.png
└── train_label
    ├── baboon.png
    ├── fruits.png
    ├── lena.png
    └── peppers.png
```
