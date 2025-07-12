# Context-Sensing-and-Dual-Attention-Network
A module that designed in my undergraduate thesis, for the tasks about scene classification of remote sensing image.
This model has been modified and optimized based on the scene classification model designed by my senior Li-zq. Experimental data shows that the changes I made can improve the accuracy of the classification.

Here are concise introduction of some crucial profiles.

split.py 

In this module, datasets are splited into three parts, including train, verify, and test, at any ratio you need.

module.py 

Code in module.py shows the main modules designed to use in the whole model. The general idea of these modules is to integrate the multi-scale features of the backbone network, to enhance the multi-scale information of the features.
The upsample and downsample functions in the Pyramid model are used firstly. During forward propagation, high-resolution feature maps and low-resolution feature maps are upsampled and downsampled, respectively, followed by feature fusion to generate a set of feature maps with the same resolution but containing multi-scale information.
Then a V-Net network with dilated convolution is introduced to inplement a series of convolution and upsample operation, recovering the resolution of feature maps step by step.
The deep context sensing module consists of two identical submodules.
Innovatively, a dual attention network is introduced in the model, including a spatial attention module and a grouped channel attention module.

CACRWNet.py
In this module, you can find how this model operates in a specific process sequence.

Grad-CAM.py
Another approach to generate a heat map, compare to what showed in utils.py. This is just my attempt, you can choose which you like.
