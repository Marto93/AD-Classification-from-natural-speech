# AD-Classification-from-natural-speech
In this work we investigate the impact that convolutional neural networks (CNNs) and transfer learning application have in
discriminating Alzheimer Dementia (AD) from natural speech. In order to answer these question two different CNN architectures:
VGG-16 and InceptionV3 were selected and trained on two different datasets containing spectrograms derived from segments of 
natural speech of affected and healthy people. The architectures were trained in 3 different conditions for each dataset 
resulting in twelve final models. In the first condition models were trained without the application of transfer learning.
In the second condition the pre-saved weight from ImageNet dataset were applied but leaving the convolutional architecture
frozen. In the last condition the convolutional architecture was divided, the upper half remained frozen where the second 
was made trainable. The poor performances obtained using CNNs seem to show that, with so low data, they are not the best 
architectures to identify AD. Furthermore, even though the application of transfer learning significantly increased the 
performances of the models, showing the power of this technique for future works, this was still not enough to beat the 
level of the state of art. 
