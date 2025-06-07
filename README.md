# ws-imc-feature-extraction
A self-supervised ResNet CNN trained with SimCLR for feature extraction of whole-slide IMC via contrastive learning.


DevNotes:
- picked ResNet-18 as it seemed the most common for bioimaging
- initialized the new first convolutional layer to take the x amount of markers
    - chose Kaiming initialzation as the method (for ReLU activation)
- specified average pooling