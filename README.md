# ws-imc-feature-extraction
A self-supervised convolutional autoencoder (CAE) with a ResNet encoder for feature extraction of whole-slide IMC via contrastive learning.




- Randomized order of images, then randomized patch order
    - keeps image patches grouped together to cache the image
    - patches extracted on the fly


- new: converted to saving the preprocessed, padded, resized, and biologically sufficient patches
    - filename: {ws-img barcode/UHNL_ID}_y{y_coord}_x{x_coord}