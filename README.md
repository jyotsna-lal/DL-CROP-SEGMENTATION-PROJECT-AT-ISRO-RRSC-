> 📌 **License:** View-only. All rights reserved. This code may not be reused, copied, or modified without permission.


# DL-CROP-SEGMENTATION-PROJECT-AT-ISRO-RRSC-
Deep learning-based crop segmentation using U-Net and Masked Attention Transformers. U-Net handles initial segmentation of various crop types, while transformers refine predictions. Trained and tested on real-world satellite imagery, achieving high accuracy in practical scenarios.

This repository implements a complete deep learning pipeline using a custom U-Net architecture to perform multi-class semantic segmentation of crops from high-resolution satellite .tif images. The segmentation targets include:
Class 0 → Background
Class 1 → Pomegranate
Class 2 → Date Palm (remapped from class label 4 in masks)

The solution includes data preprocessing, custom loss functions, U-Net training, visual evaluation, and batch prediction on new images — all optimized for robust inference on real-world agricultural data.

Dataset Structure
The dataset consists of high-resolution .tif satellite images and their corresponding mask files. File naming convention:

Input images: image_<id>.tif
Masks: mask_<id>.tif

All masks use labels:

0 for background
1 for pomegranate
4 for date palm → remapped to 2 for model compatibility

Preprocessing Pipeline
Before training, multiple preprocessing steps are performed to ensure data quality:
Valid TIFF Check: Filters out non-TIFFs and corrupt .aux.xml files using tifffile.TiffFile validation.
ID Matching: Extracts unique IDs from filenames and matches image-mask pairs.
Label Remapping: Converts label 4 (date palm) to 2 so that final mask classes are [0, 1, 2].
One-Hot Encoding: Converts remapped masks into shape (256, 256, 3) for multi-class segmentation.
Train–Validation Split: Stratified split of matched IDs into 80% training and 20% validation.

Model Architecture — Custom U-Net
The model is built from scratch using Keras:

Encoder (Downsampling): 4 blocks with Conv2D → ReLU → MaxPooling2D

Bottleneck: Deepest layer with 1024 filters

Decoder (Upsampling): 4 blocks with Conv2DTranspose → Skip Connections → Conv2D

Output Layer: Conv2D with softmax for multi-class output

Input size: (256, 256, 3)
Output size: (256, 256, 3) with per-pixel class probabilities

Custom Loss & Metric
For better class balance and boundary awareness, a combined loss function is used:
Categorical Crossentropy (CCE): For multi-class pixel-wise loss
Dice Loss: Encourages overlap between predicted and ground truth masks

loss = categorical_crossentropy + (1 - multiclass_dice_coef)

Also logs:
Accuracy
Multiclass Dice Coefficient (as evaluation metric)

Data Generator
A custom DataGenerator class is used for memory-efficient training:
Loads only batch-sized data into memory
Applies:
Image resizing to 256×256
Grayscale to RGB conversion (if needed)
One-hot encoding of masks
NaN/Inf checks to prevent crashes
Shuffles data at the end of each epoch

Training
Training is done using model.fit() with the following:
Batch Size: 16
Epochs: 100
Optimizer: Adam (LR = 1e-4)
Callbacks:
ModelCheckpoint (saves best model)
EarlyStopping (patience = 25)
ReduceLROnPlateau (patience = 5)
The best model is saved as:

Evaluation
The model is evaluated on the 20% validation split:
Computes:
Final loss
Accuracy
Dice coefficient
Prints summary for easy inspection

Prediction & Visualization on Validation Set
To visually verify model performance:
Loops through validation generator
For each image:
Predicts segmentation mask using model.predict()
Displays:
Original RGB image (normalized)
Ground truth mask
Predicted mask using argmax and color map
Visualizes up to 200 examples in interactive plots using matplotlib

Final Prediction on Unseen Test Dataset
For full inference on a new dataset (e.g., Chips_subset_9Feb_2024_for_test):
Loads all .tif files in the test folder
Normalizes each image (band-wise)
Performs batch predictions (batch size = 16)
Converts softmax output to per-pixel class (argmax)
Optionally supports confidence thresholding using np.max(preds, axis=-1)
Saves predicted masks as .png images:
Class 0 → 0

Class 1 → 85

Class 2 → 170

Displays predictions in batches of 5 with side-by-side plots

Reports:

Per-image class distribution

Total class-wise counts across the dataset

Outputs
Saved model weights: best_unet_model_1/07/25.h5
Evaluation metrics: Accuracy, Dice
Interactive plots: 3-panel visualizations per sample



**Masked Attention Transformer + DUC for Semantic Segmentation of Crops from Satellite Imagery**
This repository implements a complete end-to-end semantic segmentation framework using a Transformer-based deep learning architecture with PixelShuffle (Dense Upsampling Convolution - DUC) for full-resolution multi-class mask generation. The model is trained and evaluated on real-world, high-resolution .tif satellite images, with specific focus on segmenting different crop types: background, pomegranate, and date palm.

Dataset and Preprocessing
The dataset consists of two primary directories:
augmented_images6/: Contains satellite .tif images in the format image_<id>.tif
augmented_masks6/: Contains corresponding segmentation masks in the format mask_<id>.tif
The masks use the following raw label values:

0 → Background

1 → Pomegranate

4 → Date Palm

During preprocessing, label 4 is remapped to 2 to form a consistent 3-class segmentation setup: [0, 1, 2]. Only valid .tif files are retained, while auxiliary and corrupt files are excluded using a validation routine that attempts to load each file via TiffFile. The matched image-mask pairs are split into training and validation sets (80:20) using train_test_split.

Each image and mask is resized to a uniform resolution of 256×256. All masks are one-hot encoded into shape (256, 256, 3) before training. Input images are normalized band-wise using their maximum pixel intensity to ensure stability during model inference.

Model Architecture: Transformer with DUC
The implemented model consists of the following components:

Patch Embedding:
The input image of shape (256, 256, 3) is projected into a sequence of tokens using a Conv2D layer with kernel size and stride equal to the patch size (16×16). This produces a flattened sequence of embeddings with dimension d_model = 256.

Positional Embedding:
Learnable positional encodings are added to the patch embeddings to retain spatial context.

Transformer Blocks:
The encoded sequence passes through a stack of 4 identical transformer blocks. Each block consists of:

Layer normalization

Multi-head self-attention (8 heads)

MLP with GELU activation

Residual connections

Reshape and Convolution:
The sequence is reshaped back to a spatial feature map of shape (16, 16, 256). Two convolutional layers refine the feature representation.

DUC Module with PixelShuffle2D:
A final convolutional layer produces a feature map with N_CLASSES × (4²) channels, which is then upsampled using PixelShuffle (factor 4), followed by bilinear upsampling to restore full input resolution (256, 256).

Output Layer:
A softmax layer generates per-pixel class probabilities across three classes.

Loss Function and Metrics
The model uses a compound loss function to balance class accuracy and spatial segmentation quality:

Combined Loss = Categorical Crossentropy + (1 - Multi-class Dice Coefficient)

The Dice coefficient is computed per batch using flattened vectors and smooth regularization to handle zero denominators. The model tracks two metrics during training: categorical accuracy and the Dice coefficient.

Training Procedure
Training is conducted using the Keras fit API with the following configuration:

Optimizer: Adam (learning_rate = 1e-4)

Batch size: 16

Epochs: 100

Callbacks:

ModelCheckpoint: Saves the model with the best validation accuracy

EarlyStopping: Stops training if no improvement in validation accuracy for 15 epochs

ReduceLROnPlateau: Reduces learning rate by a factor of 0.1 after 5 epochs of no validation loss improvement

The final model is saved as final_trained_model.h5.

Evaluation
Post-training, the model is evaluated on the validation set using the same data generator. It computes:

Total loss (combined)

Categorical accuracy

Multi-class Dice coefficient

These metrics provide a comprehensive assessment of segmentation quality and class-level accuracy on unseen validation data.

Inference on Unseen Test Images
A separate script performs inference on high-resolution .tif test images located in a specified folder (e.g., Chips_subset_9Feb_2024_for_test). The steps are as follows:

Band Selection:
From multi-band .tif images, only Bands 2, 3, and 4 are used as RGB input. If the image is grayscale, it is broadcasted across three channels.

Normalization:
Images are normalized using the global maximum pixel value to bring values into the [0, 1] range.

Model Prediction:
The trained model predicts softmax class probabilities for each pixel. The final mask is obtained by taking the argmax across the channel dimension, producing class-wise masks of shape (256, 256).

Saving Outputs:
The predicted masks are saved in grayscale .tif format with filenames matching their corresponding inputs. These are stored in the predicted_masks_output/ directory.

Visualization:
For each test image, a side-by-side plot is generated showing:

The normalized RGB image

The predicted segmentation mask using a Jet colormap

Output Format
Each predicted mask is a grayscale .tif file with pixel values:

0 for background

1 for pomegranate

2 for date palm

The masks are suitable for downstream processing or overlay with geographic metadata if needed.


