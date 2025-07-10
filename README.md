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



MASKED ATTENTION TRANSFORMER

