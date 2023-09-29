# SR-GAN-Flow
CNN-GAN for super-resolution + denoising of MRI blood flow fields

## Training

Edit and run GAN-trainer.py or trainer.py to train GAN variants or 4DFlowNet respectively.
Architecure settings can be changed in the chosen model's TrainerController.py and SR4DFlowGAN.py
All relative paths assume src/ is the current working directory.

## Prediction

Edit and run GAN-predictor.py or predictor.py to predict using GAN variants or 4DFlowNet respectively.
Architecure settings must match those used during training for the model weights to load successfully.
All relative paths assume src/ is the current working directory.
