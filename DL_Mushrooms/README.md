# Mushroom Classification

- main goal: predict mushroom class with pytorch lightning

- skills learned:
   - explore torch tensors and batches
   - dataset
   - datamodule/dataloader (including stratified train/val/test split before transforms, and options for class balance)
   - data transform and augmentation
   - basic lightning model (ResNet18)
   - run model on either GPU (CUDA) or CPU
   - learning rate tuning
   - TensorBoardLogger 

- most important modules used:
   - pandas, numpy
   - torch, lightning
   - albumentations (transforms)