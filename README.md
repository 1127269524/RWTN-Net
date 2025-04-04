# RWTN-Net

#### Prerequisites
- PyTorch-2.0.1
- Python-3.9

Install the required packages by:
```
pip install -r requirements.txt
```
#### Pretrained models
The pretrain weight is the best.pth file in `weights/` folder
#### Training and Test Details
We train our model using the USCISI dataset, which consists of 100K quality synthetic images. During the training process, USCISI is split into training, validation, and test sets in the ratio of 8:1:1. All input images are resized to 320 × 320. To ensure training stability, we apply normalization based on specific mean and standard deviation values to the input images. For network optimization, we employ the stochastic gradient descent (SGD) optimizer with a momentum of 0.9 and use the binary cross-entropy loss. We conduct training for 100 epochs and the batch size is set to 16. To ensure a reasonable learning rate adjustment, we adopt the ReduceLROnPlateau scheduler. Specifically, the initial learning rate is set to 1e-2, if the model performance does not improve over 30 consecutive epochs, the learning rate is reduced to one-tenth of its current value. This process continues until the learning rate reaches a minimum value of 1e-6. To assess the model's generalization ability, we conduct experiments on additional datasets such as CASIA v2, CoMoFoD, COVERAGE, MICC - F600, and GRIP datasets.
#### Using
##### Training
Train RWTN-Net by running train.py
#### Testing
Place the test image in the root directory and name it as 'test.png'

Run test.py, then you will get the result of RWTN-Net
