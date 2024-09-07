# Improvements and Integration of HEOA and ESOA

Based on the Human Evolutionary Optimization Algorithm (HEOA) and the Egret Swarm Optimization Algorithm (ESOA), we respectively made improvements using Cauchy mutation and Gaussian mutation to enhance the search efficiency and accuracy of the algorithms. The improved HEOA and ESOA were then used to optimize the hyperparameters (Dropout and learning-rate) of the E-HEOA model. Subsequently, an E-HEOA model was trained separately on the CIKM AnalytuCup 2017 Dataset. Finally, the E-HEOA models trained by HEOA and ESOA were combined based on their weights.
# Dataset Overview

The HKO-7 dataset is a collection of meteorological data from the Hong Kong region, including several radar images taken every 6 minutes. For this study, we selected 10 days' worth of data, totaling 2,400 radar images, to train the E-HEOA model.
# Sample dataset

![Radar Image](Sample%20Dataset/RAD090627151200.png)
It contains examples of the dataset
# Code Files

### train.py
Model training file

### HEOA.py
Human Evolutionary Optimization Algorithm file

### ESOA.py
Egret Swarm Optimization Algorithm file

### E_HEOA_model.py
File containing the E_HEOA_model
# Environment configuration

Use Python 3.9 environment with Anaconda. Detailed configuration is as follows:
| Library       | Version |
|---------------|---------|
| TensorFlow    | 2.10.0  |
| Keras         | 2.10.0  |
| scikit-learn  | 1.0.2   |
| NumPy         | 1.21.5  |

This experiment was conducted in a CPU environment with an ADM Ryzen 7 5800HS Creator Edition, equipped with 40GB of RAM and an NVIDIA GeForce MX450 GPU.
