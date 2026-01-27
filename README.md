# OCDFE

## One-class deep feature extraction

Code accompanying the paper "One-class deep feature extraction of nonlinear independent components for gearbox anomaly detection" by Xiaoyun Gong, Mengxuan Hao, Chuan Li, Wenliao Du, and Ziqiang Pu (Ready to be submitted for publication).

* Tensorflow 2.0 implementation
* Inspired by Laurent **_et al_**.
[nice.py](https://github.com/bojone/flow/blob/master/nice.py): NICE (Non-linear Independent Components Estimation, [intro]( https://kexue.fm/archives/5776))
* Inspired by Seyfioğlu **_et al_**. DCAE (Deep convolutional autoencoder for radar-based classification of similar aided and unaided human activities)


# Requirements

* Python 3.6.0
* Keras == 2.8.0 
* Tensorflow == 2.8.0

# File description


* `train_ACM+STM`:   The ACM and STM modules in the model OCDFE we built.
* `train_CPM`:   The CPM modules in the model OCDFE we built.
* `sample_ACM+STM`:       The logarithmic probability distribution of independent decoupling feature set is obtained.
* `sample_CPM`:       Obtain a low-dimensional depth feature set.
* `Copod`:          Copula-based anomaly detection.
* `Data_processing`:   Signal segmentation, normalization.
* `Data_splits_config`:train/validation/test splits, random seed settings
* `DeepSVDD`:          Deep anomaly detection model.
* `Ecod`:           Empirical cumulative distribution for anomaly detection.
* `IF`:             Isolation Forest algorithm.
* `Ocsvm`:             One-class SVM for anomaly detection.
* `SVDD`:              SVDD algorithm for anomaly detection.
* `t_SNE_utils`:       t-SNE package for 2-D visualization.

# Implementation details

* The proposed method is implemented in the TensorFlow framework. The OCDFE model was trained using the provided training set
* In the training phase, 32 batches with 100 iterations each were employed. The optimizer of choice was Adam, configured with a learning rate of 0.00001.
* The activation function utilized in the model was the Leaky Rectified Linear Unit (Leaky ReLU).

# Usage

## Model description
| Architecture                                                                                                 | Description |
| -----------                                                                                                  | ----------- |
| ![4](https://github.com/123MHao/OCDFE/assets/102200358/82314519-613d-48cd-b313-eb7030aee171)                 | The experimental implementation of the proposed OCDFE is depicted. Executed using the TensorFlow framework, the method initiated by utilizing ACM and STM to decouple the original vibration signal features, granting individual meanings to each feature. Subsequently, an encoder encompassing CPM functions as a feature extractor, learning features by mapping the data to the inner layers.      |

# Result
|                               |  OCDFE_SUFD   | OCDFE_ZZUli  |
| -----------            |----------- | ----------- |
|  t-SNE     | ![1](https://github.com/123MHao/OCDFE/assets/102200358/ed0c5c32-3db2-48b1-8165-a2190485df65) |![2](https://github.com/123MHao/OCDFE/assets/102200358/97c49f63-9c3d-4891-ae54-a5dc059d3694) |   

# Acknowledgments


This work is supported in part by the National Key Research and Development Pro-gram of China (2023YFB3406104), the National Natural Science Foundation of China (52175080, 52275138), the Key R&D Projects in Henan Province (Grant No. 231111221100, 221111240200), and the Advanced Programs for Overseas Researchers of Henan Province (20221803).
