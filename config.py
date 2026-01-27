"""
配置文件，定义路径、参数和实验配置
"""

import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# 创建目录
for dir_path in [DATASET_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据集配置
DATASETS = {
    'SU_gearbox': {
        'name': 'Southeast University Benchmark Gearbox',
        'classes': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
        'normal_class': 'C0',
        'train_samples': 700,
        'test_samples_per_class': 50
    },
    'UC_gearbox': {
        'name': 'University of Connecticut Benchmark Gearbox',
        'classes': ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8'],
        'normal_class': 'P0',
        'train_samples': 700,
        'test_samples_per_class': 50
    },
    'QPZZ_II': {
        'name': 'QPZZ-II Gearbox Fault Diagnosis Setup',
        'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'normal_class': '0',
        'train_samples': 700,
        'test_samples_per_class': 50
    }
}

# 特征提取器配置
FEATURE_EXTRACTORS = ['OTD', 'SR', 'OCN', 'SIEF', 'OCDFE']

# 异常检测器配置
ANOMALY_DETECTORS = {
    'OCSVM': {
        'nu': 0.05,
        'kernel': 'rbf',
        'gamma': 'auto'
    },
    'IF': {
        'n_estimators': 100,
        'contamination': 0.1
    },
    'SVDD': {
        'C': 0.8,
        'gamma': 0.3,
        'kernel': 'rbf'
    },
    'DeepSVDD': {
        'use_ae': False,
        'epochs': 100,
        'contamination': 0.01,
        'l2_regularizer': 0.1
    },
    'COPOD': {},
    'ECOD': {}
}

# OCDFE模型参数（根据论文Table III）
OCDFE_PARAMS = {
    'input_shape': (1024,),
    'acm_layers': 4,
    'acm_params': 5312168,
    'stm_params': 1024,
    'cpm_layers': {
        'conv1': {'filters': 32, 'kernel_size': 3},
        'conv2': {'filters': 16, 'kernel_size': 3},
        'conv3': {'filters': 8, 'kernel_size': 3},
        'conv4': {'filters': 4, 'kernel_size': 3},
        'conv5': {'filters': 2, 'kernel_size': 3},
        'dense': {'units': 128}
    }
}