"""
实现6种异常检测器：OCSVM, IF, SVDD, DeepSVDD, COPOD, ECOD
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from pyod.models.svdd import SVDD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(self, detector_type='OCSVM', **kwargs):
        self.detector_type = detector_type
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        self._initialize_detector()

    def _initialize_detector(self):
        """初始化检测器"""
        if self.detector_type == 'OCSVM':
            nu = self.kwargs.get('nu', 0.05)
            kernel = self.kwargs.get('kernel', 'rbf')
            gamma = self.kwargs.get('gamma', 'auto')
            self.detector = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

        elif self.detector_type == 'IF':
            n_estimators = self.kwargs.get('n_estimators', 100)
            contamination = self.kwargs.get('contamination', 0.1)
            self.detector = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )

        elif self.detector_type == 'SVDD':
            C = self.kwargs.get('C', 0.8)
            gamma = self.kwargs.get('gamma', 0.3)
            kernel = self.kwargs.get('kernel', 'rbf')
            self.detector = SVDD(C=C, gamma=gamma, kernel=kernel)

        elif self.detector_type == 'DeepSVDD':
            use_ae = self.kwargs.get('use_ae', False)
            epochs = self.kwargs.get('epochs', 100)
            contamination = self.kwargs.get('contamination', 0.01)
            l2_regularizer = self.kwargs.get('l2_regularizer', 0.1)
            self.detector = DeepSVDD(
                use_ae=use_ae,
                epochs=epochs,
                contamination=contamination,
                l2_regularizer=l2_regularizer,
                random_state=42,
                verbose=0
            )

        elif self.detector_type == 'COPOD':
            self.detector = COPOD()

        elif self.detector_type == 'ECOD':
            self.detector = ECOD()

        else:
            raise ValueError(f"Unknown detector type: {self.detector_type}")

    def fit(self, X_train):
        """训练检测器"""
        X_scaled = self.scaler.fit_transform(X_train)

        if self.detector_type == 'OCSVM':
            self.detector.fit(X_scaled)
        else:
            # pyod模型需要标签，对于一分类，所有训练样本都是正常的（标签=0）
            y_train = np.zeros(len(X_scaled))
            self.detector.fit(X_scaled, y_train)

    def predict(self, X_test, is_normal_class=True):
        """预测样本是否为异常"""
        X_scaled = self.scaler.transform(X_test)

        if self.detector_type == 'OCSVM':
            # OCSVM: 1表示正常，-1表示异常
            predictions = self.detector.predict(X_scaled)
            if is_normal_class:
                # 对于正常类，我们希望预测为1
                anomalies = predictions == -1
            else:
                # 对于异常类，我们希望预测为-1
                anomalies = predictions == 1

        elif self.detector_type == 'IF':
            # IF: 1表示正常，-1表示异常
            predictions = self.detector.predict(X_scaled)
            if is_normal_class:
                anomalies = predictions == -1
            else:
                anomalies = predictions == 1

        else:
            # pyod模型: 0表示正常，1表示异常
            predictions = self.detector.predict(X_scaled)
            if is_normal_class:
                anomalies = predictions == 1
            else:
                anomalies = predictions == 0

        return anomalies

    def compute_accuracy(self, X_test, is_normal_class=True):
        """计算准确率"""
        anomalies = self.predict(X_test, is_normal_class)
        if is_normal_class:
            # 正常类：错误预测为异常的数量
            n_errors = np.sum(anomalies)
        else:
            # 异常类：错误预测为正常的数量
            n_errors = np.sum(anomalies)

        accuracy = 1 - (n_errors / len(X_test))
        return accuracy, n_errors