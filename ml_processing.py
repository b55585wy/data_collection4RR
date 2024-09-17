import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class MLProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.svm_classifier = None
        self.rf_classifier = None
        self.knn_classifier = None

    def preprocess_data(self, X, y=None):
        """
        预处理数据：标准化特征
        """
        X_scaled = self.scaler.fit_transform(X)
        if y is not None:
            return X_scaled, y
        return X_scaled

    def train_svm(self, X, y, kernel='rbf', C=1.0):
        """
        训练SVM分类器
        """
        X_scaled, y = self.preprocess_data(X, y)
        self.svm_classifier = SVC(kernel=kernel, C=C)
        self.svm_classifier.fit(X_scaled, y)

    def predict_svm(self, X):
        """
        使用SVM分类器进行预测
        """
        if self.svm_classifier is None:
            raise ValueError("SVM classifier has not been trained yet.")
        X_scaled = self.preprocess_data(X)
        return self.svm_classifier.predict(X_scaled)

    def train_random_forest(self, X, y, n_estimators=100):
        """
        训练随机森林分类器
        """
        X_scaled, y = self.preprocess_data(X, y)
        self.rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
        self.rf_classifier.fit(X_scaled, y)

    def predict_random_forest(self, X):
        """
        使用随机森林分类器进行预测
        """
        if self.rf_classifier is None:
            raise ValueError("Random Forest classifier has not been trained yet.")
        X_scaled = self.preprocess_data(X)
        return self.rf_classifier.predict(X_scaled)

    def train_knn(self, X, y, n_neighbors=5):
        """
        训练K近邻分类器
        """
        X_scaled, y = self.preprocess_data(X, y)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.knn_classifier.fit(X_scaled, y)

    def predict_knn(self, X):
        """
        使用K近邻分类器进行预测
        """
        if self.knn_classifier is None:
            raise ValueError("KNN classifier has not been trained yet.")
        X_scaled = self.preprocess_data(X)
        return self.knn_classifier.predict(X_scaled)

    def evaluate_model(self, X, y, model_type='svm'):
        """
        评估模型性能
        """
        X_scaled = self.preprocess_data(X)
        if model_type == 'svm':
            y_pred = self.svm_classifier.predict(X_scaled)
        elif model_type == 'rf':
            y_pred = self.rf_classifier.predict(X_scaled)
        elif model_type == 'knn':
            y_pred = self.knn_classifier.predict(X_scaled)
        else:
            raise ValueError("Invalid model type. Choose 'svm', 'rf', or 'knn'.")

        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report

def process_imu_data(data, window_size=100, overlap=50):
    """
    处理IMU数据：将原始数据分割成固定大小的窗口，并提取特征
    """
    features = []
    for i in range(0, len(data) - window_size + 1, window_size - overlap):
        window = data[i:i+window_size]
        window_features = extract_features(window)
        features.append(window_features)
    return np.array(features)

def extract_features(window):
    """
    从数据窗口中提取特征
    """
    features = []
    for axis in range(6):  # 6个轴：x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro
        axis_data = window[:, axis]
        features.extend([
            np.mean(axis_data),
            np.std(axis_data),
            np.max(axis_data),
            np.min(axis_data),
            np.median(axis_data)
        ])
    return features

# 使用示例
if __name__ == "__main__":
    # 假设我们有一些示例数据
    X = np.random.rand(1000, 6)  # 1000个样本，每个样本6个特征（对应6个轴的数据）
    y = np.random.randint(0, 2, 1000)  # 二分类问题

    # 处理数据
    processed_data = process_imu_data(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=0.2, random_state=42)

    # 初始化MLProcessor
    ml_processor = MLProcessor()

    # 训练SVM模型
    ml_processor.train_svm(X_train, y_train)

    # 评估SVM模型
    accuracy, report = ml_processor.evaluate_model(X_test, y_test, model_type='svm')
    print(f"SVM Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # 你可以类似地使用随机森林或K近邻分类器
    # ml_processor.train_random_forest(X_train, y_train)
    # ml_processor.train_knn(X_train, y_train)