import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(filename)
    return data['X'], data['y']


def train_svm(X: np.ndarray, y: np.ndarray) -> SVC:
    model=SVC(random_state=42)
    model.fit(X,y)
    return model
    #TODO


def apply_pca(X: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca=PCA(n_components=10,random_state=42)
    X_reduced=pca.fit_transform(X)
    return X_reduced,pca
    #TODO



def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> float:
    #TODO
    y_pred = model.predict(X)
    return accuracy_score(y,y_pred)


def main():
    X, y = load_data_file("dataset.npz")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    svm_original = train_svm(X_train, y_train)
    acc_original = evaluate_model(svm_original, X_test, y_test)

    X_train_pca, pca_model = apply_pca(X_train)
    X_test_pca = pca_model.transform(X_test)
    svm_pca = train_svm(X_train_pca, y_train)
    acc_pca = evaluate_model(svm_pca, X_test_pca, y_test)

    print(f"原始特征 SVM 测试集准确率: {acc_original:.4f}")
    print(f"PCA 降维后 SVM 测试集准确率: {acc_pca:.4f}")
    print(f"性能提升: {acc_pca - acc_original:.4f}")


if __name__ == "__main__":
    
    main()