'''
对ORL人脸数据集进行K-means聚类
'''
import os
import numpy as np
from glob import glob
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def load_orl_dataset(data_dir):
    """
    从指定目录加载ORL数据集。

    参数
    ----------
    data_dir : str
        包含ORL数据集图片的目录。

    返回
    -------
    X : np.ndarray
        数据集，每一行为一个拉平成的一张图片。
    y : np.ndarray
        每张图片对应的标签。
    """
    image_files = glob(os.path.join(data_dir, "*", "*.jpg"))
    images = []
    labels = []

    for file in image_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img.flatten())
            label = os.path.basename(os.path.dirname(file))
            label = label.lstrip("s")  # 去除标签中的's'
            label = int(label)
            labels.append(int(label))

    X = np.array(images)
    y = np.array(labels)

    return X, y

def calculate_clustering_accuracy(y_true, y_pred):
    """
    通过比较真实标签和聚类预测标签，计算聚类准确率。

    参数
    ----------
    y_true : np.ndarray
        真实标签。
    y_pred : np.ndarray
        聚类预测标签。

    返回
    -------
    accuracy : float
        聚类准确率（百分比）。
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(y_true, y_pred) * 100

def main():
    data_dir = r"Data\ORL_jpg"
    X, y = load_orl_dataset(data_dir)

    # 降维
    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)

    # K-means聚类
    kmeans = KMeans(n_clusters=40, n_init=50)
    y_kmeans = kmeans.fit_predict(X_reduced)

    # 计算聚类准确率
    accuracy = calculate_clustering_accuracy(y, y_kmeans)
    print(f"Clustering Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()