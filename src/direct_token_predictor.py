"""
Clustering based audio-to-visual token predictor.

1. Load audio-to-visual token data. Make sure audio features and visual tokens are aligned.
2. Train a clustering model to cluster the audio frames.
3. Do a simple counting on audio token, visual token pairs. Associate each audio token to visual tokens.
4. Save the clusters.
"""
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from typing import List, Tuple, Dict


def load_audio_visual_token_data(data_dir: str, max_files: int = None):
    """
    Load audio-to-visual token data.
    """
    npy_files = glob(os.path.join(data_dir, "*.npz"))
    assert len(npy_files) > 0, f"No npy files found in {data_dir}"
    audio_visual_token_data = []
    if max_files is not None:
        npy_files = npy_files[:max_files]
    for npy_file in tqdm(npy_files):
        loaded = np.load(npy_file, allow_pickle=True)
        audio_features = loaded["audio_features"]
        visual_tokens = loaded["visual_cluster_ids"]
        assert len(audio_features.shape) == 3
        # truncate visual tokens to the same length as audio features
        visual_tokens = visual_tokens[:audio_features.shape[1]]
        audio_visual_token_data.append((audio_features, visual_tokens))
    return audio_visual_token_data


def cluster_audio_features(audio_features: np.ndarray, n_clusters: int = 100, method: str = "dbscan"):
    """
    Cluster the audio features.
    """
    # flatten the audio features
    audio_features = audio_features.reshape(-1, audio_features.shape[-1])
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(audio_features)
        return kmeans.cluster_centers_
    elif method == "dbscan":
        dbscan = DBSCAN(eps=0.0001, min_samples=1, metric="cosine")
        dbscan.fit(audio_features)
        # DBSCAN doesn't have cluster_centers_ attribute
        # Instead, compute centroids for each cluster
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        centers = np.zeros((n_clusters, audio_features.shape[1]))
        print(f"number of clusters: {n_clusters}")
        for i in range(n_clusters):
            centers[i] = audio_features[labels == i].mean(axis=0)
        return centers
    else:
        raise ValueError(f"Invalid method: {method}")


def tokenize_audio_features(audio_features: np.ndarray, cluster_centers: np.ndarray):
    """
    Tokenize the audio features.
    """
    audio_tokens = np.argmax(audio_features @ cluster_centers.T, axis=-1)
    return audio_tokens


def build_audio_to_visual_token_mapping(audio_tokens: np.ndarray, visual_tokens: np.ndarray):
    """
    Build a mapping from audio tokens to visual tokens.

    The mapping looks like:
    {
        "audio_token1": {
            "visual_token1": 2,
            "visual_token2": 5,
        },
    }
    """
    audio_token_to_visual_token_mapping = {}
    for audio_token, visual_token in zip(audio_tokens, visual_tokens):
        if audio_token not in audio_token_to_visual_token_mapping:
            audio_token_to_visual_token_mapping[audio_token] = {}
        if visual_token not in audio_token_to_visual_token_mapping[audio_token]:
            audio_token_to_visual_token_mapping[audio_token][visual_token] = 0
        audio_token_to_visual_token_mapping[audio_token][visual_token] += 1
    return audio_token_to_visual_token_mapping


def evaluate_token_prediction_accuracy(audio_token_to_visual_token_mapping: Dict[int, Dict[int, int]], audio_tokens: np.ndarray, visual_tokens: np.ndarray):
    """
    Evaluate the accuracy of the token prediction.
    """
    correct = 0
    total = 0
    for audio_token, visual_token in zip(audio_tokens, visual_tokens):
        # predict based on maximum probability
        pred_visual_token = max(audio_token_to_visual_token_mapping[audio_token], key=audio_token_to_visual_token_mapping[audio_token].get)
        if pred_visual_token == visual_token:
            correct += 1
        total += 1
    print(f"Accuracy: {correct / total}")
    return correct / total


def train_logistic_regression(audio_features: np.ndarray, visual_tokens: np.ndarray, max_iter: int = 1000):
    """
    Train a logistic regression classifier to predict visual tokens from audio features.
    
    Args:
        audio_features: Flattened audio features array of shape (n_samples, n_features)
        visual_tokens: Array of visual token labels of shape (n_samples,)
        max_iter: Maximum number of iterations for the logistic regression solver
        
    Returns:
        Trained logistic regression model
    """
    # Create base model with stronger regularization (smaller C value)
    base_model = LogisticRegression(max_iter=max_iter, C=0.1)
    
    base_model.fit(audio_features, visual_tokens)
    
    return base_model


def evaluate_logistic_regression(clf, audio_features: np.ndarray, visual_tokens: np.ndarray):
    """
    Evaluate the accuracy of the logistic regression classifier.
    
    Args:
        clf: Trained logistic regression model
        audio_features: Flattened audio features array of shape (n_samples, n_features)
        visual_tokens: Array of visual token labels of shape (n_samples,)
    """
    accuracy = clf.score(audio_features, visual_tokens)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    return accuracy


def train_knn(audio_features: np.ndarray, visual_tokens: np.ndarray, n_neighbors: int = 5, leaf_size: int = 5):
    """
    Train a KNN classifier to predict visual tokens from audio features.
    
    Args:
        audio_features: Flattened audio features array of shape (n_samples, n_features)
        visual_tokens: Array of visual token labels of shape (n_samples,)
        n_neighbors: Number of neighbors to use for KNN classification
        
    Returns:
        Trained KNN classifier
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
    knn.fit(audio_features, visual_tokens)
    return knn


def evaluate_knn(knn, audio_features: np.ndarray, visual_tokens: np.ndarray):
    """
    Evaluate the accuracy of the KNN classifier.
    
    Args:
        knn: Trained KNN classifier
        audio_features: Flattened audio features array of shape (n_samples, n_features)
        visual_tokens: Array of visual token labels of shape (n_samples,)
    """
    accuracy = knn.score(audio_features, visual_tokens)
    print(f"KNN Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    n_clusters = 500
    data_dir = "data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50/tokenized_data_mel"
    audio_visual_token_data = load_audio_visual_token_data(data_dir, max_files=1000)
    print(f"number of audio-visual token data: {len(audio_visual_token_data)}")
    sample_size = len(audio_visual_token_data)
    train_size = int(sample_size * 0.8)
    train_audio_visual_token_data = audio_visual_token_data[:train_size]
    val_audio_visual_token_data = audio_visual_token_data[train_size:]
    train_audio_features = [data[0] for data in train_audio_visual_token_data]
    train_visual_tokens = [data[1] for data in train_audio_visual_token_data]
    val_audio_features = [data[0] for data in val_audio_visual_token_data]
    val_visual_tokens = [data[1] for data in val_audio_visual_token_data]
    
    # Flatten the features
    flattened_train_audio_features = np.concatenate([feature.reshape(-1, feature.shape[-1]) for feature in train_audio_features], axis=0)
    flattened_train_visual_tokens = np.concatenate(train_visual_tokens, axis=0)
    flattened_val_audio_features = np.concatenate([feature.reshape(-1, feature.shape[-1]) for feature in val_audio_features], axis=0)
    flattened_val_visual_tokens = np.concatenate(val_visual_tokens, axis=0)
    
    print(f"flattened_train_audio_features shape: {flattened_train_audio_features.shape}")
    print(f"flattened_train_visual_tokens shape: {flattened_train_visual_tokens.shape}")
    print(f"flattened_val_audio_features shape: {flattened_val_audio_features.shape}")
    print(f"flattened_val_visual_tokens shape: {flattened_val_visual_tokens.shape}")
    
    # Train and evaluate logistic regression with regularization
    # print("Training logistic regression classifier...")
    # clf = train_logistic_regression(flattened_train_audio_features, flattened_train_visual_tokens)
    # print("Evaluating logistic regression classifier on train set...")
    # evaluate_logistic_regression(clf, flattened_train_audio_features, flattened_train_visual_tokens)
    # print("Evaluating logistic regression classifier on val set...")
    # evaluate_logistic_regression(clf, flattened_val_audio_features, flattened_val_visual_tokens)
    
    # Train and evaluate KNN
    if True:
        # First try with 10% of training data
        train_size_10p = int(len(flattened_train_audio_features) * 0.1)
        print("\nTraining KNN classifier with 10% of data...")
        knn_10p = train_knn(flattened_train_audio_features[:train_size_10p], 
                           flattened_train_visual_tokens[:train_size_10p], 
                           n_neighbors=1, leaf_size=2)
        print("Evaluating KNN (10% training data) on train set...")
        evaluate_knn(knn_10p, flattened_train_audio_features[:train_size_10p], 
                    flattened_train_visual_tokens[:train_size_10p])
        print("Evaluating KNN (10% training data) on val set...")
        evaluate_knn(knn_10p, flattened_val_audio_features, flattened_val_visual_tokens)

        # Now try with full training data
        print("\nTraining KNN classifier with full data...")
        knn = train_knn(flattened_train_audio_features, flattened_train_visual_tokens, 
                       n_neighbors=1, leaf_size=2)
        print("Evaluating KNN (full training data) on train set...")
        evaluate_knn(knn, flattened_train_audio_features, flattened_train_visual_tokens)
        print("Evaluating KNN (full training data) on val set...")
        evaluate_knn(knn, flattened_val_audio_features, flattened_val_visual_tokens)


if __name__ == "__main__":
    main()


