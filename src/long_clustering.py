"""
Chen Long's clustering method.

1. Extract audio features from audio file.
2. Cluster audio features using k-means.
3. Save the cluster assignments back to each audio object (a python dataclass).
4. For each cluster, collect a sample of members, with their audio files and audio frame indices.
5. Use JoyVASA or Wav2Lip to generate a video for each audio file that is in the sample.
6. For each cluster, collect the video frames of the sample members.
"""
import random
import torch
import torchvision
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from src.audio_utils import extract_audio_features
from src.generate_image_clusters import extract_frames_from_video
import cv2
import clip
import json
import pickle


def extract_image_embeddings(frames: List[np.ndarray], device: str = None) -> torch.Tensor:
    """
    Extract CLIP embeddings from a list of frames.
    
    Args:
        frames: List of numpy arrays representing frames (BGR format)
        device: Device to run CLIP on (default: cuda if available, else cpu)
        
    Returns:
        Tensor of normalized CLIP embeddings (N, 512)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Process frames in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        batch_images = []
        
        # Convert frames to PIL images and preprocess
        for frame in batch_frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            preprocessed_image = preprocess(image)
            batch_images.append(preprocessed_image)
        
        # Stack images into a batch
        image_batch = torch.stack(batch_images).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = model.encode_image(image_batch)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    if all_embeddings:
        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings
    else:
        raise ValueError("No valid frames found to extract embeddings from")


@dataclass
class AudioObject:
    audio_path: str
    audio_model: str = None
    audio_feature: np.ndarray = None
    audio_cluster_ids: List[int] = None
    video_path: str = None  # joyvsa generated video path


def extract_audio_features_from_files(audio_paths: List[str], audio_model: str) -> List[AudioObject]:
    audio_objects = []
    for audio_path in audio_paths:
        audio_feature = extract_audio_features(
            audio_file=audio_path,
            audio_model=audio_model,
            pad_audio=False,
            device="cuda",
            stack_adjacent_frames=False,
            sample_rate=16000,
            fps=25,
        )
        if isinstance(audio_feature, np.ndarray):
            audio_feature = torch.from_numpy(audio_feature)
        assert isinstance(audio_feature, torch.Tensor)
        audio_objects.append(AudioObject(audio_path, audio_model, audio_feature))
    return audio_objects


def cluster_audio_features(audio_objects: List[AudioObject], n_clusters: int) -> List[AudioObject]:
    audio_features = [obj.audio_feature.cpu().numpy()[0] for obj in audio_objects]  # each element of this list has a different shape, (1, L, 512)
    assert len(audio_features[0].shape) == 2
    # assert audio_features[0].shape[1] == 512, f"{audio_features[0].shape}"
    flattened_audio_features = np.concatenate(audio_features, axis=0)
    assert len(flattened_audio_features.shape) == 2
    # assert flattened_audio_features.shape[1] == 512
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(flattened_audio_features)
    for i, obj in enumerate(audio_objects):
        audio_cluster_ids = kmeans.predict(obj.audio_feature.cpu().numpy()[0])
        obj.audio_cluster_ids = audio_cluster_ids.tolist()
        assert len(obj.audio_cluster_ids) == obj.audio_feature.shape[-2]
    return audio_objects, kmeans


def collect_sample_members(audio_objects: List[AudioObject], n_samples: int, n_clusters: int) -> Dict[int, List[AudioObject]]:
    sample_members = {}
    for cluster_id in range(n_clusters):
        cluster_members = [obj for obj in audio_objects if cluster_id in obj.audio_cluster_ids]
        if len(cluster_members) == 0:
            print(f"... cluster {cluster_id} has no members")
            continue
        random.seed(cluster_id)
        sample_members[cluster_id] = random.sample(cluster_members, min(n_samples, len(cluster_members)))
        assert isinstance(sample_members[cluster_id][0], AudioObject), f"{type(sample_members[cluster_id][0])}"
    return sample_members


def generate_video(audio_path: str) -> str:
    return f"{audio_path}"


def generate_videos(sample_members: List[AudioObject]) -> List[AudioObject]:
    for obj in sample_members:
        obj.video_path = generate_video(obj.audio_path)
    return sample_members


def collect_video_frames(sample_members: List[AudioObject], cluster_idx: int, audio_offset: int = 2) -> List[np.ndarray]:
    all_video_frames = []
    
    for obj in sample_members:
        assert len(obj.audio_cluster_ids) > 0
        frame_indices = np.where(np.asarray(obj.audio_cluster_ids) == cluster_idx)[0]
        random.seed(0)
        random.shuffle(frame_indices)
        if len(frame_indices) == 0:
            continue
        video_frames = extract_frames_from_video(obj.video_path)
        print(f"audio frames: {len(obj.audio_cluster_ids)}, video frames: {len(video_frames)}")
        if frame_indices[0] + audio_offset < len(video_frames):
            frame = video_frames[frame_indices[0] + audio_offset][2]  # Get the frame array
            all_video_frames.append(frame)
    
    # Extract embeddings and find centroid frame
    if all_video_frames:
        embeddings = extract_image_embeddings(all_video_frames)
        centroid = torch.mean(embeddings, dim=0)
        similarities = torch.nn.functional.cosine_similarity(centroid.unsqueeze(0), embeddings, dim=1)
        centroid_idx = torch.argmax(similarities).item()
        
        # Move centroid frame to the front
        centroid_frame = all_video_frames[centroid_idx]
        all_video_frames.pop(centroid_idx)
        all_video_frames.insert(0, centroid_frame)
    
    # Limit to 10 frames
    all_video_frames = all_video_frames[:10]
    
    return all_video_frames


def main():
    from glob import glob
    from tqdm import tqdm

    audio_model = "wav2lip"
    # audio_model = "mel"
    # audio_model = "hubert_zh"
    audio_offset = 2
    n_samples = 15
    # output_dir = f"data/tmp_wav2lip_videos/{audio_model}_clustering_offset_{audio_offset}"
    output_dir = f"data/tmp_joyvasa_videos/{audio_model}_clustering_offset_{audio_offset}"
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for saving cluster data
    cluster_data_dir = os.path.join(output_dir, "cluster_data")
    os.makedirs(cluster_data_dir, exist_ok=True)

    audio_paths = glob("data/conversations_joyvasa_videos/bithuman_coach2/*lip.mp4")
    # audio_paths = glob("data/conversations_wav2lip_videos/*wav2lip.mp4")
    assert len(audio_paths) > 0, "No audio paths found"
    audio_paths = list(sorted(audio_paths))
    val_audio_paths = audio_paths[500:]
    audio_paths = audio_paths[:500]
    audio_objects = extract_audio_features_from_files(audio_paths, audio_model)
    n_clusters = 300
    print(f"Clustering {len(audio_objects)} audio features into {n_clusters} clusters...")
    audio_objects, kmeans = cluster_audio_features(audio_objects, n_clusters=n_clusters)
    print(f"Done clustering.")
    sample_members_all_clusters = collect_sample_members(audio_objects, n_samples=n_samples, n_clusters=n_clusters)
    
    # Store centroid frames for each cluster
    cluster_centroid_frames = []
    
    for cluster_id in tqdm(range(n_clusters), desc="Gathering video frames", total=n_clusters):
        sample_members = sample_members_all_clusters[cluster_id]
        sample_members = generate_videos(sample_members)
        video_frames = collect_video_frames(sample_members, cluster_id, audio_offset=audio_offset)
        
        frame = None
        # Save the centroid frame (first frame) for this cluster
        if video_frames:
            frame = video_frames[0]
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            cluster_centroid_frames.append(frame)
        else:
            # empty frame
            assert frame is not None, f"No frame found for cluster {cluster_id}, and we do not know the shape of the frame."
            frame = np.zeros(frame.shape, dtype=np.uint8)
            cluster_centroid_frames.append(frame)

        
        # save these video frames as a grid using torchvision
        frames = [torch.from_numpy(frame[...,::-1].copy()).permute(2,0,1) for frame in video_frames]  # Convert BGR to RGB
        if len(frames) > 0 and cluster_id < 30:
            grid = torchvision.utils.make_grid(frames, nrow=5)
            grid = grid.permute(1,2,0).numpy().astype(np.uint8)
            Image.fromarray(grid).save(f"{output_dir}/cluster_{cluster_id:03d}.png")
    
    # Save KMeans model
    kmeans_path = os.path.join(cluster_data_dir, "kmeans_model.pkl")
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Saved KMeans model to {kmeans_path}")
    
    # Save centroid frames
    ## Convert to numpy array
    cluster_centroid_frames = np.array(cluster_centroid_frames)
    centroid_frames_path = os.path.join(cluster_data_dir, "centroid_frames.npy")
    np.save(centroid_frames_path, cluster_centroid_frames)
    print(f"Saved centroid frames to {centroid_frames_path}")

    # load from the npy file and print out the shape
    centroid_frames = np.load(centroid_frames_path)
    print(f"Loaded centroid frames from {centroid_frames_path}, shape: {centroid_frames.shape}")
    
    # Save metadata
    metadata = {
        "audio_model": audio_model,
        "n_clusters": n_clusters,
        "audio_offset": audio_offset,
        "audio_paths": audio_paths,
        "val_audio_paths": val_audio_paths,
        "cluster_data_dir": cluster_data_dir,
        "output_dir": output_dir
    }
    metadata_path = os.path.join(cluster_data_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
