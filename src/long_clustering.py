"""
Chen Long's clustering method.

1. Extract audio features from audio file.
2. Cluster audio features using k-means.
3. Save the cluster assignments back to each audio object (a python dataclass).
4. For each cluster, collect a sample of members, with their audio files and audio frame indices.
5. Use JoyVASA to generate a video for each audio file that is in the sample.
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
        audio_feature = torch.from_numpy(audio_feature)
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
    return audio_objects


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


def collect_video_frames(sample_members: List[AudioObject], cluster_idx: int, audio_offset: int = 2) -> List[AudioObject]:
    all_video_frames = []
    for obj in sample_members:
        assert len(obj.audio_cluster_ids) > 0
        frame_indices = np.where(np.asarray(obj.audio_cluster_ids) == cluster_idx)[0]
        random.seed(0)
        random.shuffle(frame_indices)
        if len(frame_indices) == 0:
            continue
        # print(f"frame_indices: {frame_indices}")
        video_frames = extract_frames_from_video(obj.video_path)
        # print(f"video_frames: {len(video_frames)}")
        print(f"audio frames: {len(obj.audio_cluster_ids)}, video frames: {len(video_frames)}")
        if frame_indices[0] + audio_offset < len(video_frames):
            sampled_video_frames = [video_frames[frame_indices[0] + audio_offset][2]]  # only take the first frame
            all_video_frames.extend(sampled_video_frames)
    return all_video_frames



def main():
    from glob import glob
    from tqdm import tqdm

    # audio_model = "wav2lip"
    audio_model = "mel"
    # audio_model = "hubert_zh"
    audio_offset = 0
    output_dir = f"data/tmp/{audio_model}_clustering_offset_{audio_offset}"
    os.makedirs(output_dir, exist_ok=True)

    audio_paths = glob("data/conversations_joyvasa_videos/bithuman_coach2/*lip.mp4")
    assert len(audio_paths) > 0, "No audio paths found"
    audio_paths = list(sorted(audio_paths))
    val_audio_paths = audio_paths[500:]
    audio_paths = audio_paths[:500]
    audio_objects = extract_audio_features_from_files(audio_paths, audio_model)
    n_clusters = 300
    audio_objects = cluster_audio_features(audio_objects, n_clusters=n_clusters)
    print(f"Done clustering audio features into {n_clusters} clusters")
    sample_members_all_clusters = collect_sample_members(audio_objects, n_samples=10, n_clusters=n_clusters)
    for cluster_id in tqdm(range(n_clusters), desc="Gathering video frames", total=n_clusters):
        sample_members = sample_members_all_clusters[cluster_id]
        sample_members = generate_videos(sample_members)
        video_frames = collect_video_frames(sample_members, cluster_id, audio_offset=audio_offset)
        # save these video frames as a grid using torchvision
        frames = [torch.from_numpy(frame[...,::-1].copy()).permute(2,0,1) for frame in video_frames]  # Convert BGR to RGB
        if len(frames) > 0:
            grid = torchvision.utils.make_grid(frames, nrow=5)
            grid = grid.permute(1,2,0).numpy().astype(np.uint8)
            Image.fromarray(grid).save(f"{output_dir}/cluster_{cluster_id:03d}.png")


if __name__ == "__main__":
    main()
