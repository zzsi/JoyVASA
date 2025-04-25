"""
Load the kmeans model, centroid frames, and metadata from the cluster_data_dir.
Generate a video from an audio file.
"""
import torch
import cv2
import clip
import json
import os
import pickle
import numpy as np
from src.audio_utils import extract_audio_features
from src.generate_image_clusters import extract_frames_from_video
from src.utils.video import images2video, add_audio_to_video


class InferencePipeline:
    def __init__(self, cluster_data_dir: str):
        with open(os.path.join(cluster_data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        with open(os.path.join(cluster_data_dir, "kmeans.pkl"), "rb") as f:
            self.kmeans = pickle.load(f)
        with open(os.path.join(cluster_data_dir, "centroid_frames.npz"), "rb") as f:
            self.centroid_frames = np.load(f)
        self.audio_model = self.metadata["audio_model"]
        self.n_clusters = self.metadata["n_clusters"]

    def generate_video(self, audio_path: str, output_dir: str, overwrite: bool = False) -> str:
        output_path = os.path.join(output_dir, os.path.basename(audio_path).replace(".wav", ".mp4"))
        if os.path.exists(output_path) and not overwrite:
            print(f"Video already exists: {output_path}")
            return output_path
        # Extract audio features
        audio_features = extract_audio_features(
            audio_file=audio_path,
            audio_model=self.audio_model,
            pad_audio=False,
            device="cpu",
            stack_adjacent_frames=False,
            sample_rate=16000,
        )
        audio_features = audio_features.cpu().numpy()[0]
        assert len(audio_features.shape) == 2

        # Turn audio features into audio cluster ids
        audio_cluster_ids = self.kmeans.predict(audio_features)

        # Turn cluster ids into centroid frames
        centroid_frames = [self.centroid_frames[str(cluster_id)] for cluster_id in audio_cluster_ids]

        # Generate video
        silent_video_path = output_path.replace(".mp4", "_silent.mp4")
        images2video(centroid_frames, wfp=silent_video_path)

        # Add audio to video
        add_audio_to_video(silent_video_path, audio_path, output_path)

        # Remove silent video
        os.remove(silent_video_path)

        return output_path


def main(args):
    pipeline = InferencePipeline(args.cluster_data_dir)
    pipeline.generate_video(args.audio_path, args.output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="data/conversations/ffb948556a252fec4aa0601da677fda38bb2ab0be63cc9c726bebfd1b3500d62_tts-1_nova.wav")
    parser.add_argument("--cluster_data_dir", type=str, default="data/tmp_joyvasa_videos/wav2lip_clustering_offset_2")
    parser.add_argument("--output_dir", type=str, default="data/tmp_joyvasa_videos/wav2lip_clustering_offset_2/generated/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
