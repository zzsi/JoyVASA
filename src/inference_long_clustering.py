"""
Load the kmeans model, centroid frames, and metadata from the cluster_data_dir.
Generate a video from an audio file.
"""
import json
import os
import pickle
import numpy as np
from src.audio_utils import extract_audio_features
from src.utils.video import images2video, add_audio_to_video


class InferencePipeline:
    def __init__(self, cluster_data_dir: str):
        with open(os.path.join(cluster_data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        with open(os.path.join(cluster_data_dir, "kmeans_model.pkl"), "rb") as f:
            self.kmeans = pickle.load(f)
        with open(os.path.join(cluster_data_dir, "centroid_frames.npy"), "rb") as f:
            self.centroid_frames = np.load(f)  # this is a np.ndarray
            assert len(self.centroid_frames.shape) == 4  # (n_clusters, h, w, 3)
            assert self.centroid_frames.shape[3] == 3  # RGB
        # Print out content of first frame
        # import sys; sys.exit()
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
            fps=25,
        )
        audio_features = audio_features.cpu().numpy()[0]
        assert len(audio_features.shape) == 2

        # Turn audio features into audio cluster ids
        audio_cluster_ids = self.kmeans.predict(audio_features)
        print("audio_cluster_ids", audio_cluster_ids)

        # Turn cluster ids into centroid frames
        centroid_frames = self.centroid_frames[audio_cluster_ids].copy()

        # Generate video
        silent_video_path = output_path.replace(".mp4", "_silent.mp4")
        # make sure the directory exists
        os.makedirs(os.path.dirname(silent_video_path), exist_ok=True)
        images2video(centroid_frames, wfp=silent_video_path, image_mode="bgr")

        # Add audio to video
        add_audio_to_video(silent_video_path, audio_path, output_path, remove_temp=True)

        return output_path


def main(args):
    pipeline = InferencePipeline(args.cluster_data_dir)
    pipeline.generate_video(args.audio_path, args.output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--audio_path", type=str, default="data/conversations/ff394950e5e2b7d3a4b62f13a69e20f07c8fcea65216b02ca7a7502ce0c24efc_tts-1_nova.wav")
    parser.add_argument("--audio_path", type=str, default="data/conversations/fff95ab1997fd754d0b22e2402efbc1c848f61b85563d93f8aa8c767357d0aac_tts-1_nova.wav")
    parser.add_argument("--cluster_data_dir", type=str, default="data/tmp_joyvasa_videos/wav2lip_clustering_offset_2/cluster_data")
    parser.add_argument("--output_dir", type=str, default="data/tmp_joyvasa_videos/wav2lip_clustering_offset_2/generated/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
