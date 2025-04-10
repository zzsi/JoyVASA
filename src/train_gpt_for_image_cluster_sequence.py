"""
Use nanoGPT recipe to train a GPT model to generate image cluster sequences.

The data looks like:

video1: 142, 142, 142, 142, 290, 142, 142, 142, 142, 142, 290, 290, 290, 290, 290, 290, 290, 290, 290, 290, 142, 290, 142, 290, 290, 290, 290, 36, 183, 307, 183, 243, 336, 336, 167, 167, 256, 256, 72, 72, 72, 72, 364, 364, 395, 144, 396, 0, 182, 197, 284, 96, 426, 389, 138, 59, 158, 158, 451, 435, 435, 179, 179, 248, 248, 248, 366, 205, 99, 84, 284, 26, 125, 125, 103, 117, 117, 117, 287, 287, 287, 287, 476, 495, 176, 26, 125, 495, 495, 495, 495, 284, 284, 284, 26, 26, 125, 125, 125, 125, 228, 228, 90, 241, 284, 26, 4, 203, 307, 21, 21, 414, 483, 483, 483, 483, 134, 134, 205, 409, 409, 59, 95, 4, 172, 172, 59, 197, 15, 60, 60, 158, 378, 378, 378, 248, 248, 179, 179, 69, 215, 69, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 31, 31, 31, 179, 179, 179, 179, 179, 179, 179, 179, 179, 65, 65, 134
video2: 142, 142, 290, 290, 290, 290, 290, 290, 290, 290, 290, 2
...

First, we will tackle unconditional generation.

Then, we will add an audio (frames of audio features) prompt which aligns with the video cluster id sequence.
"""
import json
import os
import numpy as np
from cvlization.torch.training_pipeline.lm.gpt import NanoGPTTrainingPipeline


def load_video_cluster_id_sequences(json_path: str) -> list[list[int]]:
    """
    Load the video cluster id sequences from a json file.

    The json file looks like:
    ```
        {
            "data/batch_generated_videos/bithuman_coach/bithuman_coach_cropped_common_voice_ab_29347741_lip_temp.mp4": {
                "frame_indices": [
                    0,
                    1,
                    2,
                ],
                "cluster_ids": [
                    142,
                    142,
                    290,
                ],
            }
        }
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return [d["cluster_ids"] for d in data.values()]


class VideoClusterIdSequenceDatasetBuilder:
    def __init__(self, json_path: str, max_sequence_length: int = 128):
        self.video_cluster_id_sequences = load_video_cluster_id_sequences(json_path)
        self.max_sequence_length = max_sequence_length
        
        # Break down long sequences into smaller chunks
        equalized_sequences = []
        for sequence in self.video_cluster_id_sequences:
            if len(sequence) == self.max_sequence_length:
                equalized_sequences.append(sequence)
            elif len(sequence) > self.max_sequence_length:
                # Split long sequence into chunks of max_sequence_length
                for i in range(0, len(sequence), self.max_sequence_length):
                    chunk = sequence[i:i + self.max_sequence_length]
                    if len(chunk) == self.max_sequence_length:  # Only keep chunks that are exactly the max length
                        equalized_sequences.append(chunk)
        
        # Convert sequences to numpy arrays
        equalized_sequences = [np.array(seq) for seq in equalized_sequences]
        
        # Split into train and validation sets
        n_train = int(len(equalized_sequences) * 0.8)
        self.train_video_cluster_id_sequences = equalized_sequences[:n_train]
        self.val_video_cluster_id_sequences = equalized_sequences[n_train:]
        self.train_video_cluster_id_sequences = np.array(self.train_video_cluster_id_sequences)
        self.val_video_cluster_id_sequences = np.array(self.val_video_cluster_id_sequences)
        assert len(self.train_video_cluster_id_sequences.shape) == 2, f"train_video_cluster_id_sequences should be a 2D array, got {self.train_video_cluster_id_sequences.shape}"

    def training_dataset(self):
        return self.train_video_cluster_id_sequences

    def validation_dataset(self):
        return self.val_video_cluster_id_sequences


def main():
    pipeline = NanoGPTTrainingPipeline(
        config=NanoGPTTrainingPipeline.Config(
            log_dir="data/gpt_logs/unconditional_generation",  # edit me: For each run, try to use a different log directory.
            block_size=32,
            vocab_size=768,
            batch_size=32,
            flatten_tokens=False,
            n_layer=3,
            n_head=3,
            n_embd=768,
        )
    )
    dataset_builder = VideoClusterIdSequenceDatasetBuilder(
        json_path="data/batch_generated_videos/bithuman_coach_image_clusters/video_sequences.json"
    )
    # save the config to the log directory
    with open(os.path.join(pipeline.config.log_dir, "config.json"), "w") as f:
        json.dump(pipeline.config.to_dict(), f)
    pipeline.fit(dataset_builder)


if __name__ == "__main__":
    main()
