"""
Use nanoGPT recipe to train a GPT model to generate image cluster sequences.

The data looks like:

video1: 142, 142, 142, 142, 290, 142, 142, 142, 142, 142, 290, 290, 290, 290, 290, 290, 290, 290, 290, 290, 142, 290, 142, 290, 290, 290, 290, 36, 183, 307, 183, 243, 336, 336, 167, 167, 256, 256, 72, 72, 72, 72, 364, 364, 395, 144, 396, 0, 182, 197, 284, 96, 426, 389, 138, 59, 158, 158, 451, 435, 435, 179, 179, 248, 248, 248, 366, 205, 99, 84, 284, 26, 125, 125, 103, 117, 117, 117, 287, 287, 287, 287, 476, 495, 176, 26, 125, 495, 495, 495, 495, 284, 284, 284, 26, 26, 125, 125, 125, 125, 228, 228, 90, 241, 284, 26, 4, 203, 307, 21, 21, 414, 483, 483, 483, 483, 134, 134, 205, 409, 409, 59, 95, 4, 172, 172, 59, 197, 15, 60, 60, 158, 378, 378, 378, 248, 248, 179, 179, 69, 215, 69, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 179, 31, 31, 31, 179, 179, 179, 179, 179, 179, 179, 179, 179, 65, 65, 134
video2: 142, 142, 290, 290, 290, 290, 290, 290, 290, 290, 290, 2
...

First, we will tackle unconditional generation.

Then, we will add an audio (frames of audio features) prompt which aligns with the video cluster id sequence.
"""
from dataclasses import asdict
import json
import os
import numpy as np
import torch
from contextlib import nullcontext
from src.gpt import NanoGPTTrainingPipeline, GPTConfig, GPT


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


def main(start_token: int = 767):
    pipeline = NanoGPTTrainingPipeline(
        config=NanoGPTTrainingPipeline.Config(
            log_dir="data/gpt_logs/unconditional_generation",  # edit me: For each run, try to use a different log directory.
            block_size=3,
            vocab_size=start_token + 1,
            batch_size=128,
            flatten_tokens=False,
            n_layer=6,
            n_head=3,
            n_embd=36,
            start_token=start_token,
            # n_embd=768,
            log_interval=100,
            max_iters=200000,
        )
    )
    dataset_builder = VideoClusterIdSequenceDatasetBuilder(
        json_path="data/batch_generated_videos/bithuman_coach_image_clusters/video_sequences.json"
    )
    # save the config to the log directory
    with open(os.path.join(pipeline.config.log_dir, "config.json"), "w") as f:
        config_dict = asdict(pipeline.config)
        json.dump(config_dict, f)
    pipeline.fit(dataset_builder)


def detokenize(token_ids: list[int], cluster_data_dir: str, output_path: str = None, fps: int = 25) -> np.ndarray:
    """
    Detokenize the image cluster ids into images.
    
    Args:
        token_ids: List of cluster IDs to detokenize
        cluster_data_dir: Directory containing the cluster data (cluster_centers.npy, etc.)
        output_path: Optional path to save the detokenized video
        fps: Frames per second for the output video
        
    Returns:
        List of frames corresponding to the token IDs
    """
    import os
    import numpy as np
    import logging
    from src.generate_image_clusters import ImageClusterGenerator
    from src.utils.video import images2video
    
    logger = logging.getLogger(__name__)
    
    # Initialize the cluster generator and load the data
    cluster_generator = ImageClusterGenerator(device='cpu', audio_feature_type=None)
    cluster_generator.load_cluster_data(cluster_data_dir)
    
    # Get the first frame from any cluster to determine dimensions
    first_valid_frame = None
    # print(cluster_generator.cluster_to_frames)
    for cluster_id in range(cluster_generator.n_clusters):
        if cluster_id in cluster_generator.cluster_to_frames and cluster_generator.cluster_to_frames[cluster_id]:
            first_valid_frame = cluster_generator.cluster_to_frames[cluster_id][0][2]  # Get the frame array
            break
    
    if first_valid_frame is None:
        logger.warning("No valid frames found in any cluster.")
        return []
    
    height, width = first_valid_frame.shape[:2]
    
    # Process each cluster ID in the sequence
    frames = []
    logger.info(f"Detokenizing the cluster sequence: {token_ids}")
    
    for cluster_id in token_ids:
        if cluster_id not in cluster_generator.cluster_to_frames or not cluster_generator.cluster_to_frames[cluster_id]:
            # If no valid frame for this cluster, use a black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Select a representative frame from the cluster
            # For simplicity, just use the first frame in the cluster
            frame = cluster_generator.cluster_to_frames[cluster_id][0][2]
        
        frames.append(frame)
    
    # Save the video if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images2video(frames, wfp=output_path, fps=fps, image_mode='bgr')
        logger.info(f"Generated detokenized video: {output_path}")
    
    return frames


class GPTInferencePipeline:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = device

    def generate(self, start_ids: torch.Tensor, max_new_tokens: int = 128, num_samples: int = 1, temperature: float = 1.0, top_k: int = 10):
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]
        ctx = (
            nullcontext()
            if self.device == "cpu"
            else torch.amp.autocast(device_type=self.device, dtype=ptdtype)
        )
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        assert start_ids.ndim == 2, f"start_ids should be a 2D tensor, got {start_ids.ndim}"
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    yield y


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or run inference with the GPT model")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", 
                        help="Mode to run the model in: train or inference")
    parser.add_argument("--ckpt_path", type=str, 
                        default="data/gpt_logs/unconditional_generation/batch128_block3/ckpt.pt",
                        help="Path to the checkpoint file for inference")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run the model on: cuda or cpu")
    parser.add_argument("--start_token", type=int, default=767,
                        help="Starting token for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k sampling parameter")
    parser.add_argument("--output_dir", type=str,
                        default="data/batch_generated_videos/bithuman_coach_image_clusters/detokenized_videos",
                        help="Directory to save generated videos")
    parser.add_argument("--log_dir", type=str,
                        default="data/gpt_logs/unconditional_generation",
                        help="Directory to save logs and checkpoints during training")
    parser.add_argument("--block_size", type=int, default=3,
                        help="Block size for the model")
    parser.add_argument("--vocab_size", type=int, default=768,
                        help="Vocabulary size for the model")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--n_layer", type=int, default=6,
                        help="Number of layers in the model")
    parser.add_argument("--n_head", type=int, default=3,
                        help="Number of attention heads in the model")
    parser.add_argument("--n_embd", type=int, default=36,
                        help="Embedding dimension in the model")
    parser.add_argument("--max_iters", type=int, default=200000,
                        help="Maximum number of training iterations")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Run training with the specified parameters
        pipeline = NanoGPTTrainingPipeline(
            config=NanoGPTTrainingPipeline.Config(
                log_dir=args.log_dir,
                block_size=args.block_size,
                vocab_size=args.vocab_size,
                batch_size=args.batch_size,
                flatten_tokens=False,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                start_token=args.start_token,
                log_interval=100,
                max_iters=args.max_iters,
            )
        )
        dataset_builder = VideoClusterIdSequenceDatasetBuilder(
            json_path="data/batch_generated_videos/bithuman_coach_image_clusters/video_sequences.json"
        )
        # Save the config to the log directory
        with open(os.path.join(args.log_dir, "config.json"), "w") as f:
            config_dict = asdict(pipeline.config)
            json.dump(config_dict, f)
        pipeline.fit(dataset_builder)
    else:  # inference
        # Create inference pipeline
        gpt_inference_pipeline = GPTInferencePipeline(
            ckpt_path=args.ckpt_path,
            device=args.device,
        )
        
        # Create initial visual token sequence
        visual_ids = np.array([[args.start_token]] * args.num_samples)
        
        # Generate samples
        os.makedirs(args.output_dir, exist_ok=True)
        for i, y in enumerate(gpt_inference_pipeline.generate(
            start_ids=visual_ids,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
        )):
            print(f"Generated sample {i+1}/{args.num_samples}")
            print(f"First few tokens: {y[0, :10].tolist()}")
            
            # Detokenize and save the generated sequence
            detokenize(
                token_ids=y[0].tolist(),
                cluster_data_dir="data/batch_generated_videos/bithuman_coach_image_clusters",
                output_path=os.path.join(args.output_dir, f"generated_{args.start_token}_{i}.mp4"),
                fps=25,
            )
            
            # Test latency on the first sample
            if i == 0:
                import time
                start_time = time.time()
                n = 10  # Reduced from 100 for faster testing
                for _ in range(n):
                    gpt_inference_pipeline.generate(
                        start_ids=np.array([[args.start_token]]),
                        max_new_tokens=1,
                        num_samples=1,
                    )
                end_time = time.time()
                print(f"Average latency: {(end_time - start_time) * 1000 / n} ms")