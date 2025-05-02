"""
Prepare tokens data (visual id sequence, paired with audio features) for training Lip GPT.

1. Load image clusters
2. For each talking head video, extract the audio features, and quantize the video frames into visual cluster ids, save as npy file
"""
import glob
import os
import numpy as np
from tqdm import tqdm
import torch
from src.generate_image_clusters import ImageClusterGenerator, extract_frames_from_video
from src.audio_utils import extract_audio_features



def load_image_clusters(image_clusters_dir: str, audio_model: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load image clusters from the given directory.
    """
    image_clusters = ImageClusterGenerator(device='cuda' if torch.cuda.is_available() else 'cpu', audio_feature_type=audio_model)
    image_clusters.load_cluster_data(image_clusters_dir)
    return image_clusters


def main(args):
    image_clusters_dir = args.image_clusters_dir
    output_dir = args.output_dir

    image_clusters = load_image_clusters(image_clusters_dir, audio_model=args.audio_model)
    talking_head_video_paths = glob.glob(args.talking_head_video_pattern)

    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for talking_head_video_path in tqdm(talking_head_video_paths):
        output_path = os.path.join(output_dir, os.path.basename(talking_head_video_path).replace("_lip.mp4", ".npz"))
        if not args.overwrite and os.path.exists(output_path):
            print(f"skipping {talking_head_video_path} because it already exists")
            continue
        # extract audio features
        audio_features = extract_audio_features(
            audio_file=talking_head_video_path,
            sample_rate=16000,
            fps=25,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pad_audio=True,
            audio_model=args.audio_model,
            hidden_layer_idx=None,
            stack_adjacent_frames=False,
        )
        # quantize the video frames into visual cluster ids, by finding the nearest cluster center
        video_frames = extract_frames_from_video(talking_head_video_path)
        visual_embeddings = image_clusters.extract_image_embeddings(video_frames)
        visual_cluster_ids = [image_clusters.find_nearest_cluster_given_embedding(embedding) for embedding in visual_embeddings]
        print("video path:", talking_head_video_path)
        print("tokens:", visual_cluster_ids)
        print(f"number of visual cluster ids: {len(visual_cluster_ids)}, number of audio features: {audio_features.shape}")
        # save the visual cluster ids and audio features
        if isinstance(audio_features, torch.Tensor):
            audio_features = audio_features.cpu().numpy()
        np.savez(output_path, audio_features=audio_features, visual_cluster_ids=np.array(visual_cluster_ids))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--image_clusters_dir", type=str, default="data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50")
    # parser.add_argument("--talking_head_video_pattern", type=str, default="data/conversations_joyvasa_videos/bithuman_coach2/*_lip.mp4")
    # parser.add_argument("--talking_head_video_pattern", type=str, default="data/celeb_joyvasa_videos/bithuman_coach2/*_lip.mp4")
    parser.add_argument("--talking_head_video_pattern", type=str, default="data/librispeech_joyvasa_videos/bithuman_coach2/*_lip.mp4")
    parser.add_argument("--audio_model", type=str, default="wav2lip")
    
    # Dumping all video tokens into this directory for now...
    parser.add_argument("--output_dir", type=str, default="data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50/tokenized_data_wav2lip")

    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)

