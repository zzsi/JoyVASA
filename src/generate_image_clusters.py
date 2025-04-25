"""
Take a list of video files (glob pattern), extract image embeddings (using e.g. CLIP),
and run a clustering algorithm to group the images into clusters.

Save the cluster centers and the cluster assignments for each video.

Save some images per cluster (image grid) for visualization.
"""

import os
import glob
import numpy as np
import torchvision
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
import cv2
import argparse
from typing import List, Tuple, Dict, Optional, List
import clip
import json
from .utils.video import images2video
import librosa
import soundfile as sf
import subprocess
import tempfile
import platform
from .config.base_config import make_abs_path

# suppress warnings:
# /workspace/src/generate_image_clusters.py:319: UserWarning: PySoundFile failed. Trying audioread instead.
#   audio, sr = librosa.load(video_path, sr=16000)
# /opt/conda/lib/python3.10/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
#         Deprecated as of librosa version 0.10.0.
#         It will be removed in librosa version 1.0.
#   y, sr_native = __audioread_load(path, offset, duration, dtype)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageClusterGenerator:
    def __init__(
            self, output_dir=None, n_clusters=100, device="cuda",
            model_name="ViT-B/32", audio_feature_type="hubert_zh",
            clear_output_dir=False, frame_interval=1, max_frames=1000, fps=25
        ):
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        self.device = device
        self.model_name = model_name
        self.audio_feature_type = audio_feature_type
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.fps = fps
        
        # Only create output directories if output_dir is provided
        if output_dir is not None:
            # Clear output directory if requested
            if clear_output_dir and os.path.exists(output_dir):
                logger.info(f"Clearing output directory: {output_dir}")
                import shutil
                shutil.rmtree(output_dir)
            
            # Create output directories
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "clusters"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "embeddings"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "audio_features"), exist_ok=True)
        
        # Initialize CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        
        # Initialize HuBERT model if needed
        if audio_feature_type == "hubert_zh":
            logger.info("Initializing HuBERT model for audio feature extraction")
            from .modules.hubert import HubertModel
            
            # Define model path
            model_path = '../../pretrained_weights/chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            
            # Load model
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.audio_encoder = self.audio_encoder.to(device)
        
        # Initialize lists to store data
        self.frame_paths = []
        self.embeddings = []
        self.audio_features = []
        
        # Initialize dictionaries for clustering
        self.cluster_to_frames = {}  # Maps cluster ID to list of frames
        self.frame_to_cluster = {}   # Maps frame key to cluster ID
        self.video_sequences = {}    # Maps video path to sequence of (frame_idx, cluster_id) tuples
        
    def extract_image_embeddings(self, video_frames: List[Tuple[str, int, np.ndarray]], batch_size: int = 32) -> torch.Tensor:
        """Extract embeddings for a list of video frames using CLIP"""
        logger.info(f"Extracting embeddings for {len(video_frames)} frames")
        
        all_embeddings = []
        
        # Process frames in batches
        for i in tqdm(range(0, len(video_frames), batch_size), desc="Extracting embeddings"):
            batch_frames = video_frames[i:i+batch_size]
            batch_images = []
            
            # Convert frames to PIL images and preprocess
            for video_path, frame_idx, frame in batch_frames:
                try:
                    # Convert BGR to RGB (OpenCV uses BGR)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    preprocessed_image = self.preprocess(image)
                    batch_images.append(preprocessed_image)
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_idx} from video {video_path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # Stack images into a batch
            image_batch = torch.stack(batch_images).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model.encode_image(image_batch)
                embeddings = F.normalize(embeddings, dim=-1)  # Normalize embeddings
                all_embeddings.append(embeddings)
        
        # Concatenate all embeddings
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            logger.info(f"Extracted embeddings shape: {all_embeddings.shape}")
            return all_embeddings
        else:
            raise ValueError("No valid frames found to extract embeddings from")
    
    def _get_frame_key(self, video_path: str, frame_idx: int) -> str:
        """Create a unique string key for a frame"""
        return f"{video_path}:{frame_idx}"
    
    def cluster_images(self, embeddings: torch.Tensor, video_frames: List[Tuple[str, int, np.ndarray]]) -> Dict[int, List[Tuple[str, int, np.ndarray]]]:
        """Cluster images based on their embeddings"""
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters...")
        
        # Convert embeddings to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(embeddings_np)
        self.cluster_centers = torch.from_numpy(self.kmeans.cluster_centers_).to(self.device)
        
        # Map frames to clusters
        for i, (label, frame_info) in enumerate(zip(cluster_labels, video_frames)):
            video_path, frame_idx, _ = frame_info
            frame_key = self._get_frame_key(video_path, frame_idx)
            
            if label not in self.cluster_to_frames:
                self.cluster_to_frames[label] = []
            self.cluster_to_frames[label].append(frame_info)
            self.frame_to_cluster[frame_key] = label
            
            # Build video sequences - store as (frame_idx, cluster_id) tuples
            if video_path not in self.video_sequences:
                self.video_sequences[video_path] = []
            self.video_sequences[video_path].append((frame_idx, label))
        
        # Sort sequences by frame index
        for video_path in self.video_sequences:
            self.video_sequences[video_path].sort(key=lambda x: x[0])
        
        # Log cluster statistics
        for label in range(self.n_clusters):
            logger.info(f"Cluster {label}: {len(self.cluster_to_frames.get(label, []))} frames")
        
        return self.cluster_to_frames
    
    def create_cluster_visualization(self, output_dir: str, images_per_cluster: int = 5, grid_size: int = 5):
        """Create visualization grids for each cluster using torchvision utilities"""
        logger.info(f"Creating visualizations for {self.n_clusters} clusters")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # For each cluster, create a grid of images
        for cluster_id in tqdm(range(self.n_clusters), desc="Creating visualizations", total=self.n_clusters):
            if cluster_id not in self.cluster_to_frames:
                continue
                
            cluster_frames = self.cluster_to_frames[cluster_id]
            
            # Select a subset of frames for visualization
            if len(cluster_frames) > images_per_cluster:
                # Try to select diverse frames by taking evenly spaced ones
                indices = np.linspace(0, len(cluster_frames) - 1, images_per_cluster, dtype=int)
                selected_frames = [cluster_frames[i] for i in indices]
            else:
                selected_frames = cluster_frames
            
            # Prepare tensor images for grid
            tensor_images = []
            
            for video_path, frame_idx, frame in selected_frames:
                try:
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to tensor and normalize to [0, 1]
                    tensor_image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                    tensor_images.append(tensor_image)
                except Exception as e:
                    logger.warning(f"Failed to process frame {frame_idx} from video {video_path}: {e}")
            
            if not tensor_images:
                logger.warning(f"No valid frames for cluster {cluster_id}")
                continue
            
            # Create grid using torchvision and flip RGB->BGR
            grid = torchvision.utils.make_grid(
                tensor_images,
                nrow=min(grid_size, len(tensor_images)), 
                padding=5,
                normalize=False
            )
            
            # Convert to PIL image and save directly
            grid_image = torchvision.transforms.ToPILImage()(grid)
            output_path = os.path.join(output_dir, f"cluster_{cluster_id}.png")
            grid_image.save(output_path)
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_detokenized_videos(self, output_dir: str, fps: int = 25, max_videos: int = 2):
        """Generate videos from cluster ID sequences by mapping back to original frames"""
        logger.info("Generating detokenized videos from cluster sequences")
        
        # Create output directory if it doesn't exist
        videos_dir = os.path.join(output_dir, "detokenized_videos")
        os.makedirs(videos_dir, exist_ok=True)

        cnt = 0
        
        # For each video sequence, create a detokenized video
        for video_path, sequence in tqdm(self.video_sequences.items(), desc="Generating detokenized videos"):
            if not sequence:
                continue
                
            # Get video name without extension
            video_name = os.path.basename(video_path).split('.')[0]
            temp_output_path = os.path.join(videos_dir, f"{video_name}_detokenized_temp.mp4")
            final_output_path = os.path.join(videos_dir, f"{video_name}_detokenized.mp4")
            
            # Get the first frame to determine dimensions
            first_valid_frame = None
            for _, cluster_id in sequence:
                if cluster_id in self.cluster_to_frames and self.cluster_to_frames[cluster_id]:
                    first_valid_frame = self.cluster_to_frames[cluster_id][0][2]  # Get the frame array
                    break
            
            if first_valid_frame is None:
                logger.warning(f"No valid frames found for video {video_path}")
                continue
                
            height, width = first_valid_frame.shape[:2]
            
            # Process each cluster ID in the sequence
            frames = []
            print("detokenizing the cluster sequence", [cluster_id for _, cluster_id in sequence])
            for _, cluster_id in sequence:
                if cluster_id not in self.cluster_to_frames or not self.cluster_to_frames[cluster_id]:
                    # If no valid frame for this cluster, use a black frame
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    # Select a representative frame from the cluster
                    # For simplicity, just use the first frame in the cluster
                    frame = self.cluster_to_frames[cluster_id][0][2]
                
                frames.append(frame)
            
            # Use images2video function to create the video without audio first
            images2video(frames, wfp=temp_output_path, fps=fps, image_mode='bgr')

            # Extract audio from original video and add to detokenized video
            try:
                # The audio is in the original mp4 file
                audio_path = video_path.replace("_temp.mp4", ".mp4")
                
                # Check if the audio file exists
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found at {audio_path}, trying alternative path")
                    # Try alternative path construction
                    base_name = os.path.basename(video_path)
                    if "_temp" in base_name:
                        alt_audio_path = video_path.replace("_temp", "")
                    else:
                        # Try to find the audio file in the same directory
                        dir_name = os.path.dirname(video_path)
                        base_name_no_ext = os.path.splitext(base_name)[0]
                        alt_audio_path = os.path.join(dir_name, f"{base_name_no_ext}.mp4")
                    
                    if os.path.exists(alt_audio_path):
                        audio_path = alt_audio_path
                        logger.info(f"Using alternative audio path: {audio_path}")
                    else:
                        logger.warning(f"Could not find audio file for {video_path}, skipping audio")
                        raise FileNotFoundError(f"Audio file not found for {video_path}")
                
                # Use ffmpeg to combine video and audio
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_output_path,  # Input video
                    '-i', audio_path,        # Input audio
                    '-c:v', 'copy',          # Copy video stream
                    '-c:a', 'aac',           # Re-encode audio as AAC
                    final_output_path        # Output file
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Remove temporary video file
                os.remove(temp_output_path)
                
                logger.info(f"Added audio to detokenized video: {final_output_path}")
            except Exception as e:
                logger.warning(f"Failed to add audio to video {video_path}: {e}")
                # If audio addition fails, just rename temp file to final
                os.rename(temp_output_path, final_output_path)
            
            # Save the frames as an image grid
            frames_torch = [torch.from_numpy(frame).permute(2, 0, 1) for frame in frames]
            frames_torch = [F.interpolate(frame.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0) for frame in frames_torch]
            frames_torch = frames_torch[:36]
            grid_image = torchvision.utils.make_grid(
                frames_torch,
                nrow=6,
                padding=5,
                normalize=False
            )
            grid_image = grid_image.flip(0)  # Flip RGB->BGR
            grid_image = torchvision.transforms.ToPILImage()(grid_image)
            grid_image.save(os.path.join(output_dir, f"{video_name}_detokenized_grid.png"))
            logger.info(f"Generated detokenized video: {final_output_path}")

            cnt += 1
            if cnt >= max_videos:
                break
        
        logger.info(f"All detokenized videos saved to {videos_dir}")
    
    def extract_audio_features(self, audio_path):
        from src.audio_utils import extract_audio_features
        return extract_audio_features(
            audio_file=audio_path,
            sample_rate=16000,
            fps=self.fps,
            device=self.device,
            pad_audio=True,
            audio_model=self.audio_feature_type,
        )
    def _extract_hubert_zh_features(self, audio_path):
        """Extract HuBERT-ZH features from the audio file."""
        logger.info(f"Extracting HuBERT-ZH features from {audio_path}")
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Convert audio to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        
        # Define padding function
        def pad_audio(audio):
            return F.pad(audio, (1280, 640), "constant", 0)
        
        # Calculate number of frames based on audio duration
        audio_duration = len(audio) / sr
        total_frames = int(audio_duration * self.fps)
        logger.info(f"Audio duration: {audio_duration:.2f}s, will generate {total_frames} frames")
        
        # Extract features using the encoder
        # The encoder outputs features at 50 FPS, so we need to interpolate to our desired FPS
        with torch.no_grad():
            # Get features at 50 FPS
            hidden_states = self.audio_encoder(pad_audio(audio_tensor), self.fps, frame_num=total_frames * 2).last_hidden_state
            # Transpose for interpolation: (N, L, C) -> (N, C, L)
            hidden_states = hidden_states.transpose(1, 2)
            # Interpolate from 50 FPS to desired FPS
            hidden_states = F.interpolate(hidden_states, size=total_frames, align_corners=False, mode='linear')
            # Transpose back: (N, C, L) -> (N, L, C)
            hidden_states = hidden_states.transpose(1, 2)
        
        # Convert to numpy array
        audio_features = hidden_states[0].cpu().numpy()
        
        return audio_features
        
    def _extract_mfcc_features(self, audio_path, n_mfcc=13):
        """Extract MFCC features from the audio file."""
        logger.info(f"Extracting MFCC features from {audio_path}")
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate frame duration in seconds
        frame_duration = 1.0 / self.fps
        
        # Calculate the minimum FFT size based on the shortest possible audio segment
        min_samples_per_frame = int(frame_duration * sr)
        n_fft = max(32, 2 ** int(np.log2(min(min_samples_per_frame, 2048))))
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
        
        # Calculate number of frames based on audio duration
        audio_duration = len(audio) / sr
        total_frames = int(audio_duration * self.fps)
        
        # Resample MFCC features to match the desired number of frames
        mfcc_resampled = librosa.resample(mfcc, orig_sr=mfcc.shape[1], target_sr=total_frames)
        
        return mfcc_resampled.T
        
    def _extract_mel_features(self, audio_path):
        """Extract Mel spectrogram features from the audio file."""
        logger.info(f"Extracting Mel spectrogram features from {audio_path}")
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate frame duration in seconds
        frame_duration = 1.0 / self.fps
        
        # Calculate the minimum FFT size based on the shortest possible audio segment
        min_samples_per_frame = int(frame_duration * sr)
        n_fft = max(32, 2 ** int(np.log2(min(min_samples_per_frame, 2048))))
        
        # Extract Mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft)
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Calculate number of frames based on audio duration
        audio_duration = len(audio) / sr
        total_frames = int(audio_duration * self.fps)
        
        # Resample Mel features to match the desired number of frames
        mel_spec_resampled = librosa.resample(mel_spec_db, orig_sr=mel_spec_db.shape[1], target_sr=total_frames)
        
        return mel_spec_resampled.T
    
    def save_cluster_data(self, output_dir: str):
        """Save cluster centers and video sequences to files"""
        logger.info(f"Saving cluster data to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cluster centers
        cluster_centers_path = os.path.join(output_dir, "cluster_centers.npy")
        np.save(cluster_centers_path, self.cluster_centers.cpu().numpy())
        logger.info(f"Saved cluster centers to {cluster_centers_path}")
        
        # Save representative frames for each cluster
        # Only save the first frame for each cluster as that's what we use for detokenization
        representative_frames = {}
        all_frames = {}  # Dictionary to store all frames in a single structure
        
        for cluster_id in self.cluster_to_frames:
            if self.cluster_to_frames[cluster_id]:
                # Store only the first frame for each cluster
                # Each entry is (video_path, frame_idx, frame_array)
                frame_data = self.cluster_to_frames[cluster_id][0]
                # Convert frame array to list for JSON serialization
                representative_frames[str(cluster_id)] = {
                    "video_path": frame_data[0],
                    "frame_idx": int(frame_data[1]),
                    "frame_shape": frame_data[2].shape
                }
                # Store the frame in our dictionary
                all_frames[cluster_id] = frame_data[2]
        
        # Save all frames in a single NPY file
        frames_path = os.path.join(output_dir, "cluster_frames.npy")
        np.save(frames_path, all_frames)
        
        frames_metadata_path = os.path.join(output_dir, "representative_frames.json")
        with open(frames_metadata_path, 'w') as f:
            json.dump(representative_frames, f, indent=2)
        logger.info(f"Saved representative frames for {len(representative_frames)} clusters")
        
        # Prepare video sequences data
        video_paths = list(self.video_sequences.keys())
        logger.info(f"Processing {len(video_paths)} video sequences")
        
        # Create a dictionary to store sequences
        sequences_dict = {}
        audio_features_dict = {}
        
        for video_path in video_paths:
            # Get frame indices and cluster IDs
            frame_indices = [int(idx) for idx, _ in self.video_sequences[video_path]]
            cluster_ids = [int(cid) for _, cid in self.video_sequences[video_path]]
            
            # Extract audio features for this video
            try:
                # Get video FPS
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)  # TODO: this is not used anywhere
                cap.release()
                
                # Extract audio features
                # The audio is actually in a differnet mp4.. 
                audio_path = video_path.replace("_temp.mp4", ".mp4")
                logger.info(f"Extracting audio features from {audio_path}")
                
                # Check if the audio file exists
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found at {audio_path}, trying alternative path")
                    # Try alternative path construction
                    base_name = os.path.basename(video_path)
                    if "_temp" in base_name:
                        alt_audio_path = video_path.replace("_temp", "")
                    else:
                        # Try to find the audio file in the same directory
                        dir_name = os.path.dirname(video_path)
                        base_name_no_ext = os.path.splitext(base_name)[0]
                        alt_audio_path = os.path.join(dir_name, f"{base_name_no_ext}.mp4")
                    
                    if os.path.exists(alt_audio_path):
                        audio_path = alt_audio_path
                        logger.info(f"Using alternative audio path: {audio_path}")
                    else:
                        logger.warning(f"Could not find audio file for {video_path}, skipping audio feature extraction")
                        raise FileNotFoundError(f"Audio file not found for {video_path}")
                
                # Extract audio features
                audio_features = self.extract_audio_features(audio_path)
                if len(audio_features.shape) == 3:
                    audio_features = audio_features[0]  # (seq_len, audio_feature_dim)
                cluster_ids = cluster_ids[:len(audio_features)]  # Truncate cluster IDs to match audio features length
                
                # Handle sequence length mismatch by truncating the longer sequence
                if len(audio_features) != len(cluster_ids):
                    logger.warning(f"Audio features length ({len(audio_features)}) does not match cluster IDs length ({len(cluster_ids)}) for video {video_path}")
                    # min_length = min(len(audio_features), len(cluster_ids))
                    # audio_features = audio_features[:min_length]
                    # frame_indices = frame_indices[:min_length]
                    # cluster_ids = cluster_ids[:min_length]
                    # logger.info(f"Truncated sequences to length {min_length}")
                    raise ValueError(f"Audio features length ({len(audio_features)}) does not match cluster IDs length ({len(cluster_ids)}) for video {video_path}")
                
                # Store in dictionary
                audio_features_dict[video_path] = audio_features.tolist()
                # logger.info(f"Successfully extracted audio features for {video_path}, shape: {audio_features.shape}")
                
                # Add to sequences dictionary
                sequences_dict[video_path] = {
                    "frame_indices": frame_indices,
                    "cluster_ids": cluster_ids
                }
            except Exception as e:
                logger.warning(f"Failed to extract audio features for {video_path}: {e}")
                raise
                # Still save the visual sequence even if audio extraction fails
                sequences_dict[video_path] = {
                    "frame_indices": frame_indices,
                    "cluster_ids": cluster_ids
                }
        
        # Save sequences as JSON
        sequences_path = os.path.join(output_dir, "video_sequences.json")
        with open(sequences_path, 'w') as f:
            json.dump(sequences_dict, f, indent=2)
        
        logger.info(f"Saved {len(video_paths)} video sequences to {sequences_path}")
        
        # Save audio features
        if audio_features_dict:
            audio_features_path = os.path.join(output_dir, "audio_features.npy")
            np.save(audio_features_path, audio_features_dict)
            logger.info(f"Saved audio features for {len(audio_features_dict)} videos to {audio_features_path}")
        else:
            logger.warning("No audio features were extracted, so audio_features.npy was not created")
        
        # Save metadata
        metadata = {
            "n_clusters": int(self.n_clusters),
            "model_name": self.model_name,
            "video_paths": video_paths,
            "audio_feature_type": self.audio_feature_type,
            "audio_feature_dim": 768 if self.audio_feature_type == "hubert_zh" else (13 if self.audio_feature_type == "mfcc" else 128),  # MFCC uses 13 coefficients, Mel uses 128 bins
            "device": self.device,
            "frame_interval": self.frame_interval,
            "max_frames": self.max_frames,
            "fps": self.fps,  # Default FPS used for audio feature extraction
            "audio_sampling_rate": 16000,  # Default sampling rate for audio processing
            "cluster_algorithm": "kmeans",
            "embedding_model": "CLIP",
            "embedding_dim": 512  # CLIP embedding dimension
        }
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_cluster_data(self, input_dir: str):
        """Load cluster centers and video sequences from files"""
        logger.info(f"Loading cluster data from {input_dir}")
        
        # Load cluster centers
        cluster_centers_path = os.path.join(input_dir, "cluster_centers.npy")
        self.cluster_centers = torch.from_numpy(np.load(cluster_centers_path)).to(self.device)
        self.n_clusters = self.cluster_centers.shape[0]
        
        # Load metadata
        metadata_path = os.path.join(input_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata.get("model_name", "ViT-B/32")
        video_paths = metadata.get("video_paths", [])
        
        # Load sequences
        sequences_path = os.path.join(input_dir, "video_sequences.json")
        with open(sequences_path, 'r') as f:
            sequences_dict = json.load(f)
        
        # Reconstruct video sequences
        self.video_sequences = {}
        for video_path in video_paths:
            if video_path in sequences_dict:
                frame_indices = sequences_dict[video_path]["frame_indices"]
                cluster_ids = sequences_dict[video_path]["cluster_ids"]
                # Reconstruct as (frame_idx, cluster_id) tuples
                self.video_sequences[video_path] = list(zip(frame_indices, cluster_ids))
        
        # Load audio features if available
        audio_features_path = os.path.join(input_dir, "audio_features.npy")
        if os.path.exists(audio_features_path):
            self.audio_features = np.load(audio_features_path, allow_pickle=True).item()
            logger.info(f"Loaded audio features for {len(self.audio_features)} videos")
        
        # Load representative frames
        frames_path = os.path.join(input_dir, "cluster_frames.npy")
        all_frames = np.load(frames_path, allow_pickle=True).item()
        
        # Load frames metadata
        frames_metadata_path = os.path.join(input_dir, "representative_frames.json")
        with open(frames_metadata_path, 'r') as f:
            representative_frames = json.load(f)
            
        # Reconstruct cluster_to_frames dictionary
        self.cluster_to_frames = {}
        for cluster_id, frame_array in all_frames.items():
            metadata = representative_frames[str(cluster_id)]
            self.cluster_to_frames[cluster_id] = [(
                metadata["video_path"],
                metadata["frame_idx"],
                frame_array
            )]
        
        logger.info(f"Cluster data loaded successfully with {self.n_clusters} clusters")
    
    def find_nearest_cluster_given_embedding(self, embedding: torch.Tensor) -> int:
        """Find the nearest cluster for a given embedding"""
        similarities = F.cosine_similarity(embedding, self.cluster_centers, dim=1)
        nearest_cluster = torch.argmax(similarities).item()
        return nearest_cluster

    def find_nearest_cluster(self, frame: np.ndarray) -> int:
        """Find the nearest cluster for a given frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        preprocessed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.encode_image(preprocessed_image)
            embedding = F.normalize(embedding, dim=-1)
        
        # Find nearest cluster center
        similarities = F.cosine_similarity(embedding, self.cluster_centers, dim=1)
        nearest_cluster = torch.argmax(similarities).item()
        
        return nearest_cluster

    @classmethod
    def from_pretrained(cls, model_dir, device="cuda"):
        """Load a pretrained ImageClusterGenerator from a directory."""
        # Load metadata to get parameters
        metadata_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance with parameters from metadata
        instance = cls(
            output_dir=None,  # Don't create output directories when loading
            n_clusters=metadata["n_clusters"],
            device=device,
            model_name=metadata["model_name"],
            audio_feature_type=metadata["audio_feature_type"],
            frame_interval=metadata["frame_interval"],
            max_frames=metadata["max_frames"],
            fps=metadata["fps"]
        )
        
        # Load cluster data
        instance.load_cluster_data(model_dir)
        
        return instance

def extract_frames_from_video(video_path: str, frame_interval: int = 1, verbose: bool = False) -> List[Tuple[str, int, np.ndarray]]:
    """Extract frames from a video file and return them as numpy arrays"""
    # logger.info(f"Extracting frames from video: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"video fps: {fps}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        logger.info(f"Video: {frame_count} frames at {fps} FPS")
    
    # Extract frames
    frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame at specified intervals
        if frame_idx % frame_interval == 0:
            frames.append((video_path, frame_idx, frame))
            
        frame_idx += 1
    
    cap.release()
    if verbose:
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
    
    return frames

def process_videos(video_pattern: str, output_dir: str, n_clusters: int = 100, 
                  frame_interval: int = 1, device: str = 'cuda', model_name: str = "ViT-B/32",
                  max_frames: int = 1000, audio_feature_type: str = "hubert_zh", clear_output_dir: bool = False, fps: int = 25):
    """Process videos matching the pattern and generate clusters"""
    # Find all videos matching the pattern
    video_paths = glob.glob(video_pattern)
    if not video_paths:
        logger.warning(f"No videos found matching pattern: {video_pattern}")
        return
    
    logger.info(f"Found {len(video_paths)} videos to process")
    video_paths = list(sorted(video_paths))
    
    # Create output directory for clusters
    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)
    
    # Extract frames from all videos
    all_frames = []
    progbar = tqdm(video_paths, desc="Extracting frames from videos", postfix={"total_frames": 0})
    for video_path in progbar:
        frames = extract_frames_from_video(video_path, frame_interval, verbose=False)
        all_frames.extend(frames)
        # Update progress bar with total frames count
        tqdm.write(f"--- Total frames extracted: {len(all_frames)}")
        # Update the postfix in the progress bar
        progbar.set_postfix(total_frames=len(all_frames))
        # Check if we've reached the maximum number of frames
        if len(all_frames) >= max_frames:
            logger.info(f"Reached maximum frame limit ({max_frames}). Stopping extraction.")
            # Print the remaining videos that would be excluded
            remaining_videos = video_paths[video_paths.index(video_path)+1:]
            logger.info(f"The following {len(remaining_videos)} videos would be excluded due to max_frames limit:")
            for excluded_video in remaining_videos:
                logger.info(f"  {excluded_video}")
            break
    
    logger.info(f"Extracted {len(all_frames)} frames in total")
    
    # Initialize cluster generator
    cluster_generator = ImageClusterGenerator(
        output_dir=output_dir, 
        n_clusters=n_clusters, 
        device=device, 
        model_name=model_name, 
        audio_feature_type=audio_feature_type, 
        clear_output_dir=clear_output_dir,
        frame_interval=frame_interval,
        max_frames=max_frames,
        fps=fps
    )
    
    # Extract embeddings and cluster images
    embeddings = cluster_generator.extract_image_embeddings(all_frames)
    cluster_generator.cluster_images(embeddings, all_frames)
    
    # Create visualizations
    cluster_generator.generate_detokenized_videos(output_dir)
    cluster_generator.create_cluster_visualization(clusters_dir)
    
    # Save cluster data (including audio features)
    cluster_generator.save_cluster_data(output_dir)
    
    logger.info(f"Processing complete. Results saved to {output_dir}")
    
    return cluster_generator

def main():
    parser = argparse.ArgumentParser(description='Generate image clusters from videos')
    parser.add_argument('--video_pattern', type=str, required=True, help='Pattern to match video files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters to generate')
    parser.add_argument('--frame_interval', type=int, default=1, help='Interval between frames to extract')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--model_name', type=str, default="ViT-B/32", help='CLIP model to use')
    parser.add_argument('--max_frames', type=int, default=1000, help='Maximum number of frames to process')
    parser.add_argument('--audio_feature_type', type=str, default="hubert_zh", 
                        choices=["hubert_zh", "mfcc", "mel"], 
                        help='Type of audio features to extract (default: hubert_zh)')
    parser.add_argument('--clear_output_dir', action='store_true', 
                        help='Clear the output directory before processing')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second for audio feature extraction (default: 25)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process videos
    process_videos(
        video_pattern=args.video_pattern,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        frame_interval=args.frame_interval,
        device=args.device,
        model_name=args.model_name,
        max_frames=args.max_frames,
        audio_feature_type=args.audio_feature_type,
        clear_output_dir=args.clear_output_dir,
        fps=args.fps
    )

if __name__ == "__main__":
    main()


