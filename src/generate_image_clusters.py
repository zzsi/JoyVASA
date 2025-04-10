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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageClusterGenerator:
    def __init__(self, device='cuda', n_clusters=100, model_name="ViT-B/32"):
        """Initialize the image cluster generator with CLIP model and clustering parameters"""
        self.device = device
        self.n_clusters = n_clusters
        self.model_name = model_name
        
        logger.info(f"Initializing ImageClusterGenerator on device: {device}")
        logger.info(f"Loading CLIP model: {model_name}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        logger.info("CLIP model loaded successfully")
        
        # Initialize clustering components
        self.kmeans = None
        self.cluster_centers = None
        self.cluster_to_frames = {}  # Maps cluster ID to list of (video_path, frame_idx) tuples
        self.frame_to_cluster = {}   # Maps frame key to cluster ID
        self.video_sequences = {}    # Maps video path to list of (frame_idx, cluster_id) tuples
        
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
            
            # Create grid using torchvision
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
            output_path = os.path.join(videos_dir, f"{video_name}_detokenized.mp4")
            
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
            
            # Use images2video function to create the video
            images2video(frames, wfp=output_path, fps=fps, image_mode='bgr')
            # also save the frames as an image grid
            frames_torch = [torch.from_numpy(frame).permute(2, 0, 1) for frame in frames]
            # print(frames_torch[0].shape, "this is the shape of each frame")
            # resize to 64x64 for each frame
            frames_torch = [F.interpolate(frame.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0) for frame in frames_torch]
            frames_torch = frames_torch[:36]
            grid_image = torchvision.utils.make_grid(
                frames_torch,
                nrow=6,
                padding=5,
                normalize=False
            )
            # print("grid_image.shape", grid_image.shape)
            grid_image = torchvision.transforms.ToPILImage()(grid_image)
            grid_image.save(os.path.join(output_dir, f"{video_name}_detokenized_grid.png"))
            logger.info(f"Generated detokenized video: {output_path}")

            cnt += 1
            if cnt >= max_videos:
                break
        
        logger.info(f"All detokenized videos saved to {videos_dir}")
    
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
        
        # Create a dictionary to store sequences
        sequences_dict = {}
        for video_path in video_paths:
            # Convert (frame_idx, cluster_id) tuples to a dictionary
            # Convert numpy types to Python native types for JSON serialization
            frame_indices = [int(idx) for idx, _ in self.video_sequences[video_path]]
            cluster_ids = [int(cid) for _, cid in self.video_sequences[video_path]]
            sequences_dict[video_path] = {
                "frame_indices": frame_indices,
                "cluster_ids": cluster_ids
            }
        
        # Save sequences as JSON
        sequences_path = os.path.join(output_dir, "video_sequences.json")
        with open(sequences_path, 'w') as f:
            json.dump(sequences_dict, f, indent=2)
        
        logger.info(f"Saved {len(video_paths)} video sequences to {sequences_path}")
        
        # Save metadata
        metadata = {
            "n_clusters": int(self.n_clusters),
            "model_name": self.model_name,
            "video_paths": video_paths
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
        
        logger.info(f"Cluster data loaded successfully with {self.n_clusters} clusters")
    
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

def extract_frames_from_video(video_path: str, frame_interval: int = 1, verbose: bool = False) -> List[Tuple[str, int, np.ndarray]]:
    """Extract frames from a video file and return them as numpy arrays"""
    logger.info(f"Extracting frames from video: {video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
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
                  frame_interval: int = 5, device: str = 'cuda', model_name: str = "ViT-B/32",
                  max_frames: int = 1000):
    """Process videos matching the pattern and generate clusters"""
    # Find all videos matching the pattern
    video_paths = glob.glob(video_pattern)
    if not video_paths:
        logger.warning(f"No videos found matching pattern: {video_pattern}")
        return
    
    logger.info(f"Found {len(video_paths)} videos to process")
    
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
            break
    
    logger.info(f"Extracted {len(all_frames)} frames in total")
    
    # Initialize cluster generator
    cluster_generator = ImageClusterGenerator(device=device, n_clusters=n_clusters, model_name=model_name)
    
    # Extract embeddings and cluster images
    embeddings = cluster_generator.extract_image_embeddings(all_frames)
    cluster_generator.cluster_images(embeddings, all_frames)
    
    # Create visualizations
    cluster_generator.generate_detokenized_videos(output_dir)
    cluster_generator.create_cluster_visualization(clusters_dir)
    
    # Save cluster data
    cluster_generator.save_cluster_data(output_dir)
    
    logger.info(f"Processing complete. Results saved to {output_dir}")
    
    return cluster_generator

def main():
    parser = argparse.ArgumentParser(description="Generate image clusters from videos")
    parser.add_argument("--video_pattern", type=str, required=False, help="Glob pattern for video files",
                        default="data/batch_generated_videos/bithuman_coach/*_temp.mp4")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory for results",
                        default="data/batch_generated_videos/bithuman_coach_image_clusters")
    parser.add_argument("--n_clusters", type=int, default=128, help="Number of clusters")
    parser.add_argument("--frame_interval", type=int, default=1, help="Extract every Nth frame")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--model_name", type=str, default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--max_frames", type=int, default=90000, help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    # Process videos and generate clusters
    process_videos(
        video_pattern=args.video_pattern,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        frame_interval=args.frame_interval,
        device=args.device,
        model_name=args.model_name,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()


