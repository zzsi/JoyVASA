# Suppress FutureWarning and UserWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
import librosa
import cv2
from tqdm import tqdm
import logging
import os
from .modules.wav2vec2 import Wav2Vec2Model
from src.utils.video import add_audio_to_video
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastLipSync:
    def __init__(self, device='cpu', n_clusters=1000, n_motions=100, fps=25, pad_mode='zero'):
        """Initialize FastLipSync with wav2vec2 model and clustering parameters"""
        self.device = device
        logger.info(f"Initializing FastLipSync on device: {device}")
        self.n_clusters = n_clusters
        self.n_motions = n_motions
        self.fps = fps
        self.pad_mode = pad_mode
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        
        logger.info("Loading wav2vec2 model...")
        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_encoder.to(device)
        logger.info("wav2vec2 model loaded successfully")
        
        self.kmeans = None
        self.cluster_to_frames = {}  # Maps cluster ID to list of (frame_idx, frame) tuples
        self.cluster_centers = None
        
    def preprocess_audio(self, audio: torch.Tensor, n_frames: int = None, verbose: bool = True) -> torch.Tensor:
        """Preprocess audio with proper padding and subdivision"""
        if verbose:
            logger.info("Preprocessing audio...")
        current_audio_samples = len(audio)
        
        if n_frames is not None:
            # Training mode: adjust audio length to match video frames
            expected_audio_samples = int(16000 * n_frames / self.fps)
            if verbose:
                logger.info(f"Audio samples - Current: {current_audio_samples}, Expected: {expected_audio_samples}")
            
            # Trim or pad audio to match expected length
            if current_audio_samples > expected_audio_samples:
                if verbose:
                    logger.info(f"Trimming {current_audio_samples - expected_audio_samples} audio samples")
                audio = audio[:expected_audio_samples]
            elif current_audio_samples < expected_audio_samples:
                padding_size = expected_audio_samples - current_audio_samples
                if verbose:
                    logger.info(f"Padding {padding_size} audio samples")
                if self.pad_mode == 'zero':
                    padding_value = 0
                elif self.pad_mode == 'replicate':
                    padding_value = audio[-1]
                else:
                    raise ValueError(f'Unknown pad mode: {self.pad_mode}')
                audio = F.pad(audio, (0, padding_size), value=padding_value)
        else:
            # Generation mode: use audio as is
            if verbose:
                logger.info(f"Using original audio length: {current_audio_samples} samples")
            
        return audio
    
    def extract_audio_features(self, audio: torch.Tensor, n_frames: int = None, verbose: bool = True) -> torch.Tensor:
        """Extract audio features using wav2vec2 with proper preprocessing"""
        if verbose:
            logger.info("Extracting audio features...")
        # Preprocess audio
        audio = self.preprocess_audio(audio, n_frames, verbose)
        
        # Extract features
        with torch.no_grad():
            features = self.audio_encoder(audio.unsqueeze(0), output_fps=self.fps).last_hidden_state
            features = features.squeeze(0)  # Remove batch dimension
        
        if verbose:
            logger.info(f"Extracted features shape: {features.shape}")
        return features
    
    def train_with_multiple_videos(self, audio_paths: List[str], video_paths: List[str], fps: int = 25):
        """Train the model on multiple audio-video pairs"""
        # Gather all video frames and all audio features    
        video_frames = []
        audio_features = []
        for audio_path, video_path in tqdm(zip(audio_paths, video_paths), desc="Gathering video frames and audio features", total=len(audio_paths)):
            extra_video_frames, _ = read_video_frames(video_path, verbose=False)
            video_frames.extend(extra_video_frames)
            # Load and preprocess audio
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            audio = torch.from_numpy(audio).to(self.device)
            extra_audio_features = self.extract_audio_features(audio, n_frames=len(extra_video_frames), verbose=False)
            audio_features.extend(extra_audio_features)
        
        # convert to numpy arrays
        audio_features = np.stack(audio_features)
            
        # Perform clustering on audio features
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters using {audio_features.shape[0]} audio features...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(audio_features)
        self.cluster_centers = torch.from_numpy(self.kmeans.cluster_centers_).to(self.device)
            
        # Store one representative frame per cluster
        logger.info("Storing representative frames for each cluster...")
        self.cluster_to_frames = {}  # Maps cluster ID to a single representative frame
        for label, frame in zip(cluster_labels, video_frames):
            if label not in self.cluster_to_frames:
                # Store only the first frame we see for each cluster
                self.cluster_to_frames[label] = frame
        
        # Log cluster statistics
        logger.info(f"Stored {len(self.cluster_to_frames)} representative frames")
            
    
    def train(self, audio_path: str, video_frames: List[np.ndarray], fps: int = 25):
        """Train the model on audio-video pairs"""
        logger.info(f"Starting training with audio: {audio_path}")
        self.fps = fps
        
        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = torch.from_numpy(audio).to(self.device)
        logger.info(f"Loaded audio shape: {audio.shape}")
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio, len(video_frames))
        
        # Perform clustering on audio features
        logger.info(f"Performing K-means clustering with {self.n_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(audio_features.cpu().numpy())
        self.cluster_centers = torch.from_numpy(self.kmeans.cluster_centers_).to(self.device)
        
        # Store one representative frame per cluster
        logger.info("Storing representative frames for each cluster...")
        self.cluster_to_frames = {}  # Maps cluster ID to a single representative frame
        for label, frame in zip(cluster_labels, video_frames):
            if label not in self.cluster_to_frames:
                # Store only the first frame we see for each cluster
                self.cluster_to_frames[label] = frame
        
        # Log cluster statistics
        logger.info(f"Stored {len(self.cluster_to_frames)} representative frames")
            
    def generate(self, audio_path: str) -> List[np.ndarray]:
        """Generate video frames from audio"""
        logger.info(f"Generating video from audio: {audio_path}")
        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = torch.from_numpy(audio).to(self.device)
        logger.info(f"Loaded audio shape: {audio.shape}")
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio)
        
        # Find nearest cluster for each audio feature
        generated_frames = []
        logger.info("Generating frames...")
        for feature in tqdm(audio_features, desc="Generating frames"):
            # Find nearest cluster center
            distances = F.cosine_similarity(feature.unsqueeze(0), self.cluster_centers, dim=1)
            nearest_cluster = torch.argmax(distances).item()
            
            # Get the representative frame for this cluster
            if nearest_cluster in self.cluster_to_frames:
                frame = self.cluster_to_frames[nearest_cluster]
                generated_frames.append(frame)
            else:
                # If cluster is empty, use a default frame
                default_cluster = list(self.cluster_to_frames.keys())[0]
                generated_frames.append(self.cluster_to_frames[default_cluster])
                
        logger.info(f"Generated {len(generated_frames)} frames")
        return generated_frames
    
    def save_video(self, frames: List[np.ndarray], output_path: str):
        """Save generated frames as video"""
        if not frames:
            logger.warning("No frames to save")
            return
            
        logger.info(f"Saving video to: {output_path}")
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
        logger.info("Video saved successfully")

    def save_model(self, model_path: str):
        """Save the model to a file"""
        logger.info(f"Saving model to: {model_path}")
        model_data = {
            'cluster_centers': self.cluster_centers.cpu().numpy(),
            'cluster_to_frames': self.cluster_to_frames,
            'kmeans': self.kmeans,
            'n_clusters': self.n_clusters,
            'fps': self.fps,
            'pad_mode': self.pad_mode
        }
        torch.save(model_data, model_path)
        logger.info("Model saved successfully")

    def load_model(self, model_path: str):
        """Load the model from a file"""
        logger.info(f"Loading model from: {model_path}")
        model_data = torch.load(model_path)
        
        # Load model parameters
        self.n_clusters = model_data['n_clusters']
        self.fps = model_data['fps']
        self.pad_mode = model_data['pad_mode']
        
        # Load cluster data
        self.cluster_centers = torch.from_numpy(model_data['cluster_centers']).to(self.device)
        self.cluster_to_frames = model_data['cluster_to_frames']
        self.kmeans = model_data['kmeans']
        
        logger.info(f"Model loaded successfully with {self.n_clusters} clusters")

    def generate_with_beam_search(self, audio_path: str, beam_width: int = 5, 
                                  singleton_weight: float = 1.0, pairwise_weight: float = 0.5) -> List[np.ndarray]:
        """Generate video frames from audio using beam search optimization
        
        Args:
            audio_path: Path to audio file
            beam_width: Width of beam for search
            singleton_weight: Weight for audio-visual alignment cost
            pairwise_weight: Weight for temporal smoothness cost
            
        Returns:
            List of generated frames
        """
        logger.info(f"Generating video from audio with beam search: {audio_path}")
        
        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio = torch.from_numpy(audio).to(self.device)
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio)
        n_frames = len(audio_features)
        
        logger.info(f"Optimizing sequence of {n_frames} frames with beam search (width={beam_width})")
        
        # Calculate singleton costs: distance from each audio feature to each cluster center
        # Shape: [n_frames, n_clusters]
        singleton_costs = torch.zeros((n_frames, self.n_clusters))
        
        for i, feature in enumerate(tqdm(audio_features, desc="Computing singleton costs")):
            # Compute negative cosine similarity (lower is better)
            similarities = F.cosine_similarity(feature.unsqueeze(0), self.cluster_centers, dim=1)
            singleton_costs[i] = -similarities  # Convert to cost (negative similarity)
        
        # Precompute pairwise costs: visual difference between all pairs of frames
        # Shape: [n_clusters, n_clusters]
        pairwise_costs = torch.zeros((self.n_clusters, self.n_clusters))
        
        logger.info("Computing pairwise costs between frames...")
        for i in tqdm(range(self.n_clusters), desc="Computing pairwise costs"):
            if i not in self.cluster_to_frames:
                continue
                
            frame_i = torch.from_numpy(self.cluster_to_frames[i]).float()
            
            for j in range(self.n_clusters):
                if j not in self.cluster_to_frames:
                    continue
                    
                frame_j = torch.from_numpy(self.cluster_to_frames[j]).float()
                
                # Compute frame difference (L2 distance between frames)
                # Could use more sophisticated measures focused on lip region
                diff = torch.mean((frame_i - frame_j) ** 2)
                pairwise_costs[i, j] = diff
        
        # Normalize costs to be in similar ranges
        singleton_costs = (singleton_costs - singleton_costs.mean()) / singleton_costs.std()
        pairwise_costs = (pairwise_costs - pairwise_costs.mean()) / pairwise_costs.std()
        
        # Initialize beam search
        # Each beam element is (sequence, total_cost)
        beams = [([], 0.0)]
        
        # Perform beam search
        for frame_idx in tqdm(range(n_frames), desc="Beam search progress"):
            candidates = []
            
            for sequence, cost in beams:
                for cluster_idx in range(self.n_clusters):
                    if cluster_idx not in self.cluster_to_frames:
                        continue
                        
                    # Compute new cost with this cluster
                    new_cost = cost + singleton_weight * singleton_costs[frame_idx, cluster_idx].item()
                    
                    # Add pairwise cost if this isn't the first frame
                    if sequence:
                        prev_cluster = sequence[-1]
                        new_cost += pairwise_weight * pairwise_costs[prev_cluster, cluster_idx].item()
                    
                    # Create new candidate
                    new_sequence = sequence + [cluster_idx]
                    candidates.append((new_sequence, new_cost))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1])  # Sort by cost (lower is better)
            beams = candidates[:beam_width]
        
        # Get best sequence
        best_sequence, best_cost = beams[0]
        logger.info(f"Found best sequence with cost: {best_cost}")
        
        # Convert sequence to frames
        generated_frames = [self.cluster_to_frames[cluster_idx] for cluster_idx in best_sequence]
        
        return generated_frames


def read_video_frames(video_path: str, verbose: bool = True) -> Tuple[List[np.ndarray], float]:
    """Read video frames from video file and return frames and fps"""
    if verbose:
        logger.info(f"Reading video frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if verbose:
        logger.info(f"Read {len(frames)} frames at {fps} FPS")
    return frames, fps



if __name__ == "__main__":
    """
    Example usage:
    python -m src.fast_lipsync --mode train --train_audio_path audio.wav --train_video_path video.mp4 --model_path model.pth
    python -m src.fast_lipsync --mode generate --input_audio_path audio.wav --model_path model.pth --output_path output.mp4
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="generate", choices=["train", "generate"])
    parser.add_argument("--model_path", type=str, default="fast_lipsync_dating_coach.pth", help="Path to the model file")
    # parser.add_argument("--train_dir", type=str, default="data/batch_generated_videos/bithuman_coach", help="Path to the directory containing audio and video files for training")
    parser.add_argument("--train_dir", type=str, default="data/conversations_joyvasa_videos/bithuman_coach2", help="Path to the directory containing audio and video files for training")
    parser.add_argument("--train_audio_path", type=str, default="data/raw-video.wav", help="Path to the audio file for training")
    parser.add_argument("--train_video_path", type=str, default="animations/joyvasa_005_raw-video_lip_temp.mp4", help="Path to the video file for training")
    parser.add_argument("--n_clusters", type=int, default=1000, help="Number of clusters for clustering")
    # parser.add_argument("--input_audio_path", type=str, required=False, help="Path to the input audio file for generation", default="assets/examples/audios/joyvasa_001.wav")
    parser.add_argument("--input_audio_path", type=str, required=False, help="Path to the input audio file for generation", default="data/conversations/002e4b0241534fc6f83d62452488bf1c7c05bc2ba69d840947a41d9a4727ae55_tts-1_nova.wav")
    parser.add_argument("--output_path", type=str, help="Path to the output video file", default="data/tmp.mp4")
    parser.add_argument("--fps", type=int, default=25, help="FPS of the output video for generation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--use_beam_search", action="store_true", help="Use beam search for temporal smoothing")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--singleton_weight", type=float, default=1.0, help="Weight for audio-visual alignment")
    parser.add_argument("--pairwise_weight", type=float, default=0.5, help="Weight for temporal smoothness")
    args = parser.parse_args()

    if args.mode == "generate" and args.output_path is None:
        import datetime

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = (
            args.input_audio_path.replace(".mp3", "_fast_lipsync.mp4")
            .replace(".wav", "_fast_lipsync.mp4")
            .replace("fast_lipsync.mp4", f"fast_lipsync_{current_time}.mp4")
        )

    logger.info(f"Running in {args.mode} mode on device: {args.device}")

    if args.mode == "train":
        fast_lipsync = FastLipSync(device=args.device, n_clusters=args.n_clusters)
        # Read video frames from video file
        
        if args.train_dir:
            import glob

            wildcard = os.path.join(args.train_dir, "*_lip.mp4")
            logger.info(f"Training with multiple videos in directory: {wildcard}")
            video_mp4_paths = list(glob.glob(wildcard))[:1000]
            audio_paths = [x.replace("_temp.mp4", ".mp4") for x in video_mp4_paths]
            assert len(video_mp4_paths) > 0, f"No video files found in the training directory: {wildcard}"
            assert len(audio_paths) > 0, f"No audio files found in the training directory: {wildcard}"
            fast_lipsync.train_with_multiple_videos(
                audio_paths=audio_paths,
                video_paths=video_mp4_paths,
                fps=args.fps
            )
        else:
            video_frames, fps = read_video_frames(args.train_video_path)
            fast_lipsync.train(args.train_audio_path, video_frames, fps)
            fast_lipsync.save_video(frames=video_frames, output_path=args.output_path)
        fast_lipsync.save_model(model_path=args.model_path)
    elif args.mode == "generate":
        # Make sure the input audio file exists
        assert os.path.exists(args.input_audio_path), f"Input audio file does not exist: {args.input_audio_path}"
        fast_lipsync = FastLipSync(device=args.device)
        fast_lipsync.load_model(model_path=args.model_path)
        
        if args.use_beam_search:
            frames = fast_lipsync.generate_with_beam_search(
                audio_path=args.input_audio_path,
                beam_width=args.beam_width,
                singleton_weight=args.singleton_weight,
                pairwise_weight=args.pairwise_weight
            )
        else:
            frames = fast_lipsync.generate(audio_path=args.input_audio_path)
            
        fast_lipsync.save_video(frames=frames, output_path=args.output_path)
        # add audio to the video
        add_audio_to_video(
            silent_video_path=args.output_path,
            audio_video_path=args.input_audio_path,
            output_video_path=args.output_path.replace(".mp4", "_with_audio.mp4")
        )

