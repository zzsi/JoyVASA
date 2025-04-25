"""
Conditional GPT that generates image cluster sequences conditioned on audio features.

The model takes both audio features and previous visual tokens as input to predict the next visual token.
Architecture:
1. Audio features are projected to the same dimension as token embeddings
2. Visual tokens are embedded via standard token embeddings
3. The sequences are combined and processed through transformer blocks
4. The model predicts the next visual token
"""

from dataclasses import asdict, dataclass
import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from src.lip_gpt0 import GPT, GPTConfig, NanoGPTTrainingPipeline, load_video_cluster_id_sequences
from tqdm import tqdm
from glob import glob



def load_audio_visual_token_data(data_dir: str, max_files: int = None):
    """
    Load audio-to-visual token data.
    """
    npy_files = glob(os.path.join(data_dir, "*.npz"))
    assert len(npy_files) > 0, f"No npy files found in {data_dir}"
    audio_visual_token_data = []
    if max_files is not None:
        npy_files = npy_files[:max_files]
    for npy_file in tqdm(npy_files):
        loaded = np.load(npy_file, allow_pickle=True)
        audio_features = loaded["audio_features"]
        visual_tokens = loaded["visual_cluster_ids"]
        assert len(audio_features.shape) == 3
        # truncate visual tokens to the same length as audio features
        visual_tokens = visual_tokens[:audio_features.shape[1]]
        audio_visual_token_data.append((audio_features, visual_tokens))
    return audio_visual_token_data


@dataclass
class AudioGPTConfig(GPTConfig):
    """Configuration for the AudioGPT model."""
    audio_feature_dim: int = 13  # dimension of input audio features
    audio_proj_dim: int = None  # projection dimension for audio features (will match n_embd)
    
    def __post_init__(self):
        # super().__post_init__()
        # Ensure audio_proj_dim matches n_embd if not explicitly set
        if self.audio_proj_dim is None:
            self.audio_proj_dim = self.n_embd


class AudioProjection(nn.Module):
    """Projects audio features to the model's embedding dimension."""
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(config.audio_feature_dim, config.audio_proj_dim)
        print(f"Audio projection dim: {config.audio_proj_dim}")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Reshape the input if needed to match the expected dimensions
        if x.shape[-1] != self.projection.in_features:
            # If the input dimension doesn't match, we need to adjust the projection
            # This is a fallback in case the audio feature dimension doesn't match what the model expects
            print(f"Warning: Audio feature dimension mismatch. Expected {self.projection.in_features}, got {x.shape[-1]}")
            # Create a new projection layer with the correct input dimension
            self.projection = nn.Linear(x.shape[-1], self.projection.out_features).to(x.device)
        return self.dropout(self.projection(x))


class ConditionalGPT(GPT):
    """GPT model that conditions on audio features."""
    
    def __init__(self, config):
        super().__init__(config)
        self.audio_projection = AudioProjection(config)
        # Add a projection layer to reduce concatenated embeddings back to n_embd
        self.combined_projection = nn.Sequential(
            nn.Linear(config.n_embd + config.audio_proj_dim, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        # Add a direct audio-to-visual projection for simplified mode
        self.direct_audio_projection = nn.Sequential(
            nn.Linear(config.audio_proj_dim, config.audio_proj_dim),
            nn.ReLU(),
            nn.Linear(config.audio_proj_dim, config.vocab_size)
        )
        
    def forward(self, visual_ids, audio_features, targets=None, mode="gpt"):
        """
        Forward pass with two modes:
        - "gpt": Full GPT functionality using both visual and audio context
        - "direct": Simplified audio-to-visual prediction using only current audio frame
        """
        device = visual_ids.device
        b, t = visual_ids.size()
        assert audio_features.size()[:2] == (b, t), f"Audio features shape {audio_features.shape} doesn't match visual ids shape {visual_ids.shape}"
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        if mode == "direct":
            # Simplified mode: use only current audio frame to predict next visual token
            audio_emb = self.audio_projection(audio_features[:, -1:])  # (b, 1, n_embd)
            logits = self.direct_audio_projection(audio_emb.squeeze(1))  # (b, vocab_size)
            
            if targets is not None:
                loss = F.cross_entropy(logits, targets[:, -1], ignore_index=-1)
            else:
                loss = None
                
            return logits, loss
            
        else:  # mode == "gpt"
            # Full GPT functionality
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(visual_ids)
            if self.config.audio_proj_dim > 0:
                audio_emb = self.audio_projection(audio_features)
                combined_emb = self.combined_projection(torch.cat([audio_emb, tok_emb], dim=-1))
            else:
                combined_emb = tok_emb
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(combined_emb + pos_emb)
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            
            if targets is not None:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                logits = self.lm_head(x[:, [-1], :])
                loss = None
                
            return logits, loss

    def generate(self, visual_ids, audio_features, temperature=1.0, top_k=None, mode="gpt"):
        """
        Generate visual tokens with two modes:
        - "gpt": Full GPT generation using both visual and audio context
        - "direct": Simplified generation using only current audio frame

        Args:
            visual_ids: (b, t) tensor of visual tokens as the prompt
            audio_features: (b, T, audio_feature_dim) tensor of audio features, T is the number of audio frames and can be larger than the number of visual tokens in the prompt
            temperature: float, temperature for sampling
            top_k: int, top-k for sampling
            mode: str, "gpt" or "direct"
        """
        max_new_tokens = audio_features.shape[1]
        if mode == "direct":
            for i in range(max_new_tokens):
                # Get current audio frame
                current_audio = audio_features[:, i:i+1]  # (b, 1, audio_feature_dim)
                
                # Project audio and get logits
                audio_emb = self.audio_projection(current_audio)
                logits = self.direct_audio_projection(audio_emb.squeeze(1))
                logits = logits / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply softmax and sample
                probs = nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append sampled index to the running sequence
                visual_ids = torch.cat((visual_ids, idx_next), dim=1)
                
        else:  # mode == "gpt"
            for i in range(max_new_tokens):
                # Crop context if needed
                idx_cond = visual_ids if visual_ids.size(1) <= self.config.block_size else visual_ids[:, -self.config.block_size:]
                audio_cond = audio_features[:, 0:i+1]
                # truncate audio_cond to the same length as idx_cond, keep the last frames
                if audio_cond.size(1) > idx_cond.size(1):
                    audio_cond = audio_cond[:, -idx_cond.size(1):]
                
                assert audio_cond.size(1) == idx_cond.size(1), f"audio_cond.size(1) ({audio_cond.size(1)}) does not match idx_cond.size(1) ({idx_cond.size(1)})"

                # Forward pass
                logits, _ = self(idx_cond, audio_cond, mode="gpt")
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply softmax and sample
                probs = nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append sampled index to the running sequence
                visual_ids = torch.cat((visual_ids, idx_next), dim=1)
                
        return visual_ids


class AudioVisualDatasetBuilder:
    """Handles loading and preprocessing of aligned audio-visual data."""
    def __init__(self, json_path: str = None, audio_features_path: str = None, data_dir: str = None, 
                 max_sequence_length: int = 80, start_token: int = 767, max_files: int = None):
        """
        Initialize the dataset builder.
        
        Args:
            json_path: Path to JSON file containing video cluster ID sequences
            audio_features_path: Path to numpy file containing audio features
            data_dir: Directory containing numpy files with audio-visual token data
            max_sequence_length: Maximum length of sequences to process
            start_token: Token to use at the start of sequences
            max_files: Maximum number of files to process
        """
        if data_dir is not None:
            # Load data from numpy files
            audio_visual_token_data = load_audio_visual_token_data(data_dir, max_files)
            self.video_cluster_id_sequences = [data[1] for data in audio_visual_token_data]  # visual tokens
            self.audio_features = [data[0] for data in audio_visual_token_data]  # audio features
            if len(self.audio_features[0].shape) == 3:
                self.audio_features = [audio_features[0] for audio_features in self.audio_features]
        else:
            # Load data from JSON and numpy files
            assert json_path is not None and audio_features_path is not None, \
                "Both json_path and audio_features_path must be provided when data_dir is not specified"
            with open(json_path, "r") as f:
                self.video_cluster_id_sequences = json.load(f)
            # Allow pickle=True to load object arrays
            self.audio_features = np.load(audio_features_path, allow_pickle=True).item()  # Should be aligned with video sequences

        self.max_sequence_length = max_sequence_length
        self.start_token = start_token
        self.max_files = max_files

        # Convert audio_features dict to a list aligned with video_cluster_id_sequences
        # Process video sequences and audio features to ensure alignment
        audio_features_list = []
        visual_sequences_list = []
        
        if data_dir is None:
            # Process data from JSON and numpy files
            for video_path, video_data in self.video_cluster_id_sequences.items():
                if video_path in self.audio_features:
                    # Extract the cluster IDs from the video data
                    visual_seq = video_data["cluster_ids"]
                    audio_seq = self.audio_features[video_path]
                    
                    # Check if lengths match before adding
                    if len(visual_seq) == len(audio_seq):
                        visual_sequences_list.append(visual_seq)
                        audio_features_list.append(audio_seq)
                    else:
                        print(f"Warning: Skipping {video_path} - Visual sequence length ({len(visual_seq)}) does not match audio sequence length ({len(audio_seq)})")
                else:
                    print(f"Warning: Skipping {video_path} - Missing audio features")
            
            # Replace the original data with the aligned lists
            self.video_cluster_id_sequences = visual_sequences_list
            self.audio_features = audio_features_list
        else:
            # Data is already in the correct format from numpy files
            visual_sequences_list = self.video_cluster_id_sequences
            audio_features_list = self.audio_features

        # Apply max_files limit
        self.video_cluster_id_sequences = self.video_cluster_id_sequences[:self.max_files]
        self.audio_features = self.audio_features[:self.max_files]
        
        # Verify that we have data to work with
        assert len(self.video_cluster_id_sequences) > 0, "No valid aligned audio-visual sequences found"
        # Break down long sequences into smaller chunks
        equalized_visual_sequences = []
        equalized_audio_sequences = []
        
        print(f"There are {len(self.video_cluster_id_sequences)} sequences in the dataset")
        for i, (visual_seq, audio_seq) in enumerate(zip(self.video_cluster_id_sequences, self.audio_features)):
            # Verify that each audio sequence has the same length as its corresponding visual sequence
            assert len(visual_seq) == len(audio_seq), \
                f"Sequence {i}: Visual sequence length ({len(visual_seq)}) does not match audio sequence length ({len(audio_seq)})."
            
            if len(visual_seq) == self.max_sequence_length:
                equalized_visual_sequences.append(visual_seq)
                equalized_audio_sequences.append(audio_seq)
            elif len(visual_seq) > self.max_sequence_length:
                # Split long sequence into chunks of max_sequence_length
                for j in range(0, len(visual_seq), self.max_sequence_length):
                    visual_chunk = visual_seq[j:j + self.max_sequence_length]
                    audio_chunk = audio_seq[j:j + self.max_sequence_length]
                    
                    if len(visual_chunk) == self.max_sequence_length:  # Only keep chunks that are exactly the max length
                        equalized_visual_sequences.append(visual_chunk)
                        equalized_audio_sequences.append(audio_chunk)
            else:
                print(f"Warning: Skipping sequence {i} - Visual sequence length ({len(visual_seq)}) is less than max sequence length ({self.max_sequence_length})")
            
        print(f"There are {len(equalized_visual_sequences)} sequences in the dataset after equalization")
        
        # Convert sequences to numpy arrays
        equalized_visual_sequences = [np.array(seq) for seq in equalized_visual_sequences]
        equalized_audio_sequences = [np.array(seq) for seq in equalized_audio_sequences]

        # append start token to the beginning of each sequence
        equalized_visual_sequences = [np.concatenate([[self.start_token], seq]) for seq in equalized_visual_sequences]
        
        # Split into train and validation sets
        n_train = int(len(equalized_visual_sequences) * 0.8)
        
        # Split visual sequences
        self.train_video_cluster_id_sequences = equalized_visual_sequences[:n_train]
        self.val_video_cluster_id_sequences = equalized_visual_sequences[n_train:]
        self.train_video_cluster_id_sequences = np.array(self.train_video_cluster_id_sequences)
        self.val_video_cluster_id_sequences = np.array(self.val_video_cluster_id_sequences)
        
        # Split audio sequences
        self.train_audio_sequences = equalized_audio_sequences[:n_train]
        self.val_audio_sequences = equalized_audio_sequences[n_train:]
        self.train_audio_sequences = np.array(self.train_audio_sequences)
        self.val_audio_sequences = np.array(self.val_audio_sequences)
        
        # Verify shapes
        assert len(self.train_video_cluster_id_sequences.shape) == 2, \
            f"train_video_cluster_id_sequences should be a 2D array, got {self.train_video_cluster_id_sequences.shape}"
        assert len(self.train_audio_sequences.shape) == 3, \
            f"train_audio_sequences should be a 3D array, got {self.train_audio_sequences.shape}"
        assert self.train_video_cluster_id_sequences.shape[0] == self.train_audio_sequences.shape[0], \
            f"Number of training visual sequences ({self.train_video_cluster_id_sequences.shape[0]}) does not match number of training audio sequences ({self.train_audio_sequences.shape[0]})"
        assert self.train_video_cluster_id_sequences.shape[1] == self.train_audio_sequences.shape[1] + 1, \
            f"Length of training visual sequences ({self.train_video_cluster_id_sequences.shape[1]}) does not match length of training audio sequences ({self.train_audio_sequences.shape[1]})"
    
        # Print out the shapes of the training and validation datasets
        print(f"Training dataset shapes: visual {self.train_video_cluster_id_sequences.shape}, audio {self.train_audio_sequences.shape}")
        print(f"Validation dataset shapes: visual {self.val_video_cluster_id_sequences.shape}, audio {self.val_audio_sequences.shape}")

    def num_training_frames(self):
        """Returns the number of training frames."""
        return self.train_audio_sequences.shape[1] * self.train_audio_sequences.shape[0]

    def num_validation_frames(self):
        """Returns the number of validation frames."""
        return self.val_audio_sequences.shape[1] * self.val_audio_sequences.shape[0]

    def training_dataset(self):
        """Returns a tuple of (visual_sequences, audio_sequences) for training."""
        return self.train_video_cluster_id_sequences, self.train_audio_sequences

    def validation_dataset(self):
        """Returns a tuple of (visual_sequences, audio_sequences) for validation."""
        return self.val_video_cluster_id_sequences, self.val_audio_sequences


class ConditionalGPTTrainingPipeline(NanoGPTTrainingPipeline):
    """Training pipeline for the conditional GPT model."""
    
    @dataclass 
    class Config(NanoGPTTrainingPipeline.Config):
        audio_feature_dim: int = 13
        audio_proj_dim: int = 36
        model_arch: str = "gpt"  # Add model_arch with default value
        max_files: int = None
        data_dir: str = None
        num_training_tokens: int = None

    def create_model(self):
        """Create the conditional GPT model."""
        config = self.config
        model = ConditionalGPT(AudioGPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            bias=config.bias,
            dropout=config.dropout,
            audio_feature_dim=config.audio_feature_dim,
            audio_proj_dim=config.audio_proj_dim,
        ))
        
        # Store model arguments for checkpointing
        self.model_args = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "bias": config.bias,
            "vocab_size": config.vocab_size,
            "audio_feature_dim": config.audio_feature_dim,
            "audio_proj_dim": config.audio_proj_dim,
            "dropout": config.dropout,
        }
        
        if self.config.init_from == "resume":
            print(f"Resuming training from {self.out_dir}")
            # Resume training from a checkpoint
            ckpt_path = os.path.join(self.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.config.device)
            state_dict = checkpoint["model"]
            # Fix the keys of the state dictionary
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]
            self.checkpoint = checkpoint
            
            # Clear the checkpoint after loading
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            
            # Clear the checkpoint directory
            import shutil
            shutil.rmtree(self.out_dir)
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"Cleared checkpoint directory: {self.out_dir}")
            
        model.to(self.config.device)
        
        if self.config.compile:
            print("Compiling model...")
            model = torch.compile(model)
            
        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])
            
        self.model = model
        return model
        
    def get_batch(self, split: str):
        """Get a batch of data for training or validation."""
        train_visual, train_audio = self.train_data
        val_visual, val_audio = self.val_data
        
        visual_data = train_visual if split == "train" else val_visual
        audio_data = train_audio if split == "train" else val_audio
        
        batch_size = self.config.batch_size
        block_size = self.config.block_size
        device = self.config.device

        if len(visual_data.shape) == 2:
            # batch x sequence len
            irow = torch.randint(visual_data.shape[0], (batch_size,))
            ix = torch.randint(visual_data.shape[1] - block_size, (batch_size,))
            x_visual = torch.stack([
                visual_data[i, i1 : i1 + block_size].long() for i, i1 in zip(irow, ix)
            ])
            if audio_data.shape[1] == visual_data.shape[1]:
                x_audio = torch.stack([
                    audio_data[i, i1 + 1 : i1 + 1 + block_size].float() for i, i1 in zip(irow, ix)
                ])
            elif audio_data.shape[1] == visual_data.shape[1] - 1:
                x_audio = torch.stack([
                    audio_data[i, i1 : i1 + block_size].float() for i, i1 in zip(irow, ix)
                ])
            else:
                raise ValueError(f"audio_data.shape[1] ({audio_data.shape[1]}) does not match visual_data.shape[1] ({visual_data.shape[1]})")
            y = torch.stack([
                visual_data[i, i1 + 1 : i1 + 1 + block_size].long() for i, i1 in zip(irow, ix)
            ])
        else:
            raise NotImplementedError("Not implemented for shape: {visual_data.shape}")
            ix = torch.randint(len(visual_data) - block_size, (batch_size,))
            x_visual = torch.stack(
                [visual_data[i : i + block_size].long() for i in ix]
            )
            x_audio = torch.stack(
                [audio_data[i + 1 : i + 1 + block_size].float() for i in ix]
            )
            y = torch.stack(
                [visual_data[i + 1 : i + 1 + block_size].long() for i in ix]
            )

        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x_visual = x_visual.pin_memory().to(device, non_blocking=True)
            x_audio = x_audio.pin_memory().to(device, non_blocking=True) 
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x_visual = x_visual.to(device)
            x_audio = x_audio.to(device)
            y = y.to(device)

        assert x_visual.shape == (batch_size, block_size), f"x_visual.shape: {x_visual.shape}"
        assert x_audio.shape == (batch_size, block_size, self.config.audio_feature_dim), f"x_audio.shape: {x_audio.shape}"
        assert y.shape == (batch_size, block_size), f"y.shape: {y.shape}"

        return x_visual, x_audio, y
        
    def create_dataloaders(self, dataset_builder):
        """Create training and validation dataloaders."""
        train_visual, train_audio = dataset_builder.training_dataset()
        val_visual, val_audio = dataset_builder.validation_dataset()
        
        self.train_data = (
            torch.tensor(train_visual, dtype=torch.long),
            torch.tensor(train_audio, dtype=torch.float32)
        )
        self.val_data = (
            torch.tensor(val_visual, dtype=torch.long),
            torch.tensor(val_audio, dtype=torch.float32)
        )
        
        print(f"Train data shapes: visual {train_visual.shape}, audio {train_audio.shape}")
        print(f"Val data shapes: visual {val_visual.shape}, audio {val_audio.shape}")
    
    @torch.no_grad()
    def estimate_loss(self):
        model = self.model
        eval_iters = self.config.eval_iters
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y, targets = self.get_batch(split)
                # print(f"estimate_loss(): k={k}, X.shape: {X.shape}, Y.shape: {Y.shape}")
                with self.ctx:
                    logits, loss = model(X, Y, targets, mode=self.config.model_arch)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def training_loop(self):
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        ddp = self.ddp
        decay_lr = self.config.decay_lr
        learning_rate = self.config.learning_rate
        max_iters = self.config.max_iters
        grad_clip = self.config.grad_clip
        eval_interval = self.config.eval_interval
        log_interval = self.config.log_interval
        sample_interval = self.config.sample_interval
        always_save_checkpoint = self.config.always_save_checkpoint
        out_dir = self.out_dir
        wandb_log = self.config.wandb_log
        eval_only = self.config.eval_only
        batch_size = self.config.batch_size
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        master_process = self.master_process
        ctx = self.ctx

        # Get first example from training data for generation
        train_visual, train_audio = self.train_data
        example_visual = train_visual[0:1, :1]  # First example, first token
        example_audio = train_audio[0:1, :]  # First example, all audio frames
        example_audio = example_audio.to(self.config.device)
        example_visual = example_visual.to(self.config.device)

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        X, Y, targets = self.get_batch("train")  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model  # unwrap DDP container if needed
        running_mfu = -1.0
        # if wandb_log:
        #     import wandb
        #     wandb.init(project="lip_gpt", config=self.config.__dict__)
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if (iter_num + 1) % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if wandb_log:
                    import wandb
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )
                # if losses["val"] < best_val_loss or always_save_checkpoint:
                if True:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        model_args = self.model_args
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": self.config.__dict__,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                        
                        # Generate and save example sequence
                        print("Generating example sequence...")
                        with torch.no_grad():
                            generated = model.generate(
                                visual_ids=example_visual,
                                audio_features=example_audio,
                                temperature=1.0,
                                top_k=10,
                                mode=self.config.model_arch
                            )
                            # Save the generated sequence as JSON
                            example_path = os.path.join(out_dir, f"example_sequence.json")
                            sequence_data = {
                                "input_visual": example_visual.cpu().tolist(),
                                "generated": generated.cpu().tolist(),
                                "iter_num": iter_num,
                                "ground_truth": train_visual[0:1, :].cpu().tolist()
                            }
                            # print and compare ground truth and generated sequence
                            print(f"Ground truth: {train_visual[0, :].cpu().tolist()}")
                            print(f"Generated:    {generated[0].cpu().tolist()}")
                            with open(example_path, 'w') as f:
                                json.dump(sequence_data, f, indent=2)
                            print(f"Saved example sequence to {example_path}")
            if iter_num == 0 and eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (
                        micro_step == gradient_accumulation_steps - 1
                    )
                with ctx:
                    logits, loss = model(X, Y, targets, mode=self.config.model_arch)
                    # print(f"forward pass time: {end_time - start_time:.3f}s")
                    loss = (
                        loss / gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y, targets = self.get_batch("train")
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(
                        batch_size * gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break


class ConditionalGPTInferencePipeline:
    """Inference pipeline for the conditional GPT model."""
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.ckpt_path = ckpt_path
        self.device = device

    def generate(self, visual_ids: torch.Tensor, audio_features: torch.Tensor, 
                num_samples: int = 1, temperature: float = 1.0, top_k: int = 10, mode: str = "gpt"):
        """
        Generate visual token sequences conditioned on audio features.
        
        Args:
            visual_ids: Initial visual token sequence of shape (batch_size, seq_len)
            audio_features: Audio features of shape (batch_size, seq_len, audio_feature_dim)
            num_samples: Number of samples to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            mode: Model architecture to use: gpt (full transformer) or direct (simple audio-to-visual)
            
        Yields:
            Generated visual token sequences
        """
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
        
        # Load checkpoint and model
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        model_args = checkpoint["model_args"]
        gptconf = AudioGPTConfig(**model_args)
        model = ConditionalGPT(gptconf)
        
        # Load state dict
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        
        # Ensure inputs are tensors and on the correct device
        assert visual_ids.ndim == 2, f"visual_ids should be a 2D tensor, got {visual_ids.ndim}"
        assert audio_features.ndim == 3, f"audio_features should be a 3D tensor, got {audio_features.ndim}"
        assert audio_features.shape[0] == visual_ids.shape[0], f"Batch sizes don't match: visual_ids {visual_ids.shape[0]}, audio_features {audio_features.shape[0]}"
        
        x_visual = torch.tensor(visual_ids, dtype=torch.long, device=self.device)
        x_audio = torch.tensor(audio_features, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            with ctx:
                for k in range(num_samples):
                    y = model.generate(x_visual, x_audio, temperature=temperature, top_k=top_k, mode=mode)
                    yield y


def run_inference(args):
    """Run inference with the conditional GPT model."""
    # Import the detokenize function from lip_gpt0
    from src.lip_gpt0 import detokenize
    
    # Create inference pipeline
    inference_pipeline = ConditionalGPTInferencePipeline(
        ckpt_path=args.ckpt_path,
        device=args.device,
    )
    
    # Process audio file
    if args.audio_file:
        print(f"Processing audio file: {args.audio_file}")
        print(f"Using model architecture: {args.model_arch}")
        
        from src.audio_utils import extract_audio_features
        audio_features = extract_audio_features(
            audio_file=args.audio_file,
            sample_rate=args.sample_rate,
            fps=args.fps,
            device=args.device,
            pad_audio=True,
            audio_model=args.audio_model,
        )
        print(f"Extracted audio features with shape: {audio_features.shape}")
        
        # Create a batch of the same audio features for each sample
        batch_size = args.num_samples
        audio_feature_dim = audio_features.shape[2]  # Feature dimension
        
        # Prepare the input for the model
        visual_ids = np.array([[args.start_token]] * batch_size)  # Single start token to match audio length
        audio_feature_seq = audio_features.repeat(batch_size, 1, 1)
        print(f"Starting generation with single start token and all audio frames")
        print(f"Input shapes - visual_ids: {visual_ids.shape}, audio_features: {audio_feature_seq.shape}")
    else:
        print("No audio file provided. Using default audio features.")
        # Use a simple default audio feature sequence
        batch_size = args.num_samples
        audio_feature_dim = args.audio_feature_dim  # Use the expected feature dimension
        total_frames = 100  # Default number of frames when no audio is provided
        
        # Start with single start token to match audio sequence length
        visual_ids = np.array([[args.start_token]] * batch_size)
        
        # Create a simple default audio feature sequence (all zeros)
        audio_feature_seq = np.zeros((batch_size, 1, audio_feature_dim))
        
        print(f"Default input shapes - visual_ids: {visual_ids.shape}, audio_features: {audio_feature_seq.shape}")
    
    # Generate samples
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    for i, y in enumerate(inference_pipeline.generate(
        visual_ids=visual_ids,
        audio_features=audio_feature_seq,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        mode=args.model_arch,  # Pass the model architecture choice
    )):
        print(f"Generated sample {i+1}/{args.num_samples}")
        print(f"First few tokens: {y[0, :10].tolist()}")
        print(f"Avg feature value sequence: {audio_feature_seq[i, :10, :].mean(axis=1).tolist()}")
        
        # Create a descriptive filename
        if args.audio_file:
            audio_filename = os.path.basename(args.audio_file).split('.')[0]
            output_filename = f"generated_conditional_{audio_filename}_{i}.mp4"
        else:
            output_filename = f"generated_conditional_{args.start_token}_{i}.mp4"
        
        # Detokenize and save the generated sequence
        detokenize(
            token_ids=y[0].tolist()[1:],
            cluster_data_dir=args.cluster_data_dir,
            output_path=os.path.join(args.output_dir, output_filename),
            fps=args.fps,
        )
        
        # If we have an audio file, add it to the generated video
        if args.audio_file:
            from src.utils.video import add_audio_to_video
            output_path = os.path.join(args.output_dir, output_filename)
            final_output_path = os.path.join(args.output_dir, f"{audio_filename}_with_audio_{i}.mp4")
            add_audio_to_video(output_path, args.audio_file, final_output_path, remove_temp=True)
            print(f"Added audio to video: {final_output_path}")
        
        # Test latency on the first sample
        if i == 0:
            import time
            start_time = time.time()
            n = 10  # Reduced from 100 for faster testing
            total_tokens = 0
            for _ in range(n):
                assert len(audio_feature_seq.shape) == 3
                # print(f"visual_ids: {visual_ids.shape}, audio_features: {audio_feature_seq.shape}")
                generated = inference_pipeline.generate(
                    visual_ids=visual_ids[:1],  # Just use one sample for latency testing
                    audio_features=audio_feature_seq[:1, :, :],
                    num_samples=1,
                    mode=args.model_arch,  # Pass the model architecture choice
                )
                tokens_tensor = list(generated)[0].cpu()
                num_elements = tokens_tensor.numel() - 1
                total_tokens += num_elements
                
            end_time = time.time()
            print("================================================ Latency test results ================================================")
            print(f"Total tokens: {total_tokens}")
            print(f"Average latency per token: {(end_time - start_time) * 1000 / total_tokens} ms")
            print(f"Average tokens per second: {total_tokens / (end_time - start_time)}")
            print("=================================================================================================================")


def main():
    """Main function to train or run inference with the conditional GPT model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or run inference with the conditional GPT model")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", 
                        help="Mode to run the model in: train or inference")
    parser.add_argument("--model_arch", type=str, choices=["gpt", "direct"], default="gpt",
                        help="Model architecture to use: gpt (full transformer) or direct (simple audio-to-visual)")
    parser.add_argument("--ckpt_path", type=str, 
                        # default="data/gpt_logs/conditional_generation/batch128_block3/ckpt.pt",
                        default="data/gpt_logs/conditional_generation/gpt/batch128_block1/ckpt.pt",
                        help="Path to the checkpoint file for inference")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run the model on: cuda or cpu")
    parser.add_argument("--start_token", type=int, default=50,
                        help="Starting token for generation")
    parser.add_argument("--vocab_size", type=int, default=51,
                        help="Vocabulary size for the model")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k sampling parameter")
    parser.add_argument("--audio_file", type=str,
                        # default="data/conversations/0b69c3d770680d2966d07a2c85ca35a8529e03b943c4e0350f0e6e0a00fc3ad3_tts-1_nova.wav",
                        # default="data/conversations/0d25f2d1de2e0805f932a817fd254c3113443f65409612dfbd64d23c93fd6d68_tts-1_nova.wav",
                        default="data/conversations/002e4b0241534fc6f83d62452488bf1c7c05bc2ba69d840947a41d9a4727ae55_tts-1_nova.wav",  # this is good
                        ## default="data/conversations/fbc793678b9e6aee50fdbfe44cbb8a25334b96c8ab31219190087904b17ba267_tts-1_nova.wav",  # test set, short
                        # default="data/conversations/fd53b5ded29bb8bd5e7728a0227d91453233a52bb432cba23c66e8712d1ef39b_tts-1_nova.wav",  # test set, long
                        # default="data/conversations/ff08fde6732fbe0ad5c7346410e8d28d4c6070b4585b2a61701daa4c7de46e72_tts-1_nova.wav",  # test set, long
                        help="Path to an audio file to use for conditioning (WAV, MP3, etc.)")
    parser.add_argument("--output_dir", type=str,
                        default="data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50/gpt_generated_videos_gpt_block1",
                        help="Directory to save generated videos")
    parser.add_argument("--cluster_data_dir", type=str,
                        default="data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50",
                        help="Directory containing the image cluster data")
    parser.add_argument("--audio_model", type=str, default="hubert_zh",
                        choices=["hubert", "wav2vec2", "hubert_zh"],
                        help="Audio encoder model to use for feature extraction")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Sample rate for audio processing")
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second for the output video")
    parser.add_argument("--audio_feature_dim", type=int, default=768,
                        help="Expected audio feature dimension for the model")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    data_dir = "data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50/tokenized_data_mel"
    if args.mode == "train":
        # Create dataset builder
        dataset_builder = AudioVisualDatasetBuilder(
            # json_path="data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50/video_sequences.json",
            # audio_features_path="data/conversations_joyvasa_videos/bithuman_coach2_image_clusters_50/audio_features.npy",
            data_dir=data_dir,
            max_sequence_length=16,
            max_files=args.max_files,
            start_token=args.start_token,
        )

        # Define configuration
        config = ConditionalGPTTrainingPipeline.Config(
            log_dir=f"data/gpt_logs/conditional_generation/{args.model_arch}",
            block_size=7,
            vocab_size=args.vocab_size,  # Assuming this is the vocabulary size
            batch_size=128,
            flatten_tokens=False,
            # n_layer=6,
            n_layer=6,
            n_head=3,
            n_embd=36,
            start_token=args.start_token,
            log_interval=100,
            max_iters=10000,
            # audio_feature_dim=768,  # Match the hubert_zh feature dimension
            audio_feature_dim=80,
            audio_proj_dim=8,
            # audio_proj_dim=360,  # Match n_embd for the transformer
            model_arch=args.model_arch,  # Pass the model architecture choice
            max_files=args.max_files,
            data_dir=data_dir,
            num_training_tokens=dataset_builder.num_training_frames(),
            wandb_log=True,
        )
        
        
        
        # Create and run training pipeline
        pipeline = ConditionalGPTTrainingPipeline(config)
        
        # Save the config to the log directory
        with open(os.path.join(config.log_dir, "config.json"), "w") as f:
            config_dict = asdict(config)
            json.dump(config_dict, f)
        
        pipeline.fit(dataset_builder)
    else:  # inference
        run_inference(args)

if __name__ == "__main__":
    main()