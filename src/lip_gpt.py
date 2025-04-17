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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.projection(x))


class ConditionalGPT(GPT):
    """GPT model that conditions on audio features."""
    
    def __init__(self, config):
        super().__init__(config)
        self.audio_projection = AudioProjection(config)
        # Add a projection layer to reduce concatenated embeddings back to n_embd
        self.combined_projection = nn.Linear(2 * config.n_embd, config.n_embd)
        
    def forward(self, visual_ids, audio_features, targets=None):
        device = visual_ids.device
        b, t = visual_ids.size()
        assert audio_features.size()[:2] == (b, t), f"Audio features shape {audio_features.shape} doesn't match visual ids shape {visual_ids.shape}"
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(visual_ids)  # token embeddings of shape (b, t, n_embd)
        audio_emb = self.audio_projection(audio_features)  # (b, t, n_embd)
        combined_emb = self.combined_projection(torch.cat([audio_emb, tok_emb], dim=-1))
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(combined_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def generate(self, visual_ids, audio_features, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of visual indices and audio features and generate 
        the following visual tokens.
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = visual_ids if visual_ids.size(1) <= self.config.block_size else visual_ids[:, -self.config.block_size:]
            audio_cond = audio_features if audio_features.size(1) <= self.config.block_size else audio_features[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond, audio_cond)
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
    def __init__(self, json_path: str, audio_features_path: str, max_sequence_length: int = 80):
        with open(json_path, "r") as f:
            self.video_cluster_id_sequences = json.load(f)
        # Allow pickle=True to load object arrays
        self.audio_features = np.load(audio_features_path, allow_pickle=True).item()  # Should be aligned with video sequences
        self.max_sequence_length = max_sequence_length
        
        # Convert audio_features dict to a list aligned with video_cluster_id_sequences
        # Process video sequences and audio features to ensure alignment
        audio_features_list = []
        visual_sequences_list = []
        
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
        
        # Verify that we have data to work with
        assert len(self.video_cluster_id_sequences) > 0, "No valid aligned audio-visual sequences found"
        # Break down long sequences into smaller chunks
        equalized_visual_sequences = []
        equalized_audio_sequences = []
        
        print(f"There are {len(self.video_cluster_id_sequences)} sequences in the dataset")
        for i, (visual_seq, audio_seq) in enumerate(zip(self.video_cluster_id_sequences, self.audio_features)):
            # Verify that each audio sequence has the same length as its corresponding visual sequence
            assert len(visual_seq) == len(audio_seq), \
                f"Sequence {i}: Visual sequence length ({len(visual_seq)}) does not match audio sequence length ({len(audio_seq)})"
            
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
                print(f"Warning: Skipping {video_path} - Visual sequence length ({len(visual_seq)}) is less than max sequence length ({self.max_sequence_length})")
            
        print(f"There are {len(equalized_visual_sequences)} sequences in the dataset after equalization")
        
        # Convert sequences to numpy arrays
        equalized_visual_sequences = [np.array(seq) for seq in equalized_visual_sequences]
        equalized_audio_sequences = [np.array(seq) for seq in equalized_audio_sequences]
        
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
        assert self.train_video_cluster_id_sequences.shape[1] == self.train_audio_sequences.shape[1], \
            f"Length of training visual sequences ({self.train_video_cluster_id_sequences.shape[1]}) does not match length of training audio sequences ({self.train_audio_sequences.shape[1]})"
    
        # Print out the shapes of the training and validation datasets
        print(f"Training dataset shapes: visual {self.train_video_cluster_id_sequences.shape}, audio {self.train_audio_sequences.shape}")
        print(f"Validation dataset shapes: visual {self.val_video_cluster_id_sequences.shape}, audio {self.val_audio_sequences.shape}")

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
            x_audio = torch.stack([
                audio_data[i, i1 + 1 : i1 + 1 + block_size].float() for i, i1 in zip(irow, ix)
            ])
            y = torch.stack([
                visual_data[i, i1 + 1 : i1 + 1 + block_size].long() for i, i1 in zip(irow, ix)
            ])
        else:
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
                    logits, loss = model(X, Y, targets)
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
        print("master_process:", master_process)
        print("batch_size:", batch_size)
        print("gradient_accumulation_steps:", gradient_accumulation_steps)

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        X, Y, targets = self.get_batch("train")  # fetch the very first batch
        print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")
        print(f"X: {X[0, :5]}")
        # import sys; sys.exit(0)
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model.module if ddp else model  # unwrap DDP container if needed
        running_mfu = -1.0
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
                # print("batch_size:", batch_size)
                if wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )
                if losses["val"] < best_val_loss or always_save_checkpoint:
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
                    logits, loss = model(X, Y, targets)
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



def main():
    """Main function to train the conditional GPT model."""
    # Define configuration
    config = ConditionalGPTTrainingPipeline.Config(
        log_dir="data/gpt_logs/conditional_generation",
        block_size=10,
        vocab_size=768,  # Assuming this is the vocabulary size
        batch_size=128,
        flatten_tokens=False,
        n_layer=6,
        n_head=3,
        n_embd=36,
        start_token=767,
        log_interval=100,
        max_iters=200000,
        audio_feature_dim=13,  # Match the MFCC feature dimension
        audio_proj_dim=36,  # Match n_embd for the transformer
    )
    
    # Create dataset builder
    dataset_builder = AudioVisualDatasetBuilder(
        json_path="data/batch_generated_videos/bithuman_coach_image_clusters/video_sequences.json",
        audio_features_path="data/batch_generated_videos/bithuman_coach_image_clusters/audio_features.npy",
    )
    
    # Create and run training pipeline
    pipeline = ConditionalGPTTrainingPipeline(config)
    
    # Save the config to the log directory
    with open(os.path.join(config.log_dir, "config.json"), "w") as f:
        config_dict = asdict(config)
        json.dump(config_dict, f)
    
    pipeline.fit(dataset_builder)


if __name__ == "__main__":
    main()