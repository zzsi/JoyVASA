import os
import os.path as osp
import subprocess
import numpy as np
import pickle
from typing import List, Optional, Dict
from tqdm import tqdm
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wmg_pipeline import LivePortraitPipeline
from src.live_portrait_wmg_pipeline_animal import LivePortraitPipelineAnimal
from src.live_portrait_lip_pipeline import LivePortraitLipPipeline

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def convert_to_wav(audio_path: str, output_dir: str) -> Optional[str]:
    """Convert audio file to WAV format if needed."""
    if audio_path.lower().endswith('.wav'):
        return audio_path
        
    output_path = osp.join(output_dir, osp.splitext(osp.basename(audio_path))[0] + '.wav')
    
    try:
        # Convert to WAV format with 16kHz sample rate and mono channel
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            '-c:a', 'pcm_s16le',  # Use PCM 16-bit format
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {audio_path} to WAV: {e.stderr.decode()}")
        return None

def process_single_audio(
    pipeline,
    reference_path: str,
    audio_path: str,
    output_dir: str,
    **kwargs
) -> Optional[Dict[str, str]]:
    """Process a single audio file and return the output paths if successful."""
    try:
        # Convert audio to WAV if needed
        wav_path = convert_to_wav(audio_path, output_dir)
        if wav_path is None:
            return None
            
        # Create base config with required arguments
        base_args = {
            "reference": reference_path,
            "audio": wav_path,
            "output_dir": output_dir,
            **kwargs
        }
        
        # Create configs
        args = ArgumentConfig(**base_args)
        
        # Execute pipeline
        result = pipeline.execute(args)
        
        # For lip pipeline, result will be a tuple of (video_path, features_path)
        if isinstance(result, tuple):
            video_path, features_path = result
            return {
                "video": video_path,
                "features": features_path
            }
        else:
            # For other pipelines, result is just the video path
            return {
                "video": result
            }
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def find_audio_files(directory: str) -> List[str]:
    """Recursively find all audio files in the given directory and its subdirectories."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(osp.join(root, file))
    return audio_files

def batch_process_audio(
    reference_path: str,
    audio_dir: str,
    output_dir: str,
    animation_mode: str = "human",
    **kwargs
) -> List[Dict[str, str]]:
    """Process all audio files in a directory and its subdirectories and return list of successful outputs."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files recursively
    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print(f"No .wav or .mp3 files found in {audio_dir} or its subdirectories")
        return []
    audio_files = sorted(audio_files)

    # Create base config with required arguments
    base_args = {
        "reference": reference_path,
        "audio": "",  # Will be set for each file
        "output_dir": output_dir,
        "animation_mode": animation_mode,
        **kwargs
    }
    
    # Create configs once
    args = ArgumentConfig(**base_args)
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # Initialize appropriate pipeline once
    if animation_mode == "animal":
        pipeline = LivePortraitPipelineAnimal(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    elif animation_mode == "human":
        pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    elif animation_mode == "lip":
        pipeline = LivePortraitLipPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
    else:
        raise ValueError(f"Unsupported animation mode: {animation_mode}")
    
    # Process each audio file using the same pipeline
    successful_outputs = []
    for audio_path in tqdm(audio_files, desc="Processing audio files", unit="file"):
        output_paths = process_single_audio(
            pipeline=pipeline,
            reference_path=reference_path,
            audio_path=audio_path,
            output_dir=output_dir,
            **kwargs
        )
        if output_paths:
            successful_outputs.append(output_paths)
    
    return successful_outputs


if __name__ == "__main__":
    # Example usage
    reference_path = "assets/examples/imgs/bithuman_coach_cropped2.png"
    # audio_dir = "data/audio_files_for_batch_inference"
    # audio_dir = "data/conversations"
    audio_dir = "data/celeb_extracted_audios/videos"
    # output_dir = "data/batch_generated_videos/bithuman_coach"
    output_dir = "data/celab_joyvasa_videos/bithuman_coach2"
    successful_outputs = batch_process_audio(
        reference_path=reference_path,
        audio_dir=audio_dir,
        output_dir=output_dir,
        animation_mode="lip",  # or "animal" or "human"
        # animation_mode="human"
        overwrite=False,
    )
    
    print(f"\nSuccessfully processed {len(successful_outputs)} files:")
    for output in successful_outputs:
        print(f"- Video: {output['video']}")
        if 'features' in output:
            print(f"  Features: {output['features']}") 