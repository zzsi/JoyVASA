import os
import os.path as osp
import subprocess
from typing import List, Optional
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
    reference_path: str,
    audio_path: str,
    output_dir: str,
    animation_mode: str = "human",
    **kwargs
) -> Optional[str]:
    """Process a single audio file and return the output path if successful."""
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
            "animation_mode": animation_mode,
            **kwargs
        }
        
        # Create configs
        args = ArgumentConfig(**base_args)
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        # Initialize appropriate pipeline
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

        # Execute pipeline
        pipeline.execute(args)
        
        # Return the output path
        return osp.join(output_dir, osp.basename(audio_path).replace(".wav", ".mp4"))
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def batch_process_audio(
    reference_path: str,
    audio_dir: str,
    output_dir: str,
    animation_mode: str = "human",
    **kwargs
) -> List[str]:
    """Process all audio files in a directory and return list of successful outputs."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files (both WAV and MP3)
    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(('.wav', '.mp3'))]
    if not audio_files:
        print(f"No .wav or .mp3 files found in {audio_dir}")
        return []
    
    # Process each audio file
    successful_outputs = []
    for audio_file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        audio_path = osp.join(audio_dir, audio_file)
        output_path = process_single_audio(
            reference_path=reference_path,
            audio_path=audio_path,
            output_dir=output_dir,
            animation_mode=animation_mode,
            **kwargs
        )
        if output_path:
            successful_outputs.append(output_path)
    
    return successful_outputs

if __name__ == "__main__":
    # Example usage
    reference_path = "assets/examples/imgs/bithuman_coach_cropped.png"
    audio_dir = "assets/examples/audios"
    output_dir = "animations/joyvasa_test"
    
    successful_outputs = batch_process_audio(
        reference_path=reference_path,
        audio_dir=audio_dir,
        output_dir=output_dir,
        animation_mode="human"  # or "animal" or "lip"
    )
    
    print(f"\nSuccessfully processed {len(successful_outputs)} files:")
    for output in successful_outputs:
        print(f"- {output}") 