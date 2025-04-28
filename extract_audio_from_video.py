"""
Extract audios from videos.

Given a directory of videos, or a tar file of videos, extract the audios and save them as wav files into a directory.
"""

import os
import tarfile
import argparse
import soundfile as sf
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import tempfile
import re
import warnings

# suppress warnings
warnings.filterwarnings("ignore")


def extract_audio_from_video(video_path: str, output_path: str, sr: int = 16000):
    """
    Extract audio from a video file and save it as a wav file.
    Returns True if successful, False if no audio track found.
    """
    try:
        # Load the video file
        video = VideoFileClip(video_path)
        
        # Extract the audio
        audio = video.audio
        
        if audio is None:
            print(f"No audio track found in {video_path}")
            video.close()
            return False
        
        # Write the audio to a temporary file
        temp_audio_path = output_path + '.temp.wav'
        audio.write_audiofile(temp_audio_path, fps=sr, verbose=False, logger=None)
        
        # Read the audio file and write it with soundfile
        audio_data, sr = sf.read(temp_audio_path)
        sf.write(output_path, audio_data, sr)
        
        # Clean up
        video.close()
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        if 'video' in locals():
            video.close()
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters and replacing them with underscores.
    """
    # Remove any leading/trailing whitespace and invalid characters
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    # Remove any leading dashes
    filename = re.sub(r'^-+', '', filename)
    return filename


def extract_audio_from_tar(tar_path: str, output_dir: str):
    """
    Extract audios from a tar file of videos and save them as wav files into a directory.
    """
    # Create a temporary directory to store extracted videos
    with tempfile.TemporaryDirectory() as temp_dir:
        # Count total video files
        with tarfile.open(tar_path, "r") as tar:
            total_videos = sum(1 for member in tar.getmembers() 
                             if member.type == tarfile.REGTYPE and member.name.endswith(".mp4"))
        
        # Initialize counters
        successful_extractions = 0
        
        # Iterate over the tar file and extract the audios
        with tarfile.open(tar_path, "r") as tar:
            pbar = tqdm(tar.getmembers(), total=total_videos, 
                       desc=f"Extracting audio (0/{total_videos} successful)")
            
            for member in pbar:
                if member.type == tarfile.REGTYPE and member.name.endswith(".mp4"):
                    # Sanitize the filename
                    safe_filename = sanitize_filename(os.path.basename(member.name))
                    temp_video_path = os.path.join(temp_dir, safe_filename)
                    
                    # Extract video to temporary directory
                    tar.extract(member, temp_dir)
                    # Rename the extracted file to the sanitized name
                    os.rename(os.path.join(temp_dir, member.name), temp_video_path)
                    
                    # Create output path for audio
                    audio_path = os.path.join(output_dir, member.name.replace(".mp4", ".wav"))
                    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                    
                    # Extract audio from video
                    if extract_audio_from_video(temp_video_path, audio_path):
                        successful_extractions += 1
                        pbar.set_description(f"Extracting audio ({successful_extractions}/{total_videos} successful)")
                    
                    # Clean up temporary video file
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
        
        print(f"\nExtraction complete. Successfully extracted {successful_extractions} out of {total_videos} videos.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if args.video_path.endswith(".tar"):
        extract_audio_from_tar(args.video_path, args.output_dir)
    else:
        extract_audio_from_video(args.video_path, args.output_dir)


if __name__ == "__main__":
    main()
