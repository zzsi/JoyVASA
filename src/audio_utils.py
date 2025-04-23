import torch.nn as nn

__loaded_audio_encoder = {
    'wav2vec2': None,
    'hubert': None,
    'hubert_zh': None
}

def extract_audio_features(
    audio_file: str,
    sample_rate: int,
    fps: int,
    device: str,
    pad_audio: bool,
    audio_model: str,
):
    import librosa
    import numpy as np
    import torch
    import torch.nn.functional as F
    import math
    import platform
    from src.config.base_config import make_abs_path
    
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)
    print(f"Audio loaded with shape: {audio.shape}, sample rate: {sr}")
    
    # Calculate number of frames based on audio duration
    audio_duration = len(audio) / sr  # in seconds
    total_frames = max(1, int(audio_duration * fps))  # Ensure at least 1 frame
    print(f"Audio duration: {audio_duration:.2f}s, will generate {total_frames} frames")
    
    # Convert to tensor and reshape for HuBERT
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float().to(device)
    # Reshape audio to (batch_size, sequence_length)
    audio = audio.unsqueeze(0)  # Add batch dimension
    # print(f"Audio tensor shape after reshaping: {audio.shape}")
    
    # Add padding to ensure we have enough audio for the entire sequence
    if pad_audio:
        audio = F.pad(audio, (1280, 640), "constant", 0)
    
    # Extract audio features using the audio encoder model
    # print(f"Using {audio_model} audio encoder for feature extraction")
    
    # Initialize the appropriate audio encoder if not already loaded
    if __loaded_audio_encoder[audio_model] is None:
        if audio_model == 'wav2vec2':
            from src.modules.wav2vec2 import Wav2Vec2Model
            audio_encoder = Wav2Vec2Model.from_pretrained(
                make_abs_path('../../pretrained_weights/wav2vec2-base-960h')
            )
            audio_encoder.feature_extractor._freeze_parameters()
        elif audio_model == 'hubert':
            from src.modules.hubert import HubertModel
            audio_encoder = HubertModel.from_pretrained(
                make_abs_path('../../pretrained_weights/hubert-base-ls960')
            )
            audio_encoder.feature_extractor._freeze_parameters()
        elif audio_model == 'hubert_zh':
            from src.modules.hubert import HubertModel
            model_path = '../../pretrained_weights/chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            audio_encoder = HubertModel.from_pretrained(
                make_abs_path(model_path)
            )
            audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {audio_model}!')
            
        # Move encoder to device and store in cache
        audio_encoder = audio_encoder.to(device)
        __loaded_audio_encoder[audio_model] = audio_encoder
    else:
        audio_encoder = __loaded_audio_encoder[audio_model]
    
    # Define a function to pad audio similar to the one in DitTalkingHead
    def pad_audio(audio):
        # Add padding to ensure we have enough audio for the entire sequence
        return F.pad(audio, (1280, 640), "constant", 0)
    
    # Extract features using the encoder
    with torch.no_grad():
        hidden_states = audio_encoder(pad_audio(audio), fps, frame_num=total_frames * 2).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(hidden_states, size=total_frames, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)
    
    audio_features = hidden_states.cpu()

    return audio_features