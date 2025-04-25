from scipy import signal
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import math
import platform
from src.config.base_config import make_abs_path

import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

__loaded_audio_encoder = {
    'wav2vec2': None,
    'hubert': None,
    'hubert_zh': None,
    'wav2lip': None
}

class HParams:
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.data[key]

	def set_hparam(self, key, value):
		self.data[key] = value


# Default hyperparameters
hp = HParams(
	num_mels=80,  # Number of mel-spectrogram channels and local conditioning dimensionality
	#  network
	rescale=True,  # Whether to rescale audio prior to preprocessing
	rescaling_max=0.9,  # Rescaling value
	
	# Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
	# It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
	# Does not work if n_ffit is not multiple of hop_size!!
	use_lws=False,
	
	n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
    # n_fft=1280,
	hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
    
	win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
    # hop_size=640,
    # win_size=1280,
	sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
	
	frame_shift_ms=None,  # Can replace hop_size parameter. (Recommended: 12.5)
	
	# Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization=True,
	# Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
	symmetric_mels=True,
	# Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, 
	# faster and cleaner convergence)
	max_abs_value=4.,
	# max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not 
	# be too big to avoid gradient explosion, 
	# not too small for fast convergence)
	# Contribution by @begeekmyfriend
	# Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude 
	# levels. Also allows for better G&L phase reconstruction)
	preemphasize=True,  # whether to apply filter
	preemphasis=0.97,  # filter coefficient.
	
	# Limits
	min_level_db=-100,
	ref_level_db=20,
	fmin=55,
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To 
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,  # To be increased/reduced depending on data.

	###################### Our training parameters #################################
	img_size=96,
	fps=25,
	
	batch_size=16,
	initial_learning_rate=1e-4,
	nepochs=200000000000000000,  ### ctrl + c, stop whenever eval loss is consistently greater than train loss for ~10 epochs
	num_workers=16,
	checkpoint_interval=3000,
	eval_interval=3000,
    save_optimizer_state=True,

    syncnet_wt=0.0, # is initially zero, will be set automatically to 0.03 later. Leads to faster convergence. 
	syncnet_batch_size=64,
	syncnet_lr=1e-4,
	syncnet_eval_interval=10000,
	syncnet_checkpoint_interval=10000,

	disc_wt=0.07,
	disc_initial_learning_rate=1e-4,
)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav

def _linear_to_mel(S):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,
                                       n_mels=hp.num_mels,
                                       fmin=hp.fmin, fmax=hp.fmax)
    return np.dot(_mel_basis, S)

def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,
                               fmin=hp.fmin, fmax=hp.fmax)

def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip((2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                           -hp.max_abs_value, hp.max_abs_value)
        else:
            return np.clip(hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)), 0, hp.max_abs_value)
    
    assert S.max() <= 0 and S.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))

def _denormalize(D):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return (((np.clip(D, -hp.max_abs_value,
                              hp.max_abs_value) + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value))
                    + hp.min_level_db)
        else:
            return ((np.clip(D, 0, hp.max_abs_value) * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)
    
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)

def melspectrogram(wav):                   # <-- main public helper
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
    if hp.signal_normalization:            # optional log-scale normalisation
        return _normalize(S)
    return S

def _stft(y):
    return librosa.stft(y=y, n_fft=hp.n_fft,
                        hop_length=hp.hop_size, win_length=hp.win_size)

_mel_basis = None
def _linear_to_mel(S):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,
                                       n_mels=hp.num_mels,
                                       fmin=hp.fmin, fmax=hp.fmax)
    return np.dot(_mel_basis, S)


def extract_audio_features(
    audio_file: str,
    sample_rate: int,
    fps: int,
    device: str,
    pad_audio: bool,
    audio_model: str,
    hidden_layer_idx: int = None,  # or 6 for HuBERT
    stack_adjacent_frames: bool = False,
):
    
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=sample_rate) #, mono=True)
    # print(f"Audio loaded with shape: {audio.shape}, sample rate: {sr}")
    
    # Calculate number of frames based on audio duration
    audio_duration = len(audio) / sr  # in seconds
    total_frames = max(1, int(audio_duration * fps))  # Ensure at least 1 frame
    # print(f"Audio duration: {audio_duration:.2f}s, expected to generate {total_frames} frames at {fps} fps")

    if audio_model == "mel":
        audio_features = melspectrogram(audio)
        audio_features = audio_features.T
        audio_features = audio_features[np.newaxis, :, :]
        return audio_features
    elif audio_model == "wav2lip":
        mel_step_size = 16
        wav = audio
        mel = melspectrogram(wav)
        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        # print("Length of mel chunks: {}".format(len(mel_chunks)))

        mel_batch = np.asarray(mel_chunks)
        # print("mel_batch", mel_batch.shape)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        # print("mel_batch reshaped", mel_batch.shape)
        mel_torch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        # print("mel_torch", mel_torch.shape)
        if __loaded_audio_encoder[audio_model] is None:
            from src.wav2lip import Wav2Lip, load_model
            audio_encoder = load_model(make_abs_path('../../pretrained_weights/Wav2Lip-SD-NOGAN.pt'))
            audio_encoder: Wav2Lip = audio_encoder.to(device)
            __loaded_audio_encoder[audio_model] = audio_encoder
        else:
            audio_encoder: Wav2Lip = __loaded_audio_encoder[audio_model]
        
        with torch.no_grad():
            audio_features = audio_encoder.audio_encoder(mel_torch)
            audio_features = audio_features.reshape(1, audio_features.shape[0], audio_features.shape[1])  # (1, T, 512)
        return audio_features
    
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
        encoder_output = audio_encoder(pad_audio(audio), fps, frame_num=total_frames * 2, output_hidden_states=hidden_layer_idx is not None)
        if hidden_layer_idx is not None:
            hidden_states = encoder_output.hidden_states[hidden_layer_idx]
        else:
            hidden_states = encoder_output.last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        if stack_adjacent_frames:
            hidden_states = torch.cat([hidden_states[:, :, ::2], hidden_states[:, :, 1::2]], dim=1)  # (N, 768 * 2, L)
        else:
            hidden_states = F.interpolate(hidden_states, size=total_frames, align_corners=False, mode='linear')  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)
    
    audio_features = hidden_states.cpu()

    return audio_features


if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(
        audio_file="data/conversations/002e4b0241534fc6f83d62452488bf1c7c05bc2ba69d840947a41d9a4727ae55_tts-1_nova.wav",
        # audio_file="data/conversations/0a659047bd4c2383a577e553a1d8059704900584c48f0bbf7f51a7b2a327cbf4_tts-1_nova.wav",
        # audio_model="hubert_zh",
        audio_model="mel",
        sample_rate=16000,
        fps=25,
        device="cuda",
        pad_audio=True,
        hidden_layer_idx=None,
        stack_adjacent_frames=False,
    )
    audio_features = extract_audio_features(**vars(args))
    print("mel", audio_features.shape)

    args.audio_model = "wav2lip"
    audio_features = extract_audio_features(**vars(args))
    print("wav2lip", audio_features.shape)

    args.audio_model = "hubert_zh"
    audio_features = extract_audio_features(**vars(args))
    print("hubert_zh", audio_features.shape)