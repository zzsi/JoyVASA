"""
This is the data pipeline for the lip-gpt model.

1. Create sentences from natural conversation examples.
2. Use a text-to-speech model to generate audio files (wav) from the sentences.
3. Use batch_inference.py to generate the lip-synced videos from the audio files and source images.
4. Run clustering on the generated videos to get the clusters and lip tokens.
5. Run lip_gpt training pipeline to train a lip gpt model.
"""
import re
from openai import OpenAI
import hashlib
import os


def hash_sentence(sentence: str) -> str:
    """
    Hash the sentence to a unique identifier.
    """
    return hashlib.sha256(sentence.encode()).hexdigest()


def break_into_sentences(text):
    """
    Break the text into sentences, keeping the punctuation.
    """
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    return [s.strip() for s in sentences]


def conversations_examples():
    """
    Create sentences from natural conversation examples.
    """
    from .convesations.examples import examples
    for example in examples:
        yield "\n".join(example)


def sentences_from_examples():
    """
    Create sentences from natural conversation examples.
    """
    for example in conversations_examples():
        for sentence in break_into_sentences(example):
            yield sentence


class OpenAITTS:
    """
    Use OpenAI TTS to generate audio from a sentence.
    """
    def __init__(self, model: str = "tts-1", voice: str = "nova"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.voice = voice
    
    def generate_audio(self, sentence):
        """
        Generate audio from a sentence.
        """
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=sentence,
            response_format="wav"
        )
        return response.content


def generate_audio_from_a_sentence(sentence, model: str = "tts-1", voice: str = "nova", output_path: str = None, overwrite: bool = False):
    """
    Generate audio from a sentence.
    """
    tts = OpenAITTS(model, voice)
    audio = tts.generate_audio(sentence)
    if output_path:
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {output_path} because it already exists")
            return
        print(f"Saving audio to {output_path}")
        with open(output_path, "wb") as f:
            f.write(audio)
    return audio


def main():
    ## Uncomment this to generate audio from natural conversation examples
    # model = "tts-1"
    # voice = "nova"
    # for sentence in sentences_from_examples():
    #     print(f"Generating audio for: {sentence}")
    #     audio_id = f"{hash_sentence(sentence)}_{model}_{voice}"
    #     generate_audio_from_a_sentence(sentence, model, voice, f"data/conversations/{audio_id}.wav")

    ## Run batch inference to generate the lip-synced videos
    # Edit batch_inference.py, inference_config.py and argument_config.py. Run the following command to generate the lip-synced videos:
    # python batch_inference.py

    ## Run clustering on the generated videos to get the clusters and lip tokens
    # python -m src.generate_image_clusters --video_pattern "data/conversations_joyvasa_videos/bithuman_coach2/*_lip.mp4" --output_dir "data/conversations_joyvasa_videos/bithuman_coach2_image_clusters" --n_clusters 360 --max_frames 90000
    """
    These are test set (not used in training)
    2025-04-22 20:32:57,794 - INFO -   data/conversations_joyvasa_videos/bithuman_coach2/bithuman_coach_cropped2_fbc793678b9e6aee50fdbfe44cbb8a25334b96c8ab31219190087904b17ba267_tts-1_nova_lip.mp4
2025-04-22 20:32:57,794 - INFO -   data/conversations_joyvasa_videos/bithuman_coach2/bithuman_coach_cropped2_fbe7824cc19fe3ebb103710ae88c4d6dd5d84d66dc2f7b4bf9ac9118fb37be4e_tts-1_nova_lip.mp4
2025-04-22 20:32:57,794 - INFO -   data/conversations_joyvasa_videos/bithuman_coach2/bithuman_coach_cropped2_fd53b5ded29bb8bd5e7728a0227d91453233a52bb432cba23c66e8712d1ef39b_tts-1_nova_lip.mp4
2025-04-22 20:32:57,794 - INFO -   data/conversations_joyvasa_videos/bithuman_coach2/bithuman_coach_cropped2_fdf4783903ac42e1d03c6e886ac93d8b3fd5a2ba49700cbace161bee29c996c1_tts-1_nova_lip.mp4
2025-04-22 20:32:57,794 - INFO -   data/conversations_joyvasa_videos/bithuman_coach2/bithuman_coach_cropped2_ff08fde6732fbe0ad5c7346410e8d28d4c6070b4585b2a61701daa4c7de46e72_tts-1_nova_lip.mp4
    """

    ## Run lip_gpt training pipeline to train a lip_gpt model
    # python train_lip_gpt.py --config_path "configs/train_lip_gpt_config.yaml"

if __name__ == "__main__":
    main()