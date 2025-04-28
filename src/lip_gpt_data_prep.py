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
    if output_path and os.path.exists(output_path) and not overwrite:
        print(f"Skipping {output_path} because it already exists")
        return None
        
    tts = OpenAITTS(model, voice)
    audio = tts.generate_audio(sentence)
    
    if output_path:
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
    # python -m src.generate_image_clusters --video_pattern "data/conversations_joyvasa_videos/bithuman_coach2/*_lip.mp4" --output_dir "data/conversations_joyvasa_videos/bithuman_coach2_image_clusters" --n_clusters 128 --max_frames 10000

    ## Run an audio encoder to extract the audio features and prepare training data: visual id sequence, paired with audio features

    ## Run lip_gpt training pipeline to train a lip_gpt model
    # python -m src.lip_gpt --model_arch direct --mode train

if __name__ == "__main__":
    main()