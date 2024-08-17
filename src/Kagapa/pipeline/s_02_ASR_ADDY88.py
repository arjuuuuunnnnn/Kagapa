import os
import sys
from IPython import display as disp
import torch
import torchaudio
from torchaudio.transforms import Resample
import glob
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from multiprocessing import Pool, cpu_count
import functools

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from src.Kagapa import logger

os.makedirs("../../artifacts/STAGE_02", exist_ok=True)
STAGE_NAME = "Audio Transcription Stage"

class AudioTranscription:
    def __init__(self, input_dir, output_dir, model_id, target_lang):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_id = model_id
        self.target_lang = target_lang
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

    def process_audio_file(self, audio_file, i):
        audio_path = os.path.join(self.input_dir, audio_file)
        
        try:
            audio, orig_freq = torchaudio.load(audio_path)
            audio = audio.mean(dim=0, keepdim=True)
            resampler = Resample(orig_freq, 16_000)
            audio = resampler(audio)
            
            inputs = self.processor(
                audio.squeeze().numpy(),
                return_tensors="pt",
                sampling_rate=16_000
            )
            
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            cleaned_transcription = transcription.replace("<s>", "").strip()
            output_file = os.path.join(self.output_dir, f"{i+1:02d}a_transcription.txt")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_transcription)
            
            logger.info(f"Transcription for {audio_file} saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {e}")

    # def main(self):
    #     audio_files = [f for f in os.listdir(self.input_dir) if f.endswith('.mp3')]
    #     num_processes = min(cpu_count(), len(audio_files))
        
    #     process_func = functools.partial(self.process_audio_file)
        
    #     with Pool(num_processes) as pool:
    #         pool.starmap(process_func, enumerate(audio_files))

    def main(self):
        audio_files = [f for f in os.listdir(self.input_dir) if f.endswith('.mp3')]
    
        if not audio_files:
            print("No .mp3 files found in the input directory.")
            return
    
        num_processes = max(1, min(cpu_count(), len(audio_files)))
    
        process_func = functools.partial(self.process_audio_file)
    
        with Pool(num_processes) as pool:
            pool.starmap(process_func, enumerate(audio_files))

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        
        input_dir = "../../artifacts/STAGE_01/"
        output_dir = "../../artifacts/STAGE_02/"
        model_id = "addy88/wav2vec2-kannada-stt"
        target_lang = "kan"
        
        os.makedirs(output_dir, exist_ok=True)
        
        obj = AudioTranscription(input_dir, output_dir, model_id, target_lang)
        obj.main()
        
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(f"Error in pipeline execution: {str(e)}")
        raise e
