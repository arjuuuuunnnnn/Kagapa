import os
import subprocess
import sys
from IPython import display as disp
import torch
import torchaudio
from torchaudio.transforms import Resample
from denoiser.dsp import convert_audio
from denoiser import pretrained
import glob
import soundfile
from pystoi import stoi
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger
os.makedirs("../../artifacts/STAGE_00", exist_ok=True)

STAGE_NAME = "Audio Denoising Stage"

class AudioDenoising:
    def __init__(self, input_file, output_dir, model_id, target_lang):
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_id = model_id
        self.target_lang = target_lang

    def main(self):
        LIST_OF_AUDIO_FILES = glob.glob(self.input_file)
        model = pretrained.dns64()
        processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        sr_model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

        for audio_file in LIST_OF_AUDIO_FILES:
            file_name = os.path.basename(audio_file)
            if "Cleaned_denoiser_facebook_1" in file_name:
                continue

            wav, sr = torchaudio.load(audio_file)
            wav = convert_audio(wav, sr, model.sample_rate, model.chin)

            with torch.no_grad():
                denoised = model(wav[None])[0]
                disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))

                os.makedirs(self.output_dir, exist_ok=True)
                output_file = f"{self.output_dir}/Cleaned_denoiser_facebook_1_{file_name}"
                torchaudio.save(output_file, denoised.data.cpu(), model.sample_rate)

                # Perform speech recognition on the denoised audio
                inputs = processor(denoised.data.cpu().numpy(), return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
                with torch.no_grad():
                    logits = sr_model(inputs.input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
                print("Transcription:", transcription)

                # Compute STOI score
                denoised_np = denoised.data.cpu().numpy()[0]
                wav_np = wav.cpu().numpy()[0]
                stoi_score = stoi(wav_np, denoised_np, model.sample_rate, extended=False)
                logger.info(f"STOI SCORE : {stoi_score}")

if __name__ == "__main__":
    try:
        STAGE_NAME = "Audio Denoising Stage"
        logger.info(f"********** stage {STAGE_NAME} started **********")
        
        input_file = "../../artifacts/INPUT/input.mp3"
        output_dir = "../../artifacts/STAGE_00/"
        model_id = "addy88/wav2vec2-kannada-stt"
        target_lang = "kan"
        
        obj = AudioDenoising(input_file, output_dir, model_id, target_lang)
        obj.main()
        
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(f"Error in pipeline execution: {str(e)}")
        raise e

