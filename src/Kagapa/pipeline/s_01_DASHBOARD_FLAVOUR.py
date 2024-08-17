import os
import subprocess
import sys
import csv
import shutil
import torch
from pyannote.audio import Pipeline
from IPython import display as disp
import torchaudio
from torchaudio.transforms import Resample
import glob
import soundfile
from pystoi import stoi
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger

os.makedirs("../../artifacts/STAGE_01", exist_ok=True)
STAGE_NAME = "Speaker Diarization Stage"

class SpeakerDiarization:
    def __init__(self, input_dir, output_dir, auth_token):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.auth_token = auth_token
        self.intermittent_csvs = os.path.join(output_dir, "TEXT")
        self.intermittent_audios = os.path.join(output_dir, "AUDIO")

    def run_pyannote_seg_3(self, input_file, output_csv):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.auth_token
        )
        pipeline.to(torch.device("cpu"))
        
        diarization = pipeline(input_file)

        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['speaker', 'start_time', 'stop_time'])
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                csv_writer.writerow([f'speaker_{speaker}', f'{turn.start:.1f}', f'{turn.end:.1f}'])

    def process_audio(self, mp3_file):
        print(f"Now, Working On  --->  {mp3_file}")
        logger.info(f"Processing file: {mp3_file}")
        
        tmp_csv = os.path.join(self.intermittent_csvs, f'{os.path.basename(mp3_file)}_MODEL_ANALYSIS.csv')
        self.run_pyannote_seg_3(mp3_file, tmp_csv)
        
        segment_number = 0
        output_directory = self.intermittent_audios

        with open(tmp_csv, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader, None)  # Skip header
            segments = list(csv_reader)
            logger.info(f"Number of segments detected: {len(segments)}")

            for speaker, start_time, stop_time in segments:
                logger.info(f"Processing segment {segment_number}: speaker={speaker}, start={start_time}, end={stop_time}")
                output_audio = f"CUT_{os.path.basename(mp3_file)}__{segment_number}__.mp3"
                command = [
                    'ffmpeg', '-y', '-i', mp3_file,
                    '-ss', start_time, '-to', stop_time,
                    '-c:a', 'libmp3lame', '-q:a', '0', '-vn', '-c', 'copy',
                    os.path.join(output_directory, output_audio)
                ]
                logger.info(f"Running ffmpeg command: {' '.join(command)}")
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"ffmpeg error: {result.stderr}")
                else:
                    logger.info(f"Created segment file: {output_audio}")
                segment_number += 1

        new_dir = os.path.join(output_directory, f"{os.path.basename(mp3_file)}_CUTS")
        os.makedirs(new_dir, exist_ok=True)
        shutil.move(tmp_csv, os.path.join(self.intermittent_csvs, f'{os.path.basename(mp3_file)}_MODEL_ANALYSIS.csv'))
        
        for file in os.listdir(output_directory):
            if file.startswith(f"CUT_{os.path.basename(mp3_file)}"):
                shutil.move(os.path.join(output_directory, file), new_dir)

    def main(self):
        os.makedirs(self.intermittent_csvs, exist_ok=True)
        os.makedirs(self.intermittent_audios, exist_ok=True)

        mp3_files = glob.glob(os.path.join(self.input_dir, "*.mp3"))
        for mp3_file in mp3_files:
            self.process_audio(mp3_file)
        logger.info(f"Total number of files processed: {len(mp3_files)}")
        logger.info(f"Total number of output files created: {len(os.listdir(self.intermittent_audios))}")

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        
        input_dir = "../../artifacts/STAGE_00/"
        output_dir = "../../artifacts/STAGE_01/"
        auth_token = "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"
        
        obj = SpeakerDiarization(input_dir, output_dir, auth_token)
        obj.main()
        
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(f"Error in pipeline execution: {str(e)}")
        raise e
