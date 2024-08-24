# import os
# import sys
# import subprocess

# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(project_root)

# from src.Kagapa import logger
# from src.Kagapa.pipeline.s_00_DENOISER_DNS_64 import AudioDenoising
# #from src.Kagapa.pipeline.s_01_DASHBOARD_FLAVOUR import SpeakerDiarization
# from src.Kagapa.pipeline.s_02_ASR_ADDY88 import AudioTranscription
# from src.Kagapa.pipeline.s_03_PUNCT_PCS47LANG import TextPunctuation
# from src.Kagapa.pipeline.s_04_TRANSLIT_OM import TextTransliteration
# from src.Kagapa.pipeline.s_05_TRANSLATE_AI4BHARAT import TextTranslation
# from src.Kagapa.pipeline.s_06_ENG_GRAMMAR_VENNIFY import GrammarCorrection


# def run_stage_01():
#     stage_01_venv = "/home/hemanth/only_ml/CDSAML-Kan-Eng-Speect-Text/model2_env/bin/python"
#     stage_01_script = os.path.join(project_root, "run_stage_01.py")
#     subprocess.run([stage_01_venv, stage_01_script], check=True)

# def track_memory_usage():
#     max_memory = 0
#     process = psutil.Process(os.getpid())
#     while True:
#         current_memory = process.memory_info().rss / (1024 ** 3)
#         if current_memory > max_memory:
#             max_memory = current_memory
#         yield max_memory


# memory_tracker = track_memory_usage()

# STAGE_NAME = "Stage 0 Audio Denoising"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     input_file = os.path.join(project_root, "artifacts", "INPUT", "input.mp3")
#     output_dir = os.path.join(project_root, "artifacts", "STAGE_00")
#     model_id = "addy88/wav2vec2-kannada-stt"
#     target_lang = "kan"
#     audio_denoising = AudioDenoising(input_file, output_dir, model_id, target_lang)
#     audio_denoising.main()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Stage 1 Speech Diarization"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     run_stage_01()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Stage 2 Audio Transcription"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     # audio_file = os.path.join(project_root, "artifacts", "STAGE_00", "Cleaned_denoiser_facebook_1_input.mp3")
#     audio_file = os.path.join(project_root, "artifacts", "STAGE_01")
#     output_dir = os.path.join(project_root, "artifacts", "STAGE_02")
#     model_id = "addy88/wav2vec2-kannada-stt"
#     target_lang = "kan"
#     transcription = AudioTranscription(audio_file, output_dir, model_id, target_lang)
#     transcription.main()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Stage 3 Text Punctuation"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     input_file = os.path.join(project_root, "artifacts", "STAGE_02", "02_transcription.txt")
#     output_dir = os.path.join(project_root, "artifacts", "STAGE_03")
#     model_id = "pcs_47lang"
#     punctuation = TextPunctuation(input_file, output_dir, model_id)
#     punctuation.main()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Stage 4 Transliteration"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     input_file = os.path.join(project_root, "artifacts", "STAGE_03", "03_punctuation.txt")
#     output_file = os.path.join(project_root, "artifacts", "STAGE_04", "04_transliteration.txt")
#     transliteration = TextTransliteration(input_file, output_file)
#     transliteration.main()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Stage 5 Translation"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     input_file = os.path.join(project_root, "artifacts", "STAGE_03", "03_punctuation.txt")
#     output_file = os.path.join(project_root, "artifacts", "STAGE_05", "05_translation.txt")
#     model_name = "ai4bharat/indictrans2-indic-en-1B"
#     translation = TextTranslation(input_file, output_file, model_name)
#     translation.main()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Stage 6 English Grammar Correction"
# try:
#     logger.info(f"********** {STAGE_NAME} started **********")
#     input_file = os.path.join(project_root, "artifacts", "STAGE_05", "05_translation.txt")
#     output_file = os.path.join(project_root, "artifacts", "STAGE_06", "06_english_grammar.txt")
#     model_name = "vennify/t5-base-grammar-correction"
#     grammar = GrammarCorrection(input_file, output_file, model_name)
#     grammar.main()
#     logger.info(f"********** {STAGE_NAME} completed **********\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# max_memory_used = next(memory_tracker)
# logger.info(f"Max memory used: {max_memory_used} GB")


from flask import Flask, render_template, request, jsonify
import os
import sys
import subprocess
import csv

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger
from src.Kagapa.pipeline.s_00_DENOISER_DNS_64 import AudioDenoising
from src.Kagapa.pipeline.s_02_ASR_ADDY88 import AudioTranscription
from src.Kagapa.pipeline.s_03_PUNCT_PCS47LANG import TextPunctuation
from src.Kagapa.pipeline.s_04_TRANSLIT_OM import TextTransliteration
from src.Kagapa.pipeline.s_05_TRANSLATE_AI4BHARAT import TextTranslation
from src.Kagapa.pipeline.s_06_ENG_GRAMMAR_VENNIFY import GrammarCorrection
import psutil

app = Flask(__name__)

def track_memory_usage():
    max_memory = 0
    process = psutil.Process(os.getpid())
    while True:
        current_memory = process.memory_info().rss / (1024 ** 3)
        if current_memory > max_memory:
            max_memory = current_memory
        yield max_memory

memory_tracker = track_memory_usage()

def run_stage_01():
    stage_01_venv = "/home/hemanth/only_ml/CDSAML-Kan-Eng-Speect-Text/model2_env/bin/python"
    stage_01_script = os.path.join(project_root, "run_stage_01.py")
    subprocess.run([stage_01_venv, stage_01_script], check=True)

def save_to_csv(stage_name, result):
    csv_file = os.path.join(project_root, "output_results.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Stage Name", "Result"])
        writer.writerow([stage_name, result])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    try:
        results = {}
        
        # Stage 0: Audio Denoising
        STAGE_NAME = "Stage 0 Audio Denoising"
        logger.info(f"********** {STAGE_NAME} started **********")
        input_file = os.path.join(project_root, "artifacts", "INPUT", "input.mp3")
        output_dir = os.path.join(project_root, "artifacts", "STAGE_00")
        model_id = "addy88/wav2vec2-kannada-stt"
        target_lang = "kan"
        audio_denoising = AudioDenoising(input_file, output_dir, model_id, target_lang)
        audio_denoising.main()
        result = "Audio denoised successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        # Stage 1: Speech Diarization
        STAGE_NAME = "Stage 1 Speech Diarization"
        logger.info(f"********** {STAGE_NAME} started **********")
        run_stage_01()
        result = "Speech diarization completed successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        # Stage 2: Audio Transcription
        STAGE_NAME = "Stage 2 Audio Transcription"
        logger.info(f"********** {STAGE_NAME} started **********")
        audio_file = os.path.join(project_root, "artifacts", "STAGE_01")
        output_dir = os.path.join(project_root, "artifacts", "STAGE_02")
        model_id = "addy88/wav2vec2-kannada-stt"
        target_lang = "kan"
        transcription = AudioTranscription(audio_file, output_dir, model_id, target_lang)
        transcription.main()
        result = "Audio transcription completed successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        # Stage 3: Text Punctuation
        STAGE_NAME = "Stage 3 Text Punctuation"
        logger.info(f"********** {STAGE_NAME} started **********")
        input_file = os.path.join(project_root, "artifacts", "STAGE_02", "02_transcription.txt")
        output_dir = os.path.join(project_root, "artifacts", "STAGE_03")
        model_id = "pcs_47lang"
        punctuation = TextPunctuation(input_file, output_dir, model_id)
        punctuation.main()
        result = "Text punctuation completed successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        # Stage 4: Transliteration
        STAGE_NAME = "Stage 4 Transliteration"
        logger.info(f"********** {STAGE_NAME} started **********")
        input_file = os.path.join(project_root, "artifacts", "STAGE_03", "03_punctuation.txt")
        output_file = os.path.join(project_root, "artifacts", "STAGE_04", "04_transliteration.txt")
        transliteration = TextTransliteration(input_file, output_file)
        transliteration.main()
        result = "Text transliteration completed successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        # Stage 5: Translation
        STAGE_NAME = "Stage 5 Translation"
        logger.info(f"********** {STAGE_NAME} started **********")
        input_file = os.path.join(project_root, "artifacts", "STAGE_03", "03_punctuation.txt")
        output_file = os.path.join(project_root, "artifacts", "STAGE_05", "05_translation.txt")
        model_name = "ai4bharat/indictrans2-indic-en-1B"
        translation = TextTranslation(input_file, output_file, model_name)
        translation.main()
        result = "Text translation completed successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        # Stage 6: English Grammar Correction
        STAGE_NAME = "Stage 6 English Grammar Correction"
        logger.info(f"********** {STAGE_NAME} started **********")
        input_file = os.path.join(project_root, "artifacts", "STAGE_05", "05_translation.txt")
        output_file = os.path.join(project_root, "artifacts", "STAGE_06", "06_english_grammar.txt")
        model_name = "vennify/t5-base-grammar-correction"
        grammar = GrammarCorrection(input_file, output_file, model_name)
        grammar.main()
        result = "English grammar correction completed successfully"
        results[STAGE_NAME] = result
        save_to_csv(STAGE_NAME, result)
        logger.info(f"********** {STAGE_NAME} completed **********\n\n")

        max_memory_used = next(memory_tracker)
        logger.info(f"Max memory used: {max_memory_used} GB")
        results["Max Memory Used"] = f"{max_memory_used} GB"
        save_to_csv("Max Memory Used", f"{max_memory_used} GB")

        return render_template('results.html', results=results)

    except Exception as e:
        logger.exception(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
