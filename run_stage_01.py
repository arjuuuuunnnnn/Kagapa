import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger
from src.Kagapa.pipeline.s_01_DASHBOARD_FLAVOUR import SpeakerDiarization

STAGE_NAME = "Stage 1 Speech Diarization"
try:
    logger.info(f"********** {STAGE_NAME} started **********")
    input_dir = os.path.join(project_root, "artifacts", "STAGE_00")
    output_dir = os.path.join(project_root, "artifacts", "STAGE_01")
    auth_token = "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"
    audio_split = SpeakerDiarization(input_dir, output_dir, auth_token)
    audio_split.main()
    logger.info(f"********** {STAGE_NAME} completed **********\n\n")
except Exception as e:
    logger.exception(e)
    raise e
