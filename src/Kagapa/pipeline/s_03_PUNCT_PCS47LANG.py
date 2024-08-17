import os
from punctuators.models import PunctCapSegModelONNX
from typing import List
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.Kagapa import logger

STAGE_NAME = "Text Punctuation Stage"
os.makedirs("../../artifacts/STAGE_03", exist_ok=True)

class TextPunctuation:
    def __init__(self, input_file, output_dir, model_id):
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_id = model_id

    def read_input_texts(self):
        input_texts: List[str] = []
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    input_texts.append(line.strip())  # Remove leading/trailing whitespaces
            return input_texts
        except OSError as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def punctuate_texts(self, input_texts):
        model = PunctCapSegModelONNX.from_pretrained(self.model_id)
        try:
            results: List[List[str]] = model.infer(input_texts)
            return results
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def save_punctuation_results(self, input_texts, results):
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, "03_punctuation.txt")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for input_text, output_texts in zip(input_texts, results):
                    f.write(f"Input: {input_text}\n")
                    f.write("Outputs:\n")
                    for text in output_texts:
                        f.write(f"\t{text}\n")
                    f.write("\n")
            logger.info(f"Punctuation results saved to: {output_file}")
        except OSError as e:
            logger.error(f"Error writing output file: {e}")
            raise

    def main(self):
        input_texts = self.read_input_texts()
        results = self.punctuate_texts(input_texts)
        self.save_punctuation_results(input_texts, results)

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        input_file = "../../artifacts/STAGE_02/02_transcription.txt"
        output_dir = "../../artifacts/STAGE_03"
        model_id = "pcs_47lang"

        obj = TextPunctuation(input_file, output_dir, model_id)
        obj.main()
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e

