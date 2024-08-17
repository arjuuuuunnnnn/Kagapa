import os
from om_transliterator import Transliterator
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger

STAGE_NAME = "Text Transliteration Stage"
os.makedirs("../../artifacts/STAGE_04", exist_ok=True)

class TextTransliteration:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.transliterator = Transliterator()

    def read_punctuated_text(self):
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return lines
        except OSError as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def extract_text_to_transliterate(self, lines):
        start_reading = False
        original_text = ""
        try:
            for line in lines:
                if start_reading:
                    original_text += line.strip() + " "
                if "Outputs:" in line:
                    start_reading = True
            return original_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text to transliterate: {e}")
            raise

    def transliterate_text(self, text):
        try:
            transliterated_text = self.transliterator.knda_to_latn(text)
            return transliterated_text
        except Exception as e:
            logger.error(f"Error during transliteration: {e}")
            raise

    def save_transliterated_text(self, transliterated_text):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(transliterated_text)
            logger.info(f"Transliterated text saved to: {self.output_file}")
        except OSError as e:
            logger.error(f"Error writing output file: {e}")
            raise

    def main(self):
        lines = self.read_punctuated_text()
        original_text = self.extract_text_to_transliterate(lines)
        transliterated_text = self.transliterate_text(original_text)
        self.save_transliterated_text(transliterated_text)

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        input_file = "../../artifacts/STAGE_03/03_punctuation.txt"
        output_file = "../../artifacts/STAGE_04/04_transliteration.txt"
        obj = TextTransliteration(input_file, output_file)
        obj.main()
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e

