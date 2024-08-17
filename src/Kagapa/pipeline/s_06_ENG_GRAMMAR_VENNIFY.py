import os
from transformers import pipeline
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger

STAGE_NAME = "English Grammar Correction Stage"
os.makedirs("../../artifacts/STAGE_06", exist_ok=True)

class GrammarCorrection:
    def __init__(self, input_file, output_file, model_name):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.corrector = pipeline("text2text-generation", model=model_name)

    def get_text_from_file(self):
        try:
            with open(self.input_file, 'r', encoding='utf-8') as file:
                content = file.read()

            # Find the second occurrence of "Output:"
            second_output_index = content.find("Output:", content.find("Output:") + 1)
            if second_output_index == -1:
                raise ValueError("The file does not contain a second 'Output:'")

            # Extract text after the second "Output:"
            text_to_correct = content[second_output_index + len("Output:"):].strip()

            return text_to_correct
        except OSError as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def grammar_check(self, text):
        try:
            # Generate the corrected text
            corrected = self.corrector(text, max_length=len(text) + 50)[0]['generated_text']
            return corrected
        except Exception as e:
            logger.error(f"Error during grammar correction: {e}")
            raise

    def write_text_to_file(self, text):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', encoding='utf-8') as file:
                file.write("Output: " + text)
            logger.info(f"Corrected text saved to: {self.output_file}")
        except OSError as e:
            logger.error(f"Error writing output file: {e}")
            raise

    def main(self):
        input_text = self.get_text_from_file()
        corrected_text = self.grammar_check(input_text)
        self.write_text_to_file(corrected_text)
        logger.info(f"Original: {input_text}")
        logger.info(f"Corrected: {corrected_text}")

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        input_file = "../../artifacts/STAGE_05/05_translation.txt"
        output_file = "../../artifacts/STAGE_06/06_english_grammar.txt"
        model_name = "vennify/t5-base-grammar-correction"

        obj = GrammarCorrection(input_file, output_file, model_name)
        obj.main()
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e

