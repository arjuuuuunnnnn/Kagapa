import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys

tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'IndicTransTokenizer'))
sys.path.append(tokenizer_path)

from IndicTransTokenizer import IndicProcessor

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.Kagapa import logger

STAGE_NAME = "Text Translation Stage"
os.makedirs("../../artifacts/STAGE_05", exist_ok=True)

class TextTranslation:
    def __init__(self, input_file, output_file, model_name):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.ip = IndicProcessor(inference=True)
        self.src_lang = "kan_Knda"
        self.tgt_lang = "eng_Latn" 

    def read_input_texts(self):
        try:
            input_texts = []
            current_input_text = ""
            with open(self.input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().lower().startswith("outputs:"):
                        if current_input_text:
                            input_texts.append(current_input_text.strip())
                        current_input_text = ""
                    else:
                        current_input_text += line
                if current_input_text:
                    input_texts.append(current_input_text.strip())
            return input_texts
        except OSError as e:
            logger.error(f"Error reading input file: {e}")
            raise

    def preprocess_texts(self, texts):
        try:
            return self.ip.preprocess_batch(texts, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def generate_translations(self, batch):
        try:
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            generated_tokens = generated_tokens.cpu()

            with self.tokenizer.as_target_tokenizer():
                generated_tokens = self.tokenizer.batch_decode(
                    generated_tokens.tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            return self.ip.postprocess_batch(generated_tokens, lang=self.tgt_lang)
        except Exception as e:
            logger.error(f"Error during translation generation: {e}")
            raise

    def save_translations(self, translations):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, "a", encoding="utf-8") as f:
                for translation in translations:
                    if not translation.startswith(f"{self.src_lang}:") and not translation.startswith(f"{self.tgt_lang}:"):
                        f.write(f"Output: {translation}\n")
            logger.info(f"Translations saved to: {self.output_file}")
        except OSError as e:
            logger.error(f"Error writing output file: {e}")
            raise

    def main(self):
        input_texts = self.read_input_texts()
        if not input_texts:
            logger.warning("No input texts found in the file.")
            return

        batch_size = 8
        for i in range(0, len(input_texts), batch_size):
            batch_input_texts = input_texts[i:i + batch_size]
            batch = self.preprocess_texts(batch_input_texts)
            translations = self.generate_translations(batch)
            self.save_translations(translations)

if __name__ == "__main__":
    try:
        logger.info(f"********** stage {STAGE_NAME} started **********")
        input_file = "../../artifacts/STAGE_03/03_punctuation.txt"
        output_file = "../../artifacts/STAGE_05/05_translation.txt"
        model_name = "ai4bharat/indictrans2-indic-en-1B"

        obj = TextTranslation(input_file, output_file, model_name)
        obj.main()
        logger.info(f"********** stage {STAGE_NAME} completed **********")
    except Exception as e:
        logger.exception(e)
        raise e

