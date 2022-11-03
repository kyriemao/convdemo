from transformers import BartTokenizer, BartForConditionalGeneration
from clarification.codes.utils import *
from clarification.codes.config import *
from IPython import embed

class BART:
    def __init__(self, bart_model_path, output_max_len=64):
        self.tokenizer = BartTokenizer.from_pretrained(bart_model_path)
        self.model = BartForConditionalGeneration.from_pretrained(bart_model_path)

        self.tokenizer.add_tokens(['[SEP]', '[ISEP]', '[QSEP]'])
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.output_max_len = output_max_len

    def predict(self, input_text, num_beams=20):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        if len(input_ids[0]) > 1024:
            input_ids = input_ids[:, :1024]

        # Beam Search
        outputs = self.model.generate(input_ids, max_length=self.output_max_len, num_beams=num_beams,
                                      early_stopping=True, num_return_sequences=num_beams)
        beam_output_strs = []
        for j, output in enumerate(outputs):
            output_str = self.tokenizer.decode(output, skip_special_tokens=True)
            beam_output_strs.append(output_str)

        return beam_output_strs
