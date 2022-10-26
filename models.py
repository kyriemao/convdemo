from transformers import T5Tokenizer, T5ForConditionalGeneration

class Rewriter:
    def __init__(self, model_path, device):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_path)
        self.t5.to(device)
        self.max_response_length = 64
        self.max_query_length = 32
        self.max_seq_length = 256

    def rewrite(self, query, context):
        input_ids = []
        encoded_query = self.tokenizer.encode(query,
                                              add_special_tokens=True, 
                                              max_length=self.max_query_length, 
                                              truncation=True)
        input_ids.extend(encoded_query)
        last_response = context[-1]
        encoded_response = self.tokenizer.encode(last_response, 
                                                 add_special_tokens=True, 
                                                 max_length=self.max_response_length, 
                                                 truncation=True)[1:] # remove [CLS]
        input_ids.extend(encoded_response)
        
        for i in range(len(context) - 2, -1, -2):
            encoded_history = self.tokenizer.encode(context[i],
                                                    add_special_tokens=True, 
                                                    max_length=self.max_query_length, 
                                                    truncation=True)[1:] # remove [CLS]
            if len(input_ids) + len(encoded_history) > self.max_seq_length:
                break
            input_ids.extend(encoded_history)
        
        input_ids = input_ids.to(self.device)
        outputs = self.t5.generate(input_ids)
        rewrite_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewrite_text


class Retriever:
    def __init__(self) -> None:
        pass

class Reranker:
    pass
