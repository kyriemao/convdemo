from http.cookiejar import LoadError
import torch
import torch.nn as nn

import os
from os.path import join as oj


import time
import json
import copy
import faiss
import pickle
import numpy as np
from tqdm import tqdm

from abc import abstractmethod
from pyserini.search.lucene import LuceneSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import (RobertaConfig, AutoModelForSequenceClassification, AutoTokenizer,
                          RobertaForSequenceClassification, RobertaTokenizer)
from IPython import embed
from utils import pload, pstore
from spacy.lang.en import English
nlp = English()

class Rewriter:
    def __init__(self, args):
        self.tokenizer = T5Tokenizer.from_pretrained(args.rewriter_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.rewriter_path)
        self.device = args.device
        self.t5.to(self.device)
        # self.max_response_length = 64
        self.max_query_length = 32
        self.max_seq_length = 128

    def __call__(self, cur_utt_text, ctx_utts_text):
        # build input
        ctx_utts_text.reverse()
        src_seq = []
        src_seq.append(cur_utt_text)
        for i in range(len(ctx_utts_text)):
            src_seq.append(ctx_utts_text[i])                
        src_seq = " [SEP] ".join(src_seq)

        bt_src_encoding = self.tokenizer(src_seq, 
                                        padding="longest", 
                                        max_length=self.max_seq_length, 
                                        truncation=True, 
                                        return_tensors="pt")
        input_ids, attention_mask = bt_src_encoding.input_ids, bt_src_encoding.attention_mask
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        begin_time = time.time()
        outputs = self.t5.generate(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   do_sample=False,
                                   max_length=self.max_query_length)
        end_time = time.time()
        print("rewriting time cost: {}".format(end_time - begin_time))
        rewrite_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewrite_text


class Retriever:
    def __init__(self, args):
        self.retriever_path = args.retriever_path
        self.index_path = args.index_path
        self.device = args.device
        self.top_k = 20

        self.max_query_length = args.max_query_length
        self.max_response_length = args.max_response_length
        self.max_seq_length = args.max_seq_length
    
    @abstractmethod
    def __call__(self, query, context=None):
        raise NotImplementedError


class SparseRetriever(Retriever):
    def __init__(self, args):
        super().__init__(args)
        self.searcher = LuceneSearcher(self.index_path)
        self.bm25_k1 = 0.82 
        self.bm25_b = 0.68
        self.searcher.set_bm25(self.bm25_k1, self.bm25_b)
 
    def __call__(self, query):
        hits = self.searcher.search(query, k = self.top_k)
        results = []
        for hit in hits:
            hit = json.loads(hit.raw)
            hit['url'] = "https://url_sparse"
            results.append(hit)
        return results

# ANCE dense retrieval model
class ANCE(RobertaForSequenceClassification):
    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768) # ANCE has
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    
    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)

class DenseRetriever(Retriever):
    def __init__(self, args):
        super().__init__(args)
        self.doc_embeddings_path = args.doc_embeddings_path
        self.docfaissidx_to_realdocid_path = args.docfaissidx_to_realdocid_path
        self.collection_path = args.collection_path
        
        start_time = time.time()
        self.faiss_index = self.build_faiss_index()
        self.model, self.tokenizer = self.load_model(args.retriever_path)
        self.model.to(self.device)
        
        self.docid_map, self.collection = self.load_corpus()
        print("Dense Retriever setup time: {}s".format(time.time() - start_time))
        # self.collection = None




    def __call__(self, query, context=None):
        # cal query embedding
        query_emb = self.cal_query_emb(query, context)
        query_emb = query_emb.detach().cpu().numpy()
        
        # dense retrieval
        retrieved_pids = self.search_with_faiss(query_emb)
        retrieved_pids = retrieved_pids[0]
        
        # organize results
        results = []
        for pid in retrieved_pids:
            pid = int(pid)
            results.append({"doc_id": pid, "contents": self.collection[pid], "url":"https://url_dense"})
        return results

    def load_corpus(self):
        collection = {}
        with open(self.collection_path, "r") as f:
            for line in tqdm(f):
                try:
                    pid, passage = line.strip().split('\t')
                    pid = int(pid)
                except:
                    continue
                collection[pid] = passage
        docid_map = pload(self.docfaissidx_to_realdocid_path)

        return docid_map, collection

    def load_model(self, model_path):
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE.from_pretrained(model_path, config=config)
        return model, tokenizer

    def cal_query_emb(self, query, context):
        if context is not None:
            input_ids = get_conv_bert_input_no_response(query, 
                                                        context, 
                                                        self.tokenizer, 
                                                        self.max_query_length, 
                                                        self.max_response_length, 
                                                        self.max_seq_length)
        else:
            input_ids = self.tokenizer.encode(query,
                                            add_special_tokens=True, 
                                            max_length=self.max_query_length, 
                                            truncation=True)
        input_ids = torch.tensor(input_ids).to(self.device).view(1, -1)
        attention_mask = torch.ones(input_ids.size()).to(self.device).long()
        query_emb = self.model(input_ids, attention_mask)
        return query_emb

    def build_faiss_index(self):
        index = faiss.read_index(self.index_path)
        print("faiss index is trained: {}".format(index.is_trained))

        print("load doc embeddings...")
        start_time = time.time()
        doc_embeddings = pload(self.doc_embeddings_path)
        print("spend {} seconds to load embeddings.".format(time.time() - start_time))
        
        print("add doc embeddings into faiss index...")
        index.add(doc_embeddings)
        print("index add embedding ok!")

        return index

    def search_with_faiss(self, query_emb):
        tb = time.time()
        D, I = self.faiss_index.search(query_emb, self.top_k)
        elapse = time.time() - tb
        print("dense retrieval time cost: {}".format(elapse))

        doc_idxs = self.docid_map[I] # faiss_doc_idx -> real_doc_id_in_collection
        return doc_idxs



class Reranker:
    def __init__(self, args):
        self.max_seq_length = args.max_seq_length
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.reranker_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.reranker_path)

    def __call__(self, query, context, candidates):
        inputs_pairs = []
        for candidate in candidates:
            inputs_pairs.append([query, candidate['contents']])

        if context is None:
            encoded_inputs = self.tokenizer.batch_encode_plus(inputs_pairs,
                                                            add_special_tokens=True,
                                                            padding='max_length',
                                                            max_length=self.max_seq_length,
                                                            truncation=True,
                                                            return_tensors='pt')
            
            with torch.no_grad():
                output = self.model(**encoded_inputs).logits
            output = output.squeeze(-1)
            topk_values = torch.topk(output, 3).values
            topk_idxs = torch.topk(output, 3).indices

            res = []    # (content, score, url)
            for i in range(len(topk_idxs)):
                content = candidates[topk_idxs[i]]['contents']
                score = round(topk_values[i].item(), 3)
                url = candidates[topk_idxs[i]]['url']
                first_sent = content.split('.')[0]
                res.append((content, score, url, first_sent))
            return res
        else:
            raise NotImplementedError
        

# def get_conv_bert_input_no_response(query, context, tokenizer, max_query_length, max_response_length, max_seq_length):
#     input_ids = []
#     encoded_query = tokenizer.encode(query,
#                                     add_special_tokens=True, 
#                                     max_length=max_query_length, 
#                                     truncation=True)
#     input_ids.extend(encoded_query)
#     if len(context) > 1:
#         last_response = context[-1]
#         encoded_response = tokenizer.encode(last_response, 
#                                             add_special_tokens=True, 
#                                             max_length=max_response_length, 
#                                             truncation=True)[1:] # remove [CLS]
#         input_ids.extend(encoded_response)
    
#     for i in range(len(context) - 2, -1, -2):
#         encoded_history = tokenizer.encode(context[i],
#                                             add_special_tokens=True, 
#                                             max_length=max_query_length, 
#                                             truncation=True)[1:] # remove [CLS]
#         input_ids.extend(encoded_history)
#         if len(input_ids) > max_seq_length:
#             input_ids = input_ids[:max_seq_length - 1] + encoded_history[-1]    # ensure [SEP] ended
    
#     return input_ids

def get_conv_bert_input_no_response(query, context, tokenizer, max_query_length, max_response_length, max_seq_length):
    input_ids = []
    encoded_query = tokenizer.encode(query,
                                    add_special_tokens=True, 
                                    max_length=max_query_length, 
                                    truncation=True)
    input_ids.extend(encoded_query)
    for i in range(len(context) - 1, -1, -1):
        encoded_history = tokenizer.encode(context[i],
                                            add_special_tokens=True, 
                                            max_length=max_query_length, 
                                            truncation=True)[1:] # remove [CLS]
        input_ids.extend(encoded_history)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length - 1] + encoded_history[-1]    # ensure [SEP] ended
            break
    
    return input_ids



def get_t5qr_input(query, context, tokenizer, max_seq_length):
    src_text = "|||".join(context + [query])
    src_text = " ".join([tok.text for tok in nlp(src_text)])
    input_ids = tokenizer.encode(src_text,
                     add_special_tokens=True, 
                     max_length=max_seq_length, 
                     truncation=True)
    return input_ids