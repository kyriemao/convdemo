from http.cookiejar import LoadError
import torch
import torch.nn as nn

import os
from os.path import join as oj

import copy
import faiss
import pickle
import numpy as np
from time import time

from abc import abstractmethod
from pyserini.search.lucene import LuceneSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import (RobertaConfig,
                          RobertaForSequenceClassification, RobertaTokenizer)


class Rewriter:
    def __init__(self, model_path, device):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_path)
        self.t5.to(device)
        self.max_response_length = 64
        self.max_query_length = 32
        self.max_seq_length = 256

    def rewrite(self, query, context):
        input_ids = get_conv_bert_input(query, context, 
                                        self.tokenizer, 
                                        self.max_query_length, 
                                        self.max_response_length, 
                                        self.max_seq_length)
        input_ids = input_ids.to(self.device)
        outputs = self.t5.generate(input_ids)
        rewrite_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewrite_text


class Retriever:
    def __init__(self, args):
        self.model_path = args.model_path
        self.index_path = args.index_path
        self.device = args.device
        self.top_k = 50
    
    @abstractmethod
    def search(self, query, context=None):
        raise NotImplementedError


class SparseRetriever(Retriever):
    def __init__(self, args):
        super.__init__(self, args)
        self.searcher = LuceneSearcher(self.index_path)
        self.bm25_k1 = 0.82 
        self.bm25_b = 0.68
        self.searcher.set_bm25(self.bm25_k1, self.bm25_b)
 
    def search(self, query):
        hits = self.searcher.search(query, k = self.top_k)
        return hits

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
        super.__init__(self, args)
        self.n_gpu_for_faiss = args.n_gpu_for_faiss
        self.faiss_index = self.build_faiss_index()
        self.index_block_num = args.index_block_num

        self.model, self.tokenizer = self.load_model()
        self.model.to(self.device)



    def search(self, query, context=None):
        query_emb = self.cal_query_emb(query, context)
        retrieved_scores_mat, retrieved_pid_mat = self.search_one_by_one_with_faiss(query_emb)
        return retrieved_pid_mat


    def load_model(self):
        config = RobertaConfig.from_pretrained(
            self.model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            self.model_path,
            do_lower_case=True
        )
        model = ANCE.from_pretrained(self.model_path, config=config)
        return model, tokenizer

    def cal_query_emb(self, query, context):
        if context is not None:
            input_ids = get_conv_bert_input(query, 
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
        input_ids = input_ids.to(self.device)       
        query_emb = self.model(input_ids)
        return query_emb

    def build_faiss_index(self):
        print("build faiss index...")
        ngpu = self.n_gpu_for_faiss
        gpu_resources = []
        tempmem = -1

        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            if tempmem >= 0:
                res.setTempMemory(tempmem)
            gpu_resources.append(res)

        cpu_index = faiss.IndexFlatIP(768)          
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        return gpu_index

    def search_one_by_one_with_faiss(self, query_emb):
        merged_candidate_matrix = None
        if self.index_block_num < 0:
            # automaticall get the number of passage blocks
            for filename in os.listdir(self.index_path):
                try:
                    self.index_block_num = max(self.index_block_num, int(filename.split(".")[1]) + 1)
                except:
                    continue
            print("Automatically detect that the number of doc blocks is: {}".format(self.index_block_num))
        
        for block_id in range(self.index_block_num):
            print("Loading passage block " + str(block_id))
            passage_embedding = None
            passage_embedding2id = None
            try:
                with open(oj(self.index_path, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
                    passage_embedding = pickle.load(handle)
                with open(oj(self.index_path, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
                    passage_embedding2id = pickle.load(handle)
                    if isinstance(passage_embedding2id, list):
                        passage_embedding2id = np.array(passage_embedding2id)
            except:
                raise LoadError    
            
            print.info('passage embedding shape: ' + str(passage_embedding.shape))

            passage_embeddings = np.array_split(passage_embedding, args.num_split_block)
            passage_embedding2ids = np.array_split(passage_embedding2id, args.num_split_block)
            for split_idx in range(len(passage_embeddings)):
                passage_embedding = passage_embeddings[split_idx]
                passage_embedding2id = passage_embedding2ids[split_idx]
                
                print.info("Adding block {} split {} into index...".format(block_id, split_idx))
                self.faiss_index.add(passage_embedding)
                
                # ann search
                tb = time.time()
                D, I = self.faiss_index.search(query_emb, self.top_k)
                elapse = time.time() - tb

                candidate_id_matrix = passage_embedding2id[I] # passage_idx -> passage_id
                D = D.tolist()
                candidate_id_matrix = candidate_id_matrix.tolist()
                candidate_matrix = []

                for score_list, passage_list in zip(D, candidate_id_matrix):
                    candidate_matrix.append([])
                    for score, passage in zip(score_list, passage_list):
                        candidate_matrix[-1].append((score, passage))
                    assert len(candidate_matrix[-1]) == len(passage_list)
                assert len(candidate_matrix) == I.shape[0]

                self.faiss_index.reset()
                del passage_embedding
                del passage_embedding2id

                if merged_candidate_matrix == None:
                    merged_candidate_matrix = candidate_matrix
                    continue
                
                # merge
                merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
                merged_candidate_matrix = []
                for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                                candidate_matrix):
                    p1, p2 = 0, 0
                    merged_candidate_matrix.append([])
                    while p1 < self.top_k and p2 < self.top_k:
                        if merged_list[p1][0] >= cur_list[p2][0]:
                            merged_candidate_matrix[-1].append(merged_list[p1])
                            p1 += 1
                        else:
                            merged_candidate_matrix[-1].append(cur_list[p2])
                            p2 += 1
                    while p1 < self.top_k:
                        merged_candidate_matrix[-1].append(merged_list[p1])
                        p1 += 1
                    while p2 < self.top_k:
                        merged_candidate_matrix[-1].append(cur_list[p2])
                        p2 += 1

        merged_D, merged_I = [], []

        for merged_list in merged_candidate_matrix:
            merged_D.append([])
            merged_I.append([])
            for candidate in merged_list:
                merged_D[-1].append(candidate[0])
                merged_I[-1].append(candidate[1])
        merged_D, merged_I = np.array(merged_D), np.array(merged_I)

        return merged_D, merged_I


class Reranker:
    pass


def get_conv_bert_input(query, context, tokenizer, max_query_length, max_response_length, max_seq_length):
    input_ids = []
    encoded_query = tokenizer.encode(query,
                                    add_special_tokens=True, 
                                    max_length=max_query_length, 
                                    truncation=True)
    input_ids.extend(encoded_query)
    last_response = context[-1]
    encoded_response = tokenizer.encode(last_response, 
                                        add_special_tokens=True, 
                                        max_length=max_response_length, 
                                        truncation=True)[1:] # remove [CLS]
    input_ids.extend(encoded_response)
    
    for i in range(len(context) - 2, -1, -2):
        encoded_history = tokenizer.encode(context[i],
                                            add_special_tokens=True, 
                                            max_length=max_query_length, 
                                            truncation=True)[1:] # remove [CLS]
        input_ids.extend(encoded_history)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length - 1] + encoded_history[-1]    # ensure [SEP] ended
    
    return input_ids