# A Conversaitonal Search Demo

![image](https://raw.githubusercontent.com/kyriemao/convdemo/v1/demo.png)

## Environment
```
pip install -r requirements
```
Note that the `pyserini` package needs Java 11. 


## Parameters

<!-- These data files should be prepared in advance and put into the corresponding paths. -->
- mode: can be `cqr` or `cdr`. See the below `Instruction`.
- retrieval_type: sparse or dense.
- collection_path: the corpus data (only needed for dense).
- index_path: for sparse, it is an inverted index pre-built using pyserini. For dense, it is a trained faiss index. 
- doc_embeddings_path: the path of all document(passage) embeddings stored in numpy (only needed for dense).
- docfaissidx_to_realdocid_path: the doc id mapper stored in numpy. For example, [0,1,2,3] -> [101, 45, 34, 28], where the former is the doc id in the faiss while the latter is the real doc id in the collection (only needed for dense).

- rewriter_path: the path of a conversational query rewriter model.
- retriever_path: the path of a trained retriever model (only needed for cqr-dense and cdr).
- reranker_path: the path of a reranker model. 



## Instruction

We support two types of conversational search methods:
- cqr: conversational query rewriting
  - rewriter + retriever (can be sparse or dense) + BERT reranker
- cdr: conversational dense retrieval
  - conversational dense retriever + conversational reranker

Currently, the cdr mode still needed refined. We provide the example scripts for running `cqr-sparse` and `cqr-dense`.

```bash
bash run_sparse.sh  # cqr-sparse. 
bash run_dense.sh # cqr-dense. It takes around 10 minues to load all of data (~38M passages). The size of embeddings is 110G. The index.add(embeddings) in faiss takes the majority of time cost.
```

BTW, in this demo, we alway first perform query rewriting because we need the rewrite to perform query clarification.
