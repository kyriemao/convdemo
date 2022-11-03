export CUDA_VISIBLE_DEVICES=5

python myapp.py --mode="cqr" \
--rewriter_path="/data1/kelong_mao/pretrained_models/t5qr_trained_on_qrecc" \
--index_path="/data1/kelong_mao/indexes/cast/ance/doc_embeddings" \
--retriever_path="/data1/kelong_mao/pretrained_models/ance-msmarco" \
--retriever_type="dense" \
--reranker_path="cross-encoder/ms-marco-MiniLM-L-12-v2" \
--collection_path="/data1/kelong_mao/collections/cast/cast_collection/raw.tsv" 