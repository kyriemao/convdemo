export CUDA_VISIBLE_DEVICES=5

python myapp.py --mode="cqr" \
--rewriter_path="/data1/kelong_mao/pretrained_models/t5qr_trained_on_qrecc" \
--index_path="/data1/kelong_mao/indexes/cast/bm25" \
--retriever_path="" \
--retriever_type="sparse" \
--reranker_path="cross-encoder/ms-marco-MiniLM-L-12-v2" 