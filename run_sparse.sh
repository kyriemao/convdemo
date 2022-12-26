export CUDA_VISIBLE_DEVICES=5

python myapp.py --mode="cqr" \
--rewriter_path="/data1/kelong_mao/workspace/T5QR/outputs/t5qr_qrecc/checkpoints/epoch-5" \
--index_path="/data1/kelong_mao/indexes/cast/bm25" \
--retriever_path="" \
--retriever_type="sparse" \
--reranker_path="cross-encoder/ms-marco-MiniLM-L-12-v2" 
