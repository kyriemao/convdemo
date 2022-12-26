export CUDA_VISIBLE_DEVICES=5

python myapp.py --mode="cqr" \
--rewriter_path="/data1/kelong_mao/workspace/T5QR/outputs/t5qr_qrecc/checkpoints/epoch-5" \
--index_path="/data1/kelong_mao/indexes/cast/ance/cast_faiss.index" \
--retriever_path="/data1/kelong_mao/pretrained_models/ance-msmarco" \
--retriever_type="dense" \
--reranker_path="cross-encoder/ms-marco-MiniLM-L-12-v2" \
--collection_path="/data1/kelong_mao/collections/cast/cast_collection/raw.tsv" \
--doc_embeddings_path="/data1/kelong_mao/indexes/cast/ance/all_embeddings.pb" \
--docfaissidx_to_realdocid_path="/data1/kelong_mao/indexes/cast/ance/all_embeddings_ids.pb"
