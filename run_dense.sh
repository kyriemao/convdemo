export CUDA_VISIBLE_DEVICES=5

python myapp.py --mode="cqr" \
--rewriter_path="./pretrained_models/t5qr_trained_on_qrecc" \
--index_path="./indexes/cast/ance/cast_faiss.index" \
--retriever_path="./pretrained_models/ance-msmarco" \
--retriever_type="dense" \
--reranker_path="cross-encoder/ms-marco-MiniLM-L-12-v2" \
--collection_path="./collections/cast/cast_collection/raw.tsv" \
--doc_embeddings_path="./indexes/cast/ance/all_embeddings.pb" \
--docfaissidx_to_realdocid_path="./indexes/cast/ance/all_embeddings_ids.pb"
