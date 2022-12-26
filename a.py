import faiss
import pickle
import numpy as np
import os
from os.path import join as oj

index = faiss.index_factory(768, 'IVF1024_HNSW32,SQ8')
index_block_num = 6
index_path = "/data1/kelong_mao/indexes/cast/ance/doc_embeddings"
from IPython import embed


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path, high_protocol = True):
    with open(path, 'wb') as f:
        if high_protocol:  
            pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))



index = faiss.read_index('/data1/kelong_mao/indexes/cast/ance/cast_faiss.index')
embed()
input()
a = np.random.random((10000000,768)).astype(np.float32)
# index.add(a)
embed()
input()



# for block_id in range(index_block_num):
#     # with open(oj(index_path, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
#     #     passage_embedding = pickle.load(handle)
#     with open(oj(index_path, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
#         passage_embedding2id = pickle.load(handle)
#         if isinstance(passage_embedding2id, list):
#             passage_embedding2id = np.array(passage_embedding2id)


#     embeddings.append(passage_embedding2id)

# embeddings = np.concatenate(embeddings, axis=0)
# pstore(embeddings, '/data1/kelong_mao/indexes/cast/ance/all_embeddings_ids.pb')
#     embeddings.append(passage_embedding)
#     print("{} load ok".format(block_id))

# embeddings = np.concatenate(embeddings, axis=0)
# pstore(embeddings, '/data1/kelong_mao/indexes/cast/ance/concat_embeddings.pb')
# print("store ok")
# index.train(embeddings)

# print("writing the index to the ...")
# faiss.write_index(index, '/data1/kelong_mao/indexes/cast/ance/faiss.index')
# print("dump index ok")