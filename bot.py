from IPython import embed
import json
from models import Rewriter, SparseRetriever,  DenseRetriever, Reranker
from clarification.codes.main import load_models, clarify
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


BIG_STOP_WORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

class SearchBot:
    def __init__(self, args):
        self.rewriter = Rewriter(args)
        if args.retriever_type == "sparse":
            self.retriever = SparseRetriever(args)
        else:
            self.retriever = DenseRetriever(args)
        self.reranker = Reranker(args)
        # mode can be "cqr" or "cdr"
        self.mode = args.mode

        # for clarification
        sentence_model = SentenceTransformer("msmarco-bert-base-dot-v5")
        self.kw_model = KeyBERT(sentence_model)
        self.bartres, self.srqg, self.intent_cqgm = load_models()
        self.keyword_threshold = 2


    def search(self, query, context):

        # anyway, we always need rewrite to judge clarification
        rewrite = self.rewriter(query, context)
        print("rewrite: ", rewrite)

        # keyword extraction
        keywords = self.kw_model.extract_keywords(rewrite, keyphrase_ngram_range = (1,2), top_n=3, stop_words=BIG_STOP_WORDS)
        keyword_set = set()
        for kw in keywords:
            for word in kw[0].split():
                keyword_set.add(word)
        
        if len(keyword_set) <= self.keyword_threshold:
            # need clarification (or query recommendation)
            kw_query = " ".join(keyword_set)
            clarification_res = clarify(query=kw_query,
                                        bartres=self.bartres,
                                        srqg=self.srqg,
                                        intent_cqgm=self.intent_cqgm,
                                        RETRIEVAL=True,
                                        SUGGESTION=True,
                                        CLARIFICATION=True,
                                        INFORMATIVE=True,
                                        INTENT_VERB=True)
            clarify_question = clarification_res[2]
            clarify_item_candidates = str(clarification_res[3])
        else:
            clarify_item_candidates, clarify_question = None, None

        # retrieval & rerank
        if self.mode == "cqr":
            retrieval_candidates = self.retriever(query=rewrite)
            top3_res = self.reranker(query=rewrite, context=None, candidates=retrieval_candidates)
        elif self.mode == "cdr":
            retrieval_candidates = self.retriever(query=query, context=context)
            top3_res = self.reranker(query=query, context=context, candidates=retrieval_candidates)
        else:
            raise NotImplementedError

        # organize output
        res = {}
        res['rewrite'] = rewrite
        res['top1_search_result'] = top3_res[0]
        res['other_search_results'] = top3_res[1:]
        res['clarify_question'] = clarify_question
        res['clarify_item_candidates'] = clarify_item_candidates
        res = json.dumps(res)
        return res
    
        



