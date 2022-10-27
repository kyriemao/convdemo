from IPython import embed
from models import Rewriter, SparseRetriever, Reranker

class SearchBot:
    def __init__(self, args):
        self.rewriter = Rewriter(args)
        self.retriever = SparseRetriever(args)
        self.reranker = Reranker(args)
        # mode can be "cqr" or "cdr"
        self.mode = args.mode

    def search(self, query, context):
        if self.mode == "cqr":
            rewrite = self.rewriter(query, context)
            print("rewrite: ", rewrite)
            candidates = self.retriever(query=rewrite)
            top1_res = self.reranker(query=rewrite, context=None, candidates=candidates)
            return top1_res, rewrite
        elif self.mode == "cdr":
            candidates = self.retriever(query=query, context=context)
            top1_res = self.reranker(query=query, context=context, candidates=candidates)
            return top1_res
        else:
            raise NotImplementedError


