class SearchBot:
    def __init__(self, mode="cqr", rewriter=None, retriever=None, reranker=None):
        self.rewriter = rewriter
        self.rewriter = retriever
        self.reranker = reranker
        # mode can be "cqr" or "cdr"
        self.mode = mode

    def search(self, query, context):
        if self.mode == "cqr":
            rewrite = self.rewriter(query, context)
            candidates = self.retriever(query=rewrite)
            top1_res = self.reranker(query=rewrite, context=None, candidates=candidates)
            return top1_res
        elif self.mode == "cdr":
            candidates = self.retriever(query=query, context=context)
            top1_res = self.reranker(query=query, context=context, candidates=candidates)
            return top1_res
        else:
            raise NotImplementedError


mode = "cqr"
rewriter = ""
retriever = ""
reranker = ""

search_bot = SearchBot(mode, rewriter, retriever, reranker)