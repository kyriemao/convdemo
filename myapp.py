
from multiprocessing.sharedctypes import Value
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

import json
import torch
import time
import argparse
from bot import SearchBot
from IPython import embed

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'


context = []

@app.route("/", methods=['GET', 'POST'])
def home():
	global context
	context = []
	print("context: ", context)
	return render_template("home.html", val1=time.time())


@app.route("/get")
def get_bot_response():
	global context
	user_query = request.args.get('msg')
	if user_query == "DELETE":
		context = context[:-1]
		return ""
	if search_bot.mode == "cqr":
		response_obj = search_bot.search(user_query, context)
	else:
		response_obj = search_bot.search(user_query, context)
		# showing_response = respone
	context.append(user_query)
	# context.append(json.loads(response_obj)['rewrite'])
	print("current context: ", context)
	return response_obj


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--rewriter_path", type=str, required=True)
	parser.add_argument("--retriever_path", type=str, default="")
	parser.add_argument("--reranker_path", type=str, default="")
	parser.add_argument("--index_path", type=str, default="")

	parser.add_argument("--collection_path", type=str, default="", help="only for dense retrieval")
	parser.add_argument("--doc_embeddings_path", type=str, default="", help="only for dense retrieval")
	parser.add_argument("--docfaissidx_to_realdocid_path", type=str, default="", help="only for dense retrieval")
	
	parser.add_argument("--mode", type=str, required=True, choices=["cqr", "cdr"])
	parser.add_argument("--retriever_type", type=str, default="sparse", required=True, choices=["dense", "sparse"])
	parser.add_argument("--max_query_length", type=int, default=32)
	parser.add_argument("--max_response_length", type=int, default=64)
	parser.add_argument("--max_seq_length", type=int, default=256)
	
	args = parser.parse_args()
	# if args.retriever_type == "dense" and args.collection_path == "":
	# 	raise ValueError
	device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
	args.device = device
	return args


if __name__ == "__main__":
	args = get_args()
	search_bot = SearchBot(args)
	app.run(host='0.0.0.0',port=8082,debug=False)
