
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

import torch
import argparse
from models import Rewriter, Retriever, Reranker 
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

@app.route("/")
def home():
	return render_template("home.html")


context = []
@app.route("/get")
def get_bot_response():
	user_query = request.args.get('msg')
	# response = search_bot.search(user_query, context)
	response = "123"
	context.append(user_query)
	context.append(response)
	return response


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--rewriter_path", type=str, default="")
	parser.add_argument("--retriever_path", type=str, default="")
	parser.add_argument("--reranker_path", type=str, default="")
	parser.add_argument("--index_path", type=str, default="")
	parser.add_argument("--mode", type=str, required=True, choices=["cqr", "cdr"])

	args = parser.parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device
	return args


if __name__ == "__main__":
	args = get_args()
	
	app.run()