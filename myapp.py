
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from bot import search_bot
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'
# prefix = 'sqlite:////'
# app.config['SQLALCHEMY_DATABASE_URI'] =  prefix + os.path.join(os.path.dirname(app.root_path), 'data.db')
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)
# login_manager = LoginManager(app)
# login_manager.login_view = 'login'
# login_manager.login_message = 'Welcome to our conversational search bot!'


# class User(db.Model, UserMixin):
#     user_id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(20))  # 用户名
#     password_hash = db.Column(db.String(128))  # 密码散列值
#     dialog_history = db.Column(db.LONGTEXT) 
#     def set_password(self, password):  # 用来设置密码的方法，接受密码作为参数
#         self.password_hash = generate_password_hash(password)  # 将生成的密码保持到对应字段
 
#     def validate_password(self, password):  # 用于验证密码的方法，接受密码作为参数
#         return check_password_hash(self.password_hash, password)  # 返回布尔值


# @login_manager.user_loader
# def load_user(user_id): 
#     user = User.query.get(int(user_id))
#     return user

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         if not username or not password:
#             flash('Invalid input.')
#             return redirect(url_for('login'))

#         user = User.query.first()
#         if username == user.username and user.validate_password(password):
#             login_user(user)
#             flash('Login success.')
#             return redirect(url_for('index'))

#         flash('Invalid username or password.')
#         return redirect(url_for('login'))

#     return render_template('login.html')


# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     flash('Goodbye.')
#     return redirect(url_for('index'))


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


if __name__ == "__main__":
	app.run()