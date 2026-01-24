from flask import Flask

def create_app(routes):
    app = Flask(__name__)
    app.register_blueprint(routes)
    return app
