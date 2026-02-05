import os
from flask import Flask
from app.routes import create_routes
from bootstrap import bootstrap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

recommender, gender_service = bootstrap()
routes = create_routes(recommender, gender_service)
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)
