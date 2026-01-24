from flask import Flask
from app.routes import create_routes
from bootstrap import bootstrap
import os

# Create app and tell Flask where templates are
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates")
)

# Load recommender
recommender = bootstrap()
routes = create_routes(recommender)
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)
