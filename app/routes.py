from flask import Blueprint, request, render_template

def create_routes(recommender, gender_service):
    routes = Blueprint("routes", __name__)

    @routes.route("/", methods=["GET"])
    def home():
        return render_template("index.html", results=[], message=None, prompt="")

    @routes.route("/search", methods=["POST"])
    def search():
        raw_prompt = request.form.get("prompt")
        prompt_text = str(raw_prompt).strip() if raw_prompt else ""

        if not prompt_text:
            return render_template(
                "index.html",
                results=[],
                message="Please enter a search query!",
                prompt=""
            )

        # Predict gender using ML model
        predicted_gender = gender_service.predict(prompt_text)

        results = recommender.recommend(prompt_text, predicted_gender)

        if not results:
            return render_template(
                "index.html",
                results=[],
                message="No results found!",
                prompt=prompt_text
            )

        return render_template(
            "index.html",
            results=results,
            message=None,
            prompt=prompt_text
        )

    return routes
