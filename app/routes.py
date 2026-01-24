from flask import Blueprint, request, render_template

def create_routes(recommender):
    routes = Blueprint("routes", __name__)

    @routes.route("/", methods=["GET"])
    def home():
        return render_template(
            "index.html",
            results=[],
            message=None,
            prompt=""
        )

    @routes.route("/search", methods=["POST"])
    def search():
        # Get raw input
        prompt_raw = request.form.get("prompt")

        # Convert safely to string (so .lower() won’t break)
        if prompt_raw is None:
            prompt_text = ""
        else:
            prompt_text = str(prompt_raw)

        # Remove whitespace
        prompt_text = prompt_text.strip()

        # If input is empty
        if prompt_text == "":
            return render_template(
                "index.html",
                results=[],
                message="Please enter a search query!",
                prompt=""
            )

        # SAFELY lowercase the string (prompt_text is always a string)
        lower_prompt = prompt_text.lower()

        # Gender detection
        gender = None
        if "men" in lower_prompt:
            gender = "Men"
        elif "women" in lower_prompt:
            gender = "Women"

        # Call recommender
        try:
            results = recommender.recommend(prompt_text, gender)
        except Exception as e:
            # In case recommender throws error
            print("Recommender error:", e)
            results = []

        # If no results found
        if not results:
            return render_template(
                "index.html",
                results=[],
                message="No results found for your query.",
                prompt=prompt_text
            )

        # Success: show results
        return render_template(
            "index.html",
            results=results,
            message=None,
            prompt=prompt_text
        )

    return routes
