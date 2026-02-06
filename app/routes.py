from flask import Blueprint, request, render_template
from services.image_extractor import extract_product_image

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
        prompt_lower=prompt_text.lower()
        # Predict gender using ML model
        if "women" in prompt_lower or "woman" in prompt_lower:
            predicted_gender = "Women"
            print("GENDER OVERRIDE TRIGGERED → Women")

        elif "men" in prompt_lower or "man" in prompt_lower:
            predicted_gender = "Men"
            print("GENDER OVERRIDE TRIGGERED → Men")
        else:
            predicted_gender = gender_service.predict(prompt_text)
            print("predcited_gender: ",predicted_gender)


        results = recommender.recommend(prompt_text, predicted_gender)

        if not results:
            return render_template(
                "index.html",
                results=[],
                message="No results found!",
                prompt=prompt_text
            )
            
        # 🔥 ADD THIS BLOCK (THIS IS THE KEY)
        for item in results:
            product_url = item.get("product_url")
            if product_url:
                item["image_url"] = extract_product_image(product_url)
                # print("IMAGE URL:", item["image_url"]) 
            else:
                item["image_url"] = None

        return render_template(
            "index.html",
            results=results,
            message=None,
            prompt=prompt_text
        )

    return routes
