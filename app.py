from flask import Flask, render_template, request
import numpy as np
import pickle
import faiss
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import difflib

app = Flask(__name__)

# ---- Load Files ----
df = pickle.load(open("products.pkl", "rb"))
index = faiss.read_index("product_index.faiss")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vectorizer, classifier = pickle.load(open("classifier.pkl", "rb"))

# ---- Cache for faster image loading ----
image_cache = {}


def get_product_image(url):
    if url in image_cache:
        return image_cache[url]

    try:
        response = requests.get(url, timeout=6)
        soup = BeautifulSoup(response.text, "html.parser")

        div = soup.find("div", {"id": "0"})
        if div:
            img_tag = div.find("img")
            if img_tag and img_tag.get("src"):
                image_cache[url] = img_tag["src"]
                return img_tag["src"]

        img_tag = soup.find("img", {"class": "img-fluid"})
        if img_tag and img_tag.get("src"):
            image_cache[url] = img_tag["src"]
            return img_tag["src"]

    except:
        pass

    fallback = "https://via.placeholder.com/300"
    image_cache[url] = fallback
    return fallback




def detect_gender(prompt):
    text = prompt.lower()

    # Avoid substring conflict: check women first
    if any(word in text for word in ["women", "woman", "female", "girl", "lady"]):
        return "women"

    if any(word in text for word in ["men", "man", "male", "boy"]):
        return "men"

    # If user didn't specify, use classifier
    vectorized = vectorizer.transform([prompt])
    return classifier.predict(vectorized)[0].lower()

keywords = {
    "dress": ["dress", "gown", "one piece"],
    "shirt": ["shirt", "tshirt", "t-shirt", "tee"],
    "jeans": ["jeans", "denim"],
    "shorts": ["shorts", "half pant"],
    "jacket": ["jacket", "coat", "hoodie"],
}

colors = ["red","blue","black","white","green","yellow","pink","beige","brown","grey","purple"]


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    message = ""
    prompt = ""

    if request.method == "POST":
        prompt = request.form.get("prompt")

        predicted_gender = detect_gender(prompt)
        print("Predicted Gender:", predicted_gender)

        query_vec = embedding_model.encode([prompt])
        distances, indices = index.search(np.array(query_vec), 30)

        for idx in indices[0]:
            item = df.iloc[idx]

            if str(item["gender"]).lower() != predicted_gender:
                continue  # filter strictly

            results.append({
                "designer": item.get("designer", "Unknown"),
                "name": item.get("name", "No Name"),
                "price": item.get("mrp", "N/A"),
                "url": item.get("product_url", "#"),
                "image_url": get_product_image(item.get("product_url", ""))
            })

        if not results:
            message = f"❌ Nothing found for {predicted_gender}. Try another request."

    return render_template("index.html", results=results, prompt=prompt, message=message)


if __name__ == "__main__":
    app.run(debug=True)


# # ============================ OPTION 3 CODE (ONLY ADD FEATURE) ============================

# from flask import Flask, render_template, request
# import numpy as np
# import pickle
# import faiss
# import requests
# from bs4 import BeautifulSoup
# from sentence_transformers import SentenceTransformer

# app = Flask(__name__)

# # ---- Load required files only for this feature ----
# df = pickle.load(open("products.pkl", "rb"))
# index = faiss.read_index("product_index.faiss")
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# vectorizer, classifier = pickle.load(open("classifier.pkl", "rb"))

# image_cache = {}

# def get_product_image(url):
#     if url in image_cache:
#         return image_cache[url]

#     try:
#         response = requests.get(url, timeout=6)
#         soup = BeautifulSoup(response.text, "html.parser")

#         img = soup.find("img")
#         if img and img.get("src"):
#             image_cache[url] = img["src"]
#             return img["src"]
#     except:
#         pass

#     fallback = "https://via.placeholder.com/300"
#     image_cache[url] = fallback
#     return fallback


# def detect_gender(prompt):
#     text = prompt.lower()
#     if any(w in text for w in ["women","lady","female","girl","woman"]):
#         return "women"
#     if any(w in text for w in ["men","male","boy","man"]):
#         return "men"

#     return classifier.predict(vectorizer.transform([prompt]))[0].lower()


# # @app.route("/recommend", methods=["GET", "POST"])
# # def recommend():
# #     user_input = request.form["prompt"]

# #     gender = detect_gender(user_input)
# #     query_vec = embedding_model.encode([user_input])
# #     distances, ids = index.search(np.array(query_vec), 20)

# #     results = []
# #     for idx in ids[0]:
# #         item = df.iloc[idx]
# #         if item["gender"].lower() != gender:
# #             continue

# #         results.append({
# #             "name": item["name"],
# #             "price": item["mrp"],
# #             "url": item["product_url"],
# #             "image_url": get_product_image(item["product_url"])
# #         })

# #     return render_template("index.html", results=results, gender=gender)


# # if __name__ == "__main__":
# #     app.run(debug=True)



# @app.route("/recommend", methods=["GET", "POST"])
# def recommend():
#     results = []
#     gender = ""
#     user_input = ""

#     if request.method == "POST":
#         user_input = request.form.get("prompt", "")

#         if not user_input.strip():
#             return render_template("index.html", message="⚠️ Please enter something to search.")

#         gender = detect_gender(user_input)
#         query_vec = embedding_model.encode([user_input])
#         distances, ids = index.search(np.array(query_vec), 20)

#         for idx in ids[0]:
#             item = df.iloc[idx]
#             if item["gender"].lower() != gender:
#                 continue

#             results.append({
#                 "name": item["name"],
#                 "price": item["mrp"],
#                 "url": item["product_url"],
#                 "image_url": get_product_image(item["product_url"])
#             })

#         if not results:
#             return render_template("index.html", message=f"❌ No results found for {gender} category.")

#     # On GET request → just open page without error
#     return render_template("index.html", results=results, gender=gender, prompt=user_input)


# if __name__ == "__main__":
#     app.run(debug=True)
