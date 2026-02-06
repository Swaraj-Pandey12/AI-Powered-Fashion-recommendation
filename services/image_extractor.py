import requests
from bs4 import BeautifulSoup

def extract_product_image(product_url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(product_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        img_tag = soup.find("img", class_="indimg")
        # print("image tag ", img_tag)
        if img_tag and img_tag.get("src"):
            return img_tag["src"]

    except Exception as e:
        print("Image extraction error:", e)

    return None



