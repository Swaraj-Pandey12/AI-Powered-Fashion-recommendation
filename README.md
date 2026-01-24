# AI-Powered Fashion Search Engine 👗🤖

## 📌 Overview
This project is an AI-based fashion search and recommendation system that allows users to search for outfits using natural language queries. The system understands user intent and provides relevant fashion product recommendations using machine learning and semantic search techniques.

---

## 🚀 Features
- Natural language–based outfit search
- Gender detection from user queries
- Semantic product matching using embeddings
- Fast similarity search using FAISS
- Image scraping for product visualization
- Responsive and user-friendly web interface

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Machine Learning:** Logistic Regression, Random Forest  
- **NLP:** SentenceTransformer (all-MiniLM-L6-v2)  
- **Vector Search:** FAISS  
- **Web Scraping:** BeautifulSoup  
- **Backend:** Flask  
- **Frontend:** HTML, CSS, Bootstrap  

---

## 🧠 How It Works
1. User enters a fashion-related query (e.g., *"casual men's summer outfit"*).
2. The system detects gender context using ML models.
3. Product descriptions are converted into embeddings.
4. FAISS retrieves the most relevant products using similarity search.
5. Results are displayed through a Flask-based web interface.

---

## 📊 Model Performance
- Logistic Regression: **82% accuracy**
- Random Forest: **92% accuracy**

---

## 🖥️ Setup Instructions
1. Clone the repository:

Install required dependencies:
pip install -r requirements.txt

Run the application:
python app.py

Open your browser and go to:
http://127.0.0.1:5000/

📈 Impact

Improved recommendation accuracy using gender-based filtering
Enabled fast and scalable product search
Enhanced user experience with intelligent fallback image handling

👤 Author

Swaraj Pandey
B.Tech Computer Science | AI/ML Enthusiast

   ```bash
   git clone https://github.com/your-username/repo-name.git
