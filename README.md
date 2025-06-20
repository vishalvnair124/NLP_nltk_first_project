# 📧 Spam Email Classifier – NLP + Streamlit

A simple **Streamlit web application** that classifies emails as **SPAM** or **NOT SPAM** using **Natural Language Processing (NLP)** and a pre-trained machine learning model. The app uses **NLTK** for preprocessing and a **TF-IDF + classifier** model loaded via `pickle`.

---

## 🚀 Features

- 🧠 Classifies email text using a trained ML model  
- 🧹 Text preprocessing with NLTK (tokenization, stopword removal, lemmatization)  
- 📊 TF-IDF vectorization  
- 💡 Interactive UI with Streamlit  
- 🎨 Custom styling for buttons and results

---

## 🧰 Technologies Used

| Component        | Tool/Library       |
|------------------|--------------------|
| Web Interface    | Streamlit          |
| Text Preprocessing | NLTK              |
| Machine Learning | Scikit-learn       |
| Model Storage    | Pickle (`model.pkl`, `vectorizer.pkl`)  
| Language         | Python  

---

## 📁 Project Structure

```
NLP_nltk_first_project/
├── app.py                  # Main Streamlit app
├── model.pkl               # Trained ML classification model
├── vectorizer.pkl          # TF-IDF vectorizer
├── *.ipynb                 # Jupyter notebooks for training & setup
└── README.md
```

---

## ▶️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/vishalvnair124/NLP_nltk_first_project.git
cd NLP_nltk_first_project
```

### 2. Install Dependencies
Make sure you have Python 3.8+ installed. Then run:
```bash
pip install streamlit scikit-learn nltk
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 📌 Notes

- The app uses **NLTK** to download required resources (`punkt`, `stopwords`, `wordnet`) at runtime.
- If you don’t see the output properly styled, make sure you are running the latest version of Streamlit.
- The model and vectorizer are expected to be present as `model.pkl` and `vectorizer.pkl` in the same directory.

---

## 📫 Dataset

If you want the original dataset used for training, feel free to contact:

- 🌐 [LinkedIn](https://www.linkedin.com/in/vishalvnair124)  
- 📸 [Instagram](https://instagram.com/vishalvnair124)

---

## 👨‍💻 Developed By

**Vishal V Nair**  
📫 [GitHub](https://github.com/vishalvnair124)

---

> 🧠 This is a beginner-friendly NLP project for learning text preprocessing, machine learning, and Streamlit UI development.
