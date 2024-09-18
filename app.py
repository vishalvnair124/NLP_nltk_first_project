import streamlit as st
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the model and vectorizer (without caching)
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return vectorizer, model

vectorizer, model = load_model()

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Spam Email Classification</h1>", unsafe_allow_html=True)

st.write("### Enter the email content below:")

# Increase the text area size for better input
email_content = st.text_area("Email Content", "", height=250)

# Create a row with two columns for buttons
col1, col2 = st.columns([1, 1])  # Equal width columns

with col1:
    classify_button = st.button("Classify", key='classify_button')

with col2:
    clear_button = st.button("Clear", key='clear_button')

# Handle button actions
if classify_button:
    if email_content.strip() == "":
        st.warning("Please enter some email content to classify.")
    else:
        with st.spinner('Processing...'):
            # Preprocess the input email
            preprocessed_email = preprocess_text(email_content)

            # Vectorize the input email
            email_tfidf = vectorizer.transform([preprocessed_email])

            # Make a prediction
            prediction = model.predict(email_tfidf)
            result = "SPAM" if prediction[0] == 1 else "NOT SPAM"

            # Display the result with color (red for spam, green for not spam)
            if result == "SPAM":
                st.error(f"The email is classified as: {result}", icon="🚨")
            else:
                st.success(f"The email is classified as: {result}", icon="✅")

if clear_button:
    email_content = ""

# Additional styling to improve the look and feel
st.markdown("""
    <style>
    .css-1cpxqw2 { margin-top: 20px; }  /* Adds space between elements */
    .stTextArea label { font-size: 18px; } /* Increases the label font size */
    .stTextArea textarea { font-size: 16px; } /* Increases the text area font size */
    
    /* Styling for specific buttons using their keys */
    .css-1e7v2z3[class*="classify_button"] { 
        background-color: #4CAF50; /* Green background color */
        color: white; /* White text color */
        width: 150px; /* Width of the button */
        font-size: 16px; /* Font size of button text */
        padding: 10px; /* Padding inside the button */
        border-radius: 5px; /* Rounded corners */
    }
    .css-1e7v2z3[class*="classify_button"]:hover { 
        background-color: #45a049; /* Darker green on hover */
    }
    
    .css-1e7v2z3[class*="clear_button"] { 
        background-color: #f44336; /* Red background color */
        color: white; /* White text color */
        width: 150px; /* Width of the button */
        font-size: 16px; /* Font size of button text */
        padding: 10px; /* Padding inside the button */
        border-radius: 5px; /* Rounded corners */
    }
    .css-1e7v2z3[class*="clear_button"]:hover { 
        background-color: #d32f2f; /* Darker red on hover */
    }
    
    /* Flexbox styling to position buttons in the same row */
    .stButton { 
        display: inline-block; 
        margin: 5px; /* Adds margin around buttons */
    }
    /* Ensure buttons are positioned at opposite ends */
    .css-1n76uv3 { 
        display: flex; 
        justify-content: space-between; 
    }
    </style>
    """, unsafe_allow_html=True)
