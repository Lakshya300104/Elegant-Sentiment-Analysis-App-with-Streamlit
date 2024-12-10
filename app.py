import streamlit as st
import sqlite3
import bcrypt
import pickle
from datetime import datetime
import pandas as pd
from nltk.stem.porter import PorterStemmer
import re,nltk
from nltk.corpus import stopwords

custom_css =  """
<style>
/* Import the new fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Cinzel:wght@400;700&display=swap');

/* Main app background - White Fog */
[data-testid="stAppViewContainer"] {
    background-color: #FAF5F1;
    color: #292F36; /* Carbon Gray for text */
    font-family: 'Poppins', sans-serif;
}

/* Sidebar background - Light Gray */
[data-testid="stSidebar"] {
    background-color: #E0DBD8;
    color: #292F36; /* Carbon Gray for text */
    font-family: 'Cinzel', serif;
    font-size: 18px;
    border-right: 2px solid #8F7A6E; /* Soft Brown border */
}

/* Header Styling - Fresh Red */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cinzel', serif;
    color: #A41F13; /* Fresh Red for headers */
    text-shadow: 1px 1px 2px #8F7A6E; /* Subtle shadow with Soft Brown */
}

/* Button Styling */
button {
    background-color: #A41F13; /* Fresh Red */
    color: #FAF5F1; /* White Fog for text */
    border-radius: 8px;
    padding: 10px 15px;
    font-size: 16px;
    border: none;
    font-family: 'Poppins', sans-serif;
    transition: transform 0.2s ease-in-out, background-color 0.3s;
}
button:hover {
    background-color: #8F7A6E; /* Soft Brown on hover */
    transform: scale(1.1); /* Slight zoom effect */
}

/* Text Input and Text Area Styling */
textarea, input {
    border: 2px solid #A41F13; /* Fresh Red border */
    border-radius: 8px;
    padding: 10px;
    font-family: 'Poppins', sans-serif;
}

/* Navigation Sidebar Title Styling */
[data-testid="stSidebar"] .css-1d391kg {
    color: #292F36; /* Carbon Gray for text */
    font-family: 'Cinzel', serif;
    font-size: 22px;
    font-weight: bold;
}

/* Metrics Box Styling */
[data-testid="stMetricValue"] {
    font-family: 'Cinzel', serif;
    font-size: 22px;
    color: #A41F13; /* Fresh Red */
}
[data-testid="stMetricLabel"] {
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
    color: #8F7A6E; /* Soft Brown */
}

/* Add spacing and padding for a balanced layout */
.stApp {
    padding: 20px;
}
</style>
"""

#Creating Database
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

conn=get_db_connection()
c=conn.cursor()


#Creating the tables
def create_table():
    with conn:
        c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')
        c.execute('''CREATE TABLE IF NOT EXISTS sentiment_analysis_results (
                             id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, text TEXT, sentiment TEXT, 
                             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

create_table()


# Starting session state
if 'username' not in st.session_state:
    st.session_state['username'] = 'Guest'


# Signup
def signup(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        with conn:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                         (username, hashed_password.decode('utf-8')))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False



#Login
def login(username, password):
    with conn:
        user = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[1].encode('utf-8')):
            st.session_state['username'] = username
            return True
    return False


#Displaying the Card
def display_card():
    st.sidebar.title(f"{st.session_state['username'].capitalize()}")


#Loading Models
def load_naive_bias():
    loaded_vectorizer = pickle.load(open('vectorizerNB.pkl', 'rb'))
    loaded_model = pickle.load(open('trained_modelNB.sav', 'rb'))
    return loaded_model,loaded_vectorizer

def load_logistic_regression():
    loaded_vectorizer = pickle.load(open('vectorizerLR.pkl', 'rb'))
    loaded_model = pickle.load(open('trained_modelLR.sav', 'rb'))
    return loaded_model,loaded_vectorizer

#Stemming Content
def stemming(content):
    nltk.download('stopwords')
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  # Remove all characters like @#$%
    stemmed_content = stemmed_content.lower()  # Convert to lowercase
    stemmed_content = stemmed_content.split()  # Split into words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]  # Stem words and remove stopwords
    stemmed_content = ' '.join(stemmed_content)  # Join the words back into a single string
    return stemmed_content

#Making Prediction
def perform_prediction(vectorizer,model,text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return 'Positive' if prediction[0] == 1 else 'Negative'


#Main Page
# Main app interface
def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Login", "Sign Up", "Continue as Guest", "Sentiment Analysis", "Metrics"])

    if page == "Login":
        st.markdown(custom_css, unsafe_allow_html=True)
        login_page()
    elif page == "Sign Up":
        signup_page()
        st.markdown(custom_css, unsafe_allow_html=True)
    elif page == "Continue as Guest":
        guest_page()
        st.markdown(custom_css, unsafe_allow_html=True)
    elif page == "Sentiment Analysis":
        display_card()
        sentiment_page()
        st.markdown(custom_css, unsafe_allow_html=True)
    elif page == "Metrics":
        metrics_page()
        st.markdown(custom_css, unsafe_allow_html=True)



def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.success(f"Welcome, {st.session_state['username']}!")
        else:
            st.error("Invalid username or password.")


def signup_page():
    st.title("Sign Up")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Sign Up"):
        if signup(username, password):
            st.success(f"Account creation successful. Welcome, {username}!")
            st.session_state['username'] = username
        else:
            st.error("Username already taken. Please choose another.")


def guest_page():
    st.title("Continue as Guest")
    st.session_state['username'] = "Guest"
    st.info("Guest Mode Activated.")

def sentiment_page():
    st.markdown("<h1 class='big-font'>Sentiment Analysing App</h1>", unsafe_allow_html=True)
    models_av=["Select One","Logistic Regression", "Naive Bias"]
    choice = st.selectbox("Which Model do you want to use?",models_av)
    if choice == "Naive Bias":
        st.markdown("<h3 class='big-font'>You are using Naive Bias</h1>", unsafe_allow_html=True)
        model,vectorizer = load_naive_bias()

        text = st.text_area("Enter a piece of text to analyze:")
        text= stemming(text)
        if st.button("Analyze Sentiment"):
            if text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    prediction = perform_prediction(vectorizer,model,text)
                st.success("Analysis complete!")
                st.markdown(f"<h2>Sentiment: {prediction}</h2>", unsafe_allow_html=True)

                store_sentiment_analysis_results(st.session_state['username'], text, prediction)
                display_previous_results()


    elif choice == "Logistic Regression":
        st.markdown("<h3 class='big-font'>You are using Logistic Regression</h1>", unsafe_allow_html=True)
        model, vectorizer = load_logistic_regression()
        text = st.text_area("Enter a piece of text to analyze:")
        text = stemming(text)
        if st.button("Analyze Sentiment"):
            if text.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    prediction = perform_prediction(vectorizer, model, text)
                st.success("Analysis complete!")
                st.markdown(f"<h2>Sentiment: {prediction}</h2>", unsafe_allow_html=True)

                store_sentiment_analysis_results(st.session_state['username'], text, prediction)
                display_previous_results()

#Storing into DB
def store_sentiment_analysis_results(username, text, sentiment):
    with conn:
        conn.execute('INSERT INTO sentiment_analysis_results (username, text, sentiment, timestamp) VALUES (?, ?, ?, ?)',
                     (username, text, sentiment, datetime.now()))
        conn.commit()

# Display previous results
def display_previous_results():
    previous_results = conn.execute('SELECT text, sentiment, timestamp FROM sentiment_analysis_results WHERE username=? ORDER BY timestamp DESC', (st.session_state['username'],)).fetchall()
    if previous_results:
        st.write("Previous Sentiment Analysis Results:")
        results_df = pd.DataFrame(previous_results, columns=['Text', 'Sentiment', 'Timestamp'])
        st.write(results_df)



def metrics_page():
    st.title("Metrics")

    previous_results = conn.execute('SELECT sentiment FROM sentiment_analysis_results WHERE username=?',
                                    (st.session_state['username'],)).fetchall()
    if previous_results:
        sentiment_counts = {'Positive': 0, 'Negative': 0}
        for result in previous_results:
            if result[0] == 'Positive':
                sentiment_counts['Positive'] += 1
            elif result[0] == 'Negative':
                sentiment_counts['Negative'] += 1

        st.subheader("Sentiment Counts")
        sentiment_counts_df = pd.DataFrame.from_dict(sentiment_counts, orient='index', columns=['Count'])
        st.bar_chart(sentiment_counts_df)

        st.subheader("Analysis Metrics")
        st.metric(label="Total Positive Sentiments", value=sentiment_counts['Positive'])
        st.metric(label="Total Negative Sentiments", value=sentiment_counts['Negative'])
    else:
        st.info("No sentiment analysis results to display for this user.")



if __name__ == "__main__":
    main_app()