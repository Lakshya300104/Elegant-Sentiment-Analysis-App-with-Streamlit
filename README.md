# **Elegant Sentiment Analysis App with Streamlit**
![1](https://github.com/user-attachments/assets/b49701e6-fa9d-4d59-9e88-3516f294eabb)


## **Overview**

This **Sentiment Analysis Web App** is a user-friendly platform designed to analyze the emotions behind textual data. Built using **Streamlit**, it leverages **Logistic Regression** and **Naive Bayes** machine learning models for sentiment classification. Users can select their preferred model and gain insights into why that model might be the best fit for their needs.

With **pre-trained models on 1.6 million tweets**, the app ensures fast predictions and a seamless experience. By saving the trained models and vectorizers, the app avoids retraining on each use, significantly boosting performance.

---

## **Key Features**

### 🎯 **Interactive Model Selection**
- Users can choose between **Logistic Regression** and **Naive Bayes**.
- Each model comes with an explanation of its benefits, helping users understand the trade-offs.

### ⚡ **Fast Predictions**
- Models are pre-trained on a dataset of **1.6 million tweets**.
- Both the vectorizers and trained models are saved and loaded dynamically to eliminate training delays.

### 📊 **Metrics Dashboard**
- Visualize sentiment analysis results with clear and engaging metrics.
- Track positive and negative sentiment trends over time.

### 💻 **Flexible User Roles**
- **Log in** as a registered user to save results.
- **Sign up** for a new account or explore the app as a **guest**.

### 🌐 **User-Friendly Interface**
- Built with **Streamlit**, offering a smooth and intuitive user experience.
- Designed with modern, visually appealing aesthetics.

---

## **Applications**

This project has a wide range of applications across various domains:

1. **📢 Brand Sentiment Analysis**:
   - Monitor public opinion about products or services on social media platforms.
2. **💬 Social Media Insights**:
   - Analyze tweets, comments, or posts to identify sentiment trends and topics.
3. **🧑‍🤝‍🧑 Customer Support**:
   - Identify negative feedback to address issues proactively.
4. **🎯 Mental Health Tools**:
   - Detect potential signs of emotional distress in user-generated content.
5. **🕵️ Investigative Journalism**:
   - Uncover sarcasm, bias, or sentiment in textual narratives.
6. **🤔 Content Moderation**:
   - Automate the classification of user comments to maintain healthy online communities.

---

## **How It Works**

### 1. **Model Training**
- **Dataset**: The models are trained on a dataset of **1.6 million tweets**, covering a variety of sentiments.
- **Logistic Regression**: Suited for datasets with feature relationships and high interpretability.
- **Naive Bayes**: Ideal for high-dimensional, sparse data like text.

### 2. **Saved Models**
- The trained models and vectorizers are serialized and saved using **pickle**. This eliminates the need for retraining on every app restart.

### 3. **Real-Time Predictions**
- Users enter text into the app, and the selected model predicts the sentiment as **Positive** or **Negative**.
- Predictions are nearly instantaneous, thanks to pre-trained models.

### 4. **Metrics Tracking**
- Sentiment trends are stored and visualized in the metrics dashboard, helping users track their analysis.

---

## **Getting Started**

### **Prerequisites**
Make sure you have the following installed:
- Python 3.7 or later
- Streamlit
- Other dependencies (see `requirements.txt`)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/Lakshya300104/Elegant-Sentiment-Analysis-App-with-Streamlit.git
   cd Elegant-Sentiment-Analysis-App-with-Streamlit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the app in your browser at:
   ```
   http://localhost:8501
   ```

---

## **Usage**

### **1. Login Page**
- Log in as an existing user, sign up for an account, or continue as a guest.
- Enjoy fun, engaging messages that make navigation intuitive.

### **2. Sentiment Analysis**
- Select a model (**Logistic Regression** or **Naive Bayes**) for predictions.
- Enter your text, analyze sentiments, and receive instant results.
- Learn about the benefits of the chosen model before analysis.

### **3. Metrics Dashboard**
- View sentiment trends with personalized insights.
- Compare positive and negative sentiment counts in a visual bar chart.

---

## **Demo**

📹 **Watch the App in Action**: [Demo Video Link](https://drive.google.com/file/d/1MnbkVsWIXfQG77cDX3ZPwQSKyXvjDSTy/view?usp=sharing)

---

## **Screenshots**

1. **Login Page**  
   ![image](https://github.com/user-attachments/assets/93dfd31f-d9bf-48f7-ac86-1777fcf8b6d3)


2. **Sentiment Analysis**  
   ![image](https://github.com/user-attachments/assets/9f618103-010f-4f26-b76b-930f16035006)


3. **Metrics Dashboard**  
   ![image](https://github.com/user-attachments/assets/792af473-57df-4131-ae6d-28ddb7086402)
   ![image](https://github.com/user-attachments/assets/8617fdc3-aafa-4fd9-b204-ef7d8e68999f)



---

## **Technologies Used**

| **Tool/Technology** | **Purpose**                           |
|----------------------|---------------------------------------|
| **Streamlit**        | Web app development                  |
| **Python**           | Backend logic and data processing    |
| **Logistic Regression** | Sentiment classification             |
| **Naive Bayes**      | Sentiment classification             |
| **SQLite**           | User authentication and data storage |
| **Pandas**           | Data manipulation                   |

---

## **Project Structure**

```
Elegant-Sentiment-Analysis-App-with-Streamlit/
│
├── app.py                     # Main Streamlit app
├── requirements.txt           # Python dependencies
├── vectorizerNB.pkl           # Saved vectorizer for Naive Bayes
├── trained_modelNB.sav        # Pre-trained Naive Bayes model
├── vectorizerLR.pkl           # Saved vectorizer for Logistic Regression
├── trained_modelLR.sav        # Pre-trained Logistic Regression model
├── README.md                  # Project documentation
└── images/                    # Screenshots and assets
```

---

## **Future Enhancements**
- Add more advanced models like **BERT** or **RoBERTa** for improved sentiment classification.
- Expand the app to support multi-class sentiment analysis (e.g., Neutral, Positive, Negative).
- Integrate a live feed for analyzing real-time social media data.

---

## **Contributing**

Contributions are welcome! If you’d like to improve this project, feel free to:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## **Contact**

Feel free to reach out with any questions or suggestions:
- **Email**: lakshya13004@gmail.com
- **LinkedIn**: [Connect]([https://www.linkedin.com/in/your-profile](https://www.linkedin.com/in/lakshya-arora-76a567259/))

---
