
"""Importing the needed libraries"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

import nltk
nltk.download('stopwords')


"""**Data Processing**"""

# importing data
twitter_data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')

# naming columns because the columns are not correct. The first row is displayed as the column titles
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
twitter_data.columns = column_names
twitter_data = pd.read_csv('data.csv',names = column_names, encoding = 'ISO-8859-1')
twitter_data.head()
#No missing values

"""Converting the Label to 0 & 1

1 = Positive
0 = Negative

Converting the '4' to '1'.
"""

twitter_data.replace({'target':{4:1}}, inplace = True)

"""**Stemming:** Converting a word to it's root word

 Example = actor, acting, actress = act
"""

port_stem = PorterStemmer()
def stemming(content):

  stemmed_content = re.sub('[^a-zA-Z]',' ',content) #remove all characters like @#$
  stemmed_content = stemmed_content.lower() # convert everything into lowercase
  stemmed_content = stemmed_content.split() # split each and every word
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # performing stemming operation in the split words
  stemmed_content = ' '.join(stemmed_content) # join the result of stemmed list and making a single line

  return stemmed_content

print("Starting Stemming")
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming) # this takes a lot of time as dataset is large
print("Stemming Completed")


"""Separating stemmed data and target"""

X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

"""Splitting the dataset"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

"""Converting the text to numeric because machine understand numerical data"""

vectorize = TfidfVectorizer()
vectorize.fit(X_train)

X_train = vectorize.transform(X_train)
X_test = vectorize.transform(X_test)

vect_filename = 'vectorizer.pkl'
vect_file = open(vect_filename,'wb')
pickle.dump(vectorize, vect_file)




"""Training the ML Model: Logistic Regression"""

model = LogisticRegression(max_iter = 1000)

model.fit(X_train, Y_train)

"""Model Evaluation: Accuracy Score"""

X_test_prediction = model.predict(X_test) #gives a result of train set using the test model
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) # Comparison
print('Accuracy Score on the test data [Logistic Regression] :', test_data_accuracy)

"""**Model Accuracy = 77.6%**

Saving the model for further use
"""
filename = 'trained_model.sav'
file = open(filename,'wb')
pickle.dump(model, file)

