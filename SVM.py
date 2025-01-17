
"""Importing the needed libraries"""

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle
import nltk
from datetime import datetime
nltk.download('stopwords')
date1 = datetime.now()
def date_diff_in_seconds(dt2, dt1):
  # Calculate the time difference between dt2 and dt1
  timedelta = dt2 - dt1
  # Return the total time difference in seconds
  return timedelta.days * 24 * 3600 + timedelta.seconds




"""**Data Processing**"""
print("Data Preprocessing Started")
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
print("Preprocessing Done")




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
print("Model Training Started")
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values
"""Splitting the dataset"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 42)
print("DataSet Split")
"""Converting the text to numeric because machine understand numerical data"""
print("Vectorisation Started")
vectorize = TfidfVectorizer(max_features=5000)
vectorize.fit(X_train)
SVM_vect_filename = 'vectorizerSVM.pkl'
SVM_vect_file = open(SVM_vect_filename,'wb')
pickle.dump(vectorize, SVM_vect_file)
X_train = vectorize.transform(X_train)
X_test = vectorize.transform(X_test)
"""Training the ML Model: Logistic Regression"""
print("vectorisation ended")
print("Model Training Started")
model = LinearSVC(C=1, max_iter=1000, class_weight='balanced')
model.fit(X_train, Y_train)
print("Model Training completed")
"""Model Evaluation: Accuracy Score"""


X_test_prediction = model.predict(X_test)  # Gives predictions for the test set
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)  # Comparison with ground truth
print('Accuracy Score on the test data (SVM):', test_data_accuracy) #76%

"""
Saving the model for further use
"""
filename = 'trained_modelSVM.sav'
file = open(filename, 'wb')
pickle.dump(model, file)

date2 = datetime.now()
# Print the time difference in seconds between the current date and the specified date
print("\n%d seconds" %(date_diff_in_seconds(date2, date1)))
# Print an empty line
print()
