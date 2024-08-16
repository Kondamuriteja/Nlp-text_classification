import sys
import nltk
import sklearn
import pandas
import numpy

print('Python: {}'.format(sys.version))

#Load the dataset

import pandas as pd
import numpy as np

df = pd.read_table('SMSSpamCollection',header = None, encoding='utf-8')

print(df.info())
print(df.head())

#check class distribution
classes= df[0]
print(classes.value_counts())

#preprocess the Data
#convert class lables to binary values 0 = ham , 1 = spam

from sklearn.preprocessing import LabelEncoder #label encoder converts labels into binary
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
print(Y[:10])

text_messages = df[1]
print(text_messages[:10])

# use regular expressions to replace email addresses, URLs, phone numbers, other numbers

# Replace email addresses with 'email'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')


# change words to lower case - Hello, HELLO, hello are all the same word
processed = processed.str.lower()
print(processed)




from nltk.corpus import stopwords

# remove stop words from text messages

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))

# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))

#Generating Features
#Feature engineering is the process of using domain knowledge of the data to create features for machine learning
#algorithms. In this project, the words in each text message will be our features. For this purpose, it will be necessary to tokenize each word.
#We will use the 1500 most common words as features.

#import nltk
#nltk.download('punkt')

from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

# print the total number of words and the 15 most common words
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(15)))

# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]

# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Lets see an example!
features = find_features(processed[0])
for key, value in features.items():
    if value == True:
        print (key)

# Now lets do it for all the messages
messages = zip(processed, Y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
messages = list(messages)  # Convert the zip object to a list
np.random.shuffle(messages)  # Now shuffle the list


# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]

# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)


print(len(training))
print(len(testing))

#Scikit-Learn Classifiers with NLTK
#Now that we have our dataset, we can start building algorithms! Let's start with a simple linear support vector classifier, then expand to other algorithms. 
#We'll need to import each algorithm we plan on using from sklearn.
#We also need to import some performance metrics, such as accuracy_score and classification_report.


# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))

from sklearn.ensemble import VotingClassifier
from nltk.classify import SklearnClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
]

# Convert zip object to a list of tuples
models = list(zip(names, classifiers))

# Create the VotingClassifier with the list of tuples
voting_clf = VotingClassifier(estimators=models, voting='hard', n_jobs=-1)

# Wrap the VotingClassifier with SklearnClassifier for NLTK compatibility
nltk_ensemble = SklearnClassifier(voting_clf)

# Train the ensemble model
nltk_ensemble.train(training)

# Calculate accuracy on the test set
accuracy = nltk.classify.accuracy(nltk_ensemble, testing) * 100
print("Voting Classifier: Accuracy: {:.2f}%".format(accuracy))


from sklearn.ensemble import VotingClassifier
from nltk.classify import SklearnClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Define the model names and classifiers
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
]

# Convert the zip object to a list of tuples
models = list(zip(names, classifiers))

# Create the VotingClassifier using the list of tuples
voting_clf = VotingClassifier(estimators=models, voting='hard', n_jobs=-1)

# Wrap the VotingClassifier with SklearnClassifier for NLTK compatibility
nltk_ensemble = SklearnClassifier(voting_clf)

# Train the ensemble model on the training set
nltk_ensemble.train(training)

# Make predictions on the testing set
txt_features, labels = zip(*testing)
predictions = nltk_ensemble.classify_many(txt_features)

# Print a classification report
print(classification_report(labels, predictions))

# Create and display a confusion matrix
conf_matrix = pd.DataFrame(
    confusion_matrix(labels, predictions),
    index=[['actual', 'actual'], ['ham', 'spam']],
    columns=[['predicted', 'predicted'], ['ham', 'spam']]
)

print(conf_matrix)

