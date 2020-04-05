import analyzeData
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from urllib import parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
np.set_printoptions(threshold=sys.maxsize)

# Load the dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_df['keyword'] = train_df['keyword'].replace(np.nan, '', regex=True)

my_analyzer = analyzeData.ClearUpAnalyzer()

# Create new features from keywords
train_df['keyword_clear'] = train_df['keyword'].apply(lambda x: my_analyzer(parse.unquote(x), False))

keywords = []

def extractUniqueWords(tokenizedKeyWord, uniqueTokenizedKeyWord):
    for word in tokenizedKeyWord:
        if word not in uniqueTokenizedKeyWord and word != '':
            uniqueTokenizedKeyWord.append(word)

train_df['keyword_clear'].apply(lambda x: extractUniqueWords(x, keywords))

def checkKeywords(words, keyword):
    for word in words:
        if word == keyword:
            return 1
    return 0

keywordsIndicator = pd.DataFrame()
for keyword in keywords:
    keywordsIndicator['has_' + keyword] = train_df['keyword_clear'].apply(lambda x: checkKeywords(x, keyword))

# Vectorizing Data: TF-IDF
tf = TfidfVectorizer(analyzer=my_analyzer)
X_tfidf = tf.fit_transform(train_df['text'])

# Add keywords
newFeachuredVector = np.append(keywordsIndicator, X_tfidf.todense(), axis=1)
print(newFeachuredVector.shape)

# Create model
rf = RandomForestClassifier()
params = {
    'n_estimators': [10, 150, 300],
    'max_depth': [30, 60, 90]
}
gs = GridSearchCV(rf, params, cv=5, n_jobs=-1)
print('Just text: \n')
gs_fitX = gs.fit(X_tfidf, train_df['target'])
print('Best score: ', gs_fitX.best_score_, ' Error: ', 1-gs_fitX.best_score_)
print('Best max_depth: ', gs_fitX.best_estimator_.max_depth, ' Best n_estimators: ', gs_fitX.best_estimator_.n_estimators)
print('\n')

print('Text and keywords: \n')
gs_fitY = gs.fit(newFeachuredVector, train_df['target'])
print('Best score: ', gs_fitY.best_score_, ' Error: ', 1-gs_fitY.best_score_)
print('Best max_depth: ', gs_fitY.best_estimator_.max_depth, ' Best n_estimators: ', gs_fitY.best_estimator_.n_estimators)
print('\n')

#gs_fit.predict()
print("All done!")