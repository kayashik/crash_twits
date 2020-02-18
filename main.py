import analyzeData
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 30)

#Load the dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

'''
    ----- Vectorizing Data: TF-IDF ------
'''
# Мой вариант. Наверняка не самый лучший
my_analyzer = analyzeData.ClearUpAnalyzer()
tf = TfidfVectorizer(analyzer=my_analyzer)
X_tfidf = tf.fit_transform(train_df['text'])

print(X_tfidf.shape)

rf = RandomForestClassifier()
params = {
    'n_estimators': [10, 150, 300],
    'max_depth': [30, 60, 90]
}
gs = GridSearchCV(rf, params, cv=5, n_jobs=-1)
gs_fit = gs.fit(X_tfidf, train_df['target'])
print(pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False).head())

print("All done!")