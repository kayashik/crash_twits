import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import nltk
import string
import re

pd.set_option('display.max_columns', 30)
nltk.download('stopwords')
nltk.download('wordnet')

#Load the dataset
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

'''
Обработаем для начала тестовые данные (поле text)
'''
#print(train_df.head(5))

'''
    ----- Remove punctuation -----
'''

def remove_punct(text):
    text_nonpunct = "".join([char for char in text if char not in string.punctuation])
    return text_nonpunct

train_df['text_clean'] = train_df['text'].apply(lambda x: remove_punct(x))

#print(train_df.head(5))

'''
    ----- Tokenization -----
'''


def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

train_df['text_tokenize'] = train_df['text_clean'].apply(lambda x: tokenize(x.lower()))

#print(train_df.head(5))

'''
    ----- Remove stopwords ------
'''
# Минус этого пункта здесь в том, что слова типа i've есть в списке, но после удаления пунктуации осталось только ive
# а этого в списке нет

stopwordsEn = nltk.corpus.stopwords.words('english')

def remove_stop_words(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwordsEn]
    return text

train_df['text_nostop'] = train_df['text_tokenize'].apply(lambda x: remove_stop_words(x))

#print(train_df[['text_nostop', 'text_tokenize']].head(5))

'''
    ----- Preprocessing Data: Stemming ------
'''
# Удаляем окочания слов типа ing, s, ed. Но не восстанавливаем слово по смыслу а оставляем точ то осталось как есть

ps = nltk.PorterStemmer()

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

train_df['text_stemmed'] = train_df['text_nostop'].apply(lambda x: stemming(x))

'''
    ----- Preprocessing Data: Lemmatizing ------
'''
# Получаем исходную форму слова, а не просто отрезаем окончания. Работает дольше чем тот что выше

wn = nltk.WordNetLemmatizer()

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

train_df['text_lemmatized'] = train_df['text_nostop'].apply(lambda x: lemmatizing(x))

#print(train_df[['text_nostop', 'text_stemmed','text_lemmatized']].head(5))

'''
    ----- Vectorizing Data: TF-IDF ------
'''
# Вычисляет "относительную частоту" появления слова в документе по сравнению с его частотой во всех документах.
# используется для оценки поисковой системы, суммирования текста, кластеризации документов.

# Это то как написано в примере и из-за analyzer=clean_text это нифига не работает:
# clean_text is not a valid tokenization scheme/analyzer
# clean_text это метод в котором мы данные обрабатываем?
# И еще не совсем понимаю зачем мы с text это делаем если до этого так старались данные почистить
tfidf_vect = feature_extraction.text.TfidfVectorizer()
Y_tfidf = tfidf_vect.fit_transform(train_df['text']) # Почему text?

print(Y_tfidf.shape)
print(tfidf_vect.get_feature_names())

# Мой вариант. Наверняка не самый лучший
def identity_tokenizer(text):
    return text

tfidf_vect1 = feature_extraction.text.TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
X_tfidf = tfidf_vect1.fit_transform(train_df['text_lemmatized']) # Почему text?

print(X_tfidf.shape)
print(tfidf_vect1.get_feature_names())

print("All done!")