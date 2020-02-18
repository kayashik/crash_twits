import nltk
import string
import re

class ClearUpAnalyzer:

    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')


    def __call__(self, text):
        nopunct_text = self.remove_punct(text)
        tokenized_text = self.tokenize(nopunct_text.lower())
        nostop_text = self.remove_stop_words(tokenized_text)
        #data = data.apply(lambda x: self.stemming(x))
        clean_text = self.lemmatizing(nostop_text)
        return clean_text


    #  ----- Remove punctuation -----
    def remove_punct(self, text):
        text_nonpunct = "".join([char for char in text if char not in string.punctuation])
        return text_nonpunct

    # ----- Tokenization -----
    def tokenize(self, text):
        tokens = re.split('\W+', text)
        return tokens

    # ----- Remove stopwords ------
    # Минус этого пункта здесь в том, что слова типа i've есть в списке, но после удаления пунктуации осталось только ive
    # а этого в списке нет
    def remove_stop_words(self, tokenized_list):
        stopwordsEn = nltk.corpus.stopwords.words('english')
        text = [word for word in tokenized_list if word not in stopwordsEn]
        return text


    # ----- Preprocessing Data: Stemming ------
    # Удаляем окочания слов типа ing, s, ed. Но не восстанавливаем слово по смыслу а оставляем точ то осталось как есть
    def stemming(self, tokenized_text):
        ps = nltk.PorterStemmer()
        text = [ps.stem(word) for word in tokenized_text]
        return text



    # ----- Preprocessing Data: Lemmatizing ------
    # Получаем исходную форму слова, а не просто отрезаем окончания. Работает дольше чем тот что выше
    def lemmatizing(self, tokenized_text):
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in tokenized_text]
        return text
