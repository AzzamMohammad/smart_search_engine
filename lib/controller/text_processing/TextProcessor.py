import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import re

class TextProcessor:
       
    def ProcessText(self,text):
        if not text:
            return ''
        if not isinstance(text, str):
           return ''
        text = self.remove_numbers_and_dates(text)
        text = self._remove_stop_words(text)
        text = self._lemmatizer_document(text)
        text = self._stem(text)
        return text

    def _remove_stop_words(self, document):
        if not document:
           return ''
        if not isinstance(document, str):
           return ''
        words = nltk.word_tokenize(document)
        stop_words = set(stopwords.words('english'))
        filtered_words = [
            word for word in words if word.lower() not in stop_words
        ]
        filtered_text = ' '.join(filtered_words)
        return filtered_text
    
    def _stem(self, document):
        if not document:
           return ''
        if not isinstance(document, str):
           return ''
        ps = PorterStemmer()
        words = nltk.word_tokenize(document)
        stemmed_words = []
        for w in words:
            stemmed_words.append(ps.stem(w))
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text

    def _lemmatizer_document(self, document):
        if not document:
           return ''
        if not isinstance(document, str):
           return ''
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(document)
        lemmatized_words = []
        for w in words:
            lemmatized_words.append(lemmatizer.lemmatize(w))
        lemmatized_text = " ".join(lemmatized_words)
        return lemmatized_text
    
    
    def remove_numbers_and_dates(self,text):
        if not text:
           return ''
        if not isinstance(text, str):
           return ''
        number_pattern = r'\d+'
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        symbol_pattern = r'[^\w\s]'
        text = re.sub(number_pattern, '', text)
        text = re.sub(date_pattern, '', text)
        text = re.sub(symbol_pattern, '', text)
        
        return text