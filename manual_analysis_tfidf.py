# FINAL CODE
"""
 This code was created to manually analyze
 the results of the TF-IDF algorithm for text summarization.
 The mechanism of sentence selection
 from the original text of articles is examined manually.
"""
from datasets import load_dataset
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    sentences = sent_tokenize(text.lower())
    words = [
        [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words]
        for sentence in sentences]

    return sentences, words


dataset = load_dataset('ccdv/arxiv-summarization', 'document')
# number of articles to be processed
num_samples = 1000
print(num_samples)

for i in range(num_samples):
    article = dataset['train'][i]
    if not isinstance(article['article'], str) or not article['article'].strip():
        continue

    sentences, tokenized_example = preprocess_text(article['article'])

    joined_tokenized_example = [' '.join(sentence) for sentence in tokenized_example]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(joined_tokenized_example)

    sentence_scores = np.array(tfidf.sum(axis=1)).flatten()
    important_sentences = np.argsort(sentence_scores)[::-1]

    top_n = len(sent_tokenize(article['abstract']))
    print('ABSTRACT', article['abstract'])
    print('\n \n \n')
    summary_indices = sorted(important_sentences[:top_n], key=lambda x: x)
    summary_sentences = [sentences[i] for i in summary_indices]
    result = ' '.join(summary_sentences)

    reference = article['abstract']
    generated_summary = result
    print('GENERATED SUMMARY', generated_summary)
    print('\n \n \n')
