# FINAL CODE
"""
 This code was created to manually analyze
 the results of the TextRank algorithm for text summarization.
 The mechanism of sentence selection
 from the original text of articles is examined manually.
"""
from datasets import load_dataset
import evaluate
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

dataset = load_dataset('ccdv/arxiv-summarization', 'document')


def preprocess_text(text):
    sentences = sent_tokenize(text.lower())
    words = [
        [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words]
        for sentence in sentences]

    return sentences, words


def textrank_summarize(text, top_n=3):
    original_sentences, tokenized_sentences = preprocess_text(text)
    joined_tokenized_sentences = [' '.join(sentence) for sentence in tokenized_sentences]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(joined_tokenized_sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], i, sentence) for i, sentence in enumerate(original_sentences)), reverse=True)
    top_sentences = sorted(ranked_sentences[:top_n], key=lambda x: x[1])

    return " ".join([sentence for _, _, sentence in top_sentences])


# number of articles to be processed
num_samples = 1000
print(num_samples)

for i in range(num_samples):
    article = dataset['train'][i]
    text = article['article']
    reference = article['abstract']
    print('ABSTRACT', article['abstract'])
    print('\n \n \n')

    if not isinstance(text, str) or not text.strip():
        continue

    result = textrank_summarize(text, top_n=len(sent_tokenize(reference)))
    print('GENERATED SUMMARY', result)
    print('\n \n \n')
