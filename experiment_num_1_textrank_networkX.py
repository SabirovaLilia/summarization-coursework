# FINAL CODE
"""
This code was created to evaluate the quality
of the generated summaries (by Textrank) texts using
the ROUGE, BLEU, and BERTScore metrics
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

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore", model_type="distilbert-base-uncased")

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
    # create TF-IDF vectors for the sentences
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(joined_tokenized_sentences)

    # calculate the cosine proximity between the sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # create graph
    graph = nx.from_numpy_array(similarity_matrix)

    # use pagerank
    scores = nx.pagerank(graph)

    # sort the sentences by their importance
    ranked_sentences = sorted(((scores[i], i, sentence) for i, sentence in enumerate(original_sentences)), reverse=True)
    top_sentences = sorted(ranked_sentences[:top_n], key=lambda x: x[1])

    return " ".join([sentence for _, _, sentence in top_sentences])


# number of articles to be processed
num_samples = 5000
print(num_samples)
rouge_scores_list = []
bleu_scores_list = []
bertscore_scores_list = []

for i in range(num_samples):
    article = dataset['train'][i]
    text = article['article']
    reference = article['abstract']

    if not isinstance(text, str) or not text.strip():
        continue

    result = textrank_summarize(text, top_n=len(sent_tokenize(reference)))

    # metrics calculation
    rouge_results = rouge.compute(predictions=[result], references=[reference])
    bleu_score = bleu.compute(predictions=[result], references=[[reference]])
    bertscore_results = bertscore.compute(predictions=[result], references=[reference], lang="en")

    rouge_scores_list.append(rouge_results)
    bleu_scores_list.append(bleu_score['bleu'])
    bertscore_scores_list.append(np.mean(bertscore_results['f1']))

# calculation of mean values of metrics
mean_rouge = {key: np.mean([score[key] for score in rouge_scores_list]) for key in rouge_scores_list[0]}
mean_bleu = np.mean(bleu_scores_list)
mean_bertscore = np.mean(bertscore_scores_list)

print("Mean TextRank ROUGE scores:", mean_rouge)
print("Mean TextRank BLEU score:", mean_bleu)
print("Mean TextRank BERTScore:", mean_bertscore)
