# FINAL CODE
"""
This code was created to evaluate the quality
of the generated summaries (by TF-IDF) texts using
the ROUGE, BLEU, and BERTScore metrics
"""
from datasets import load_dataset
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import evaluate
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    sentences = sent_tokenize(text.lower())
    words = [
        [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words]
        for sentence in sentences]

    return sentences, words


rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore", model_type="distilbert-base-uncased")

dataset = load_dataset('ccdv/arxiv-summarization', 'document')
# number of articles to be processed
num_samples = 5000
print(num_samples)
rouge_scores_list = []
bleu_scores_list = []
bertscore_scores_list = []

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
    #print(article['abstract'])
    #print('\n \n \n')
    # sort indexes by their order in the text
    summary_indices = sorted(important_sentences[:top_n], key=lambda x: x)
    summary_sentences = [sentences[i] for i in summary_indices]
    result = ' '.join(summary_sentences)

    reference = article['abstract']
    generated_summary = result
    #print(generated_summary)
    # metrics calculation
    rouge_results = rouge.compute(predictions=[generated_summary], references=[reference])
    bleu_score = bleu.compute(predictions=[generated_summary], references=[[reference]])
    bertscore_results = bertscore.compute(predictions=[result], references=[reference], lang="en")

    rouge_scores_list.append(rouge_results)
    bleu_scores_list.append(bleu_score['bleu'])
    bertscore_scores_list.append(np.mean(bertscore_results['f1']))  # Среднее значение F1 BERTScore

# calculation of mean values of metrics
mean_rouge = {key: np.mean([score[key] for score in rouge_scores_list]) for key in rouge_scores_list[0]}
mean_bleu = np.mean(bleu_scores_list)
mean_bertscore = np.mean(bertscore_scores_list)

print("Mean TF-IDF ROUGE scores:", mean_rouge)
print("Mean TF-IDF BLEU score:", mean_bleu)
print("Mean TF-IDF BERTScore:", mean_bertscore)


