# FINAL CODE
"""
This code was created to count the number of keywords
(labeled as keywords or key_words in the original annotation)
in the generated summaries (using TexrRank)
"""
from datasets import load_dataset
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    sentences = sent_tokenize(text.lower())
    words = [
        [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalnum() and word not in stop_words]
        for sentence in sentences]
    return sentences, words


pattern_colon = re.compile(r"(?i)\bkey(?:word|words)?\b.*?\*?:\s*\*?\s*([\w\s\-;,@xmath\d]+)")
pattern_phrases = re.compile(r"(?i)\bkey\b.*?\bphrases?\b\s*[.:]?\s*\*?\s*([\w\s\-;,@xmath\d]+?)(?:\s*[_\s]*pacs|$)")


def extract_keywords(text):
    match_colon = pattern_colon.search(text)
    match_phrases = pattern_phrases.search(text)

    if match_colon:
        keywords = match_colon.group(1).strip()
    elif match_phrases:
        keywords = match_phrases.group(1).strip()
    else:
        return []

    keywords = re.sub(r"^\s*\*\s*|\s*\*\s*$", "", keywords)
    keywords = \
    re.split(r"\s*(?:[_\s]*pacs(?:\s*nos\.?| numbers)?\s*[:;,]?)\s*", keywords, maxsplit=1, flags=re.IGNORECASE)[0]
    keywords = re.sub(r"[;_/-]", ",", keywords)
    return [kw.strip() for kw in re.split(r",\s*", keywords) if kw.strip()]


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


dataset = load_dataset('ccdv/arxiv-summarization', 'document')
filtered_articles = [i for i in dataset['train'] if
                     ('keyword' in i['abstract'].lower() or 'key word' in i['abstract'].lower()) and i[
                         'article'].strip()]
filtered_articles = filtered_articles[:]
print(f"Number of filtered articles: {len(filtered_articles)}")

empty_keywords_count = 0
counter = 0
valid_articles = []

total_keywords_list = []
found_in_abstract_list = []
found_in_summary_list = []
found_in_article_list = []

for article in filtered_articles:
    abstract = article['abstract']
    print('ABSTRACT', abstract)
    keywords = extract_keywords(abstract)
    print(keywords)

    if len(keywords) == 1:
        counter += 1
        continue

    if not keywords:
        empty_keywords_count += 1
        continue

    abstract_cleaned = pattern_colon.split(abstract, maxsplit=1)[0].strip()
    abstract_cleaned = pattern_phrases.split(abstract_cleaned, maxsplit=1)[0].strip()
    print('CLEANED ABSTRACT:', abstract_cleaned)

    generated_summary = textrank_summarize(article['article'], top_n=len(sent_tokenize(abstract)))
    print('GENERATED SUMMARY', generated_summary)

    def clean_text(text):
        sentences = sent_tokenize(text.lower())
        words = [word for sentence in sentences for word in word_tokenize(sentence) if word.isalnum()]
        return ' '.join(words)


    def count_keywords(text, keywords):
        cleaned_text = clean_text(text)
        return sum(1 for kw in keywords if kw.lower() in cleaned_text)


    total_keywords = len(keywords)
    found_in_abstract = count_keywords(abstract_cleaned, keywords)
    found_in_summary = count_keywords(generated_summary, keywords)
    found_in_article = count_keywords(article['article'], keywords)

    print(f"Total keywords: {total_keywords}")
    print(f"Found in abstract: {found_in_abstract}")
    print(f"Found in summary: {found_in_summary}")
    print(f"Found in article: {found_in_article}")
    print("\n" + "-" * 50 + "\n")

    valid_articles.append(article)
    total_keywords_list.append(total_keywords)
    found_in_abstract_list.append(found_in_abstract)
    found_in_summary_list.append(found_in_summary)
    found_in_article_list.append(found_in_article)

print(f"Total empty keyword lists: {empty_keywords_count}")
print(f"Total of lists with 1 keyword {counter}")
print(f"Total valid articles: {len(valid_articles)}")

if valid_articles:
    print(f"Average total keywords: {np.mean(total_keywords_list):.2f}")
    print(f"Average found in article: {np.mean(found_in_article_list):.2f}")
    print(f"Average found in abstract: {np.mean(found_in_abstract_list):.2f}")
    print(f"Average found in summary: {np.mean(found_in_summary_list):.2f}")

