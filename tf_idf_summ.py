from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# loading the dataset
dataset = load_dataset('ccdv/arxiv-summarization', 'document')

# getting the first article
article = dataset['train'].select([0])
for example in article['article']:
    print(f"article: \n {example}")
    print("\n \n")


def tokenize(text: str, stop_words: set[str]):
    # splitting the text into sentences
    sentences = sent_tokenize(text)

    tokenized_sentences = []
    sentence_indices = []

    for i, sentence in enumerate(sentences):
        tokenized_list = []
        word = []

        for element in sentence.lower():
            if element.isalpha() or element == '@':
                word.append(element)
                continue
            if word:
                tokenized_list.append(''.join(word))
                word = []
        if word:
            tokenized_list.append(''.join(word))
        # removing stopwords and lemmatizing tokens
        tokens_without_stopwords = remove_stop_words(tokenized_list, stop_words)
        lemmatized_tokens = lemmatize_tokens(tokens_without_stopwords)

        tokenized_sentences.append(lemmatized_tokens)
        sentence_indices.append(i)

    return tokenized_sentences, sentence_indices


def remove_stop_words(tokens, stop_words):
    return [token for token in tokens if token not in stop_words]


def tokenize_documents(documents: list[str], stop_words: set[str]):
    if not (documents and isinstance(documents, list) and
            all(isinstance(doc, str) for doc in documents)):
        return None

    tokenized_documents = []
    sentence_indices_list = []

    for document in documents:
        cleaned_document = document
        tokenized_doc, sentence_indices = tokenize(cleaned_document, stop_words)
        if not tokenized_doc:
            return None
        tokenized_documents.append(tokenized_doc)
        sentence_indices_list.append(sentence_indices)

    return tokenized_documents, sentence_indices_list


lemmatizer = WordNetLemmatizer()


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [lemmatizer.lemmatize(token) for token in tokens]


# tokenization and lemmatization
tokenized_example, sentence_indices_list = tokenize_documents(article['article'], stop_words)

# merging sentences into strings
joined_tokenized_example = [
    ' '.join(sentence)
    for doc in tokenized_example if isinstance(doc, list)
    for sentence in doc if isinstance(sentence, list) and all(isinstance(token, str) for token in sentence)
]
#print(joined_tokenized_example)

# merging sentence indices for all documents
all_sentence_indices = [index for indices in sentence_indices_list for index in indices]
#print(all_sentence_indices)

# calculating tf-idf
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(joined_tokenized_example)

# calculating sentence importance
sentence_scores = np.array(tfidf.sum(axis=1)).flatten()
important_sentences = np.argsort(sentence_scores)[::-1]
top_n = 3
summary_indices = important_sentences[:top_n]
#summary_indices.sort() # думала, что поможет повысить rougeL

# selecting sentences from the original text
#summary_sentences = [joined_tokenized_example[all_sentence_indices[i]] for i in summary_indices]
original_sentences = sent_tokenize(article['article'][0])
summary_sentences = [original_sentences[all_sentence_indices[i]] for i in summary_indices]

result = ' '.join(summary_sentences)

print(f"result: \n {result} \n")
print(f"length of original_sentences: {len(original_sentences)}")
print(f"length of joined_tokenized_example: {len(joined_tokenized_example)}")
print(f"summary_indices: {summary_indices}")
#print(f"all_sentence_indices: {all_sentence_indices}")

example = dataset['train'][0]
reference = example['abstract']
generated_summary = result

# enabling stemming by setting use_stemmer=True
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
scores = scorer.score(reference, generated_summary)
print("\n")
for key in scores:
    print(f'{key}: {scores[key]}')
