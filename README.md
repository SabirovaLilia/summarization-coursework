# summarization-coursework
This repository contains code for coursework on extractive text summarization.

TF-IDF and TextRank methods are used for extractive summarization of scientific articles. Articles are taken from the dataset “ccdv/arxiv-summarization” from Hugging Face. 
## Project Structure 
- `experiment_num_1_tfidf.py` – this code was created to evaluate the quality
of the generated summaries (by TF-IDF) texts using the ROUGE, BLEU, and BERTScore metrics
- `experiment_num_1_textrank_networkX.py` – – this code was created to evaluate the quality of the generated summaries (by TF-IDF) texts using the ROUGE, BLEU, and BERTScore metrics
- `experiment_keywords_8566_1952_TFIDF.py` – this code was created to count the number of keywords (labeled as keywords or key_words in the original annotation)
in the generated summaries (using TF-IDF)
- `experiment_keywords_8566_1952_TextRank.py` – this code was created to count the number of keywords (labeled as keywords or key_words in the original annotation)
in the generated summaries (using TexrRank)
- `manual_analysis_tfidf.py` – this code was created to manually analyze
 the results of the TF-IDF algorithm for text summarization. The mechanism of sentence selection from the original text of articles is examined manually.
- `manual_analysis_textrank.py` – – this code was created to manually analyze
 the results of the TF-IDF algorithm for text summarization. The mechanism of sentence selection from the original text of articles is examined manually.
