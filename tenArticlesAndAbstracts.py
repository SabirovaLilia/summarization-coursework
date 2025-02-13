from datasets import load_dataset

# Загружаем датасет
dataset = load_dataset("ccdv/arxiv-summarization", "document")

# Берем первые 10 примеров
first_10_examples = dataset['train'][:10]

# Выводим статьи и их аннотации
for i, example in enumerate(first_10_examples['article']):
    print(f"Пример {i+1}:")
    print(f"Статья: {example}")
    print('-'*100)
    print(f"Аннотация: {first_10_examples['abstract'][i]}")
    print('_'*100)
