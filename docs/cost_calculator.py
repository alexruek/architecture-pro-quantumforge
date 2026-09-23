"""
Оценка объема базы знаний QuantumForge Software в токенах
и примерной стоимости индексации при использовании облачных эмбеддингов.
Используется как вспомогательный расчет для задания 1
"""

# средние объемы контента по данным описания проекта
MARKDOWN_FILES = 18000
MARKDOWN_AVG_WORDS = 500

CONFLUENCE_PAGES = 3000
CONFLUENCE_AVG_WORDS = 800

PDF_FILES = 250
PDF_AVG_WORDS = 2000

MONTHLY_GROWTH_PAGES = 400
MONTHLY_GROWTH_AVG_WORDS = 600

WORDS_TO_TOKENS = 1.3  # среднее соотношение токенов к словам для смешанного русско-английского текста

# стоимость OpenAI text-embedding-3-small, доллар за 1 млн токенов
PRICE_PER_MILLION_TOKENS = 0.02


def words(files, avg_words):
    return files * avg_words


def tokens(word_count):
    return int(word_count * WORDS_TO_TOKENS)


def estimate():
    total_words = (
        words(MARKDOWN_FILES, MARKDOWN_AVG_WORDS)
        + words(CONFLUENCE_PAGES, CONFLUENCE_AVG_WORDS)
        + words(PDF_FILES, PDF_AVG_WORDS)
    )
    total_tokens = tokens(total_words)

    monthly_words = words(MONTHLY_GROWTH_PAGES, MONTHLY_GROWTH_AVG_WORDS)
    monthly_tokens = tokens(monthly_words)

    initial_cost = total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS
    monthly_cost = monthly_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS

    print(f"Всего слов в базе знаний: {total_words:,}")
    print(f"Всего токенов в базе знаний: {total_tokens:,}")
    print(f"Токенов прироста в месяц: {monthly_tokens:,}")
    print()
    print(f"Стоимость первичной индексации (OpenAI Embeddings): ${initial_cost:.2f}")
    print(f"Стоимость ежемесячного обновления (OpenAI Embeddings): ${monthly_cost:.4f}")
    print("Локальная модель эмбеддингов (Sentence-Transformers): $0 при наличии своего сервера")


if __name__ == "__main__":
    estimate()
