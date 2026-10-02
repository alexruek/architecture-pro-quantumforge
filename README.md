# RAG-бот для QuantumForge Software

Проектная работа 7 спринта курса "Архитектор ПО". Бот отвечает на вопросы по базе знаний с помощью Retrieval-Augmented Generation: находит фрагменты документов в векторном индексе FAISS и формирует ответ через LLM, опираясь только на найденный текст. Если в базе ответа нет, бот честно сообщает об этом.

Ответы на вопросы заданий собраны в Project_template.md, подробности по каждому заданию - в docs/.

## Стек

| Компонент | Выбор |
| --- | --- |
| LLM | OpenAI gpt-4o-mini, без ключа - локальная google/flan-t5-base |
| Эмбеддинги | intfloat/multilingual-e5-small (Sentence-Transformers, 384 измерения) |
| Векторная база | FAISS (faiss-cpu) |
| Фреймворк | LangChain |
| Интерфейсы | консоль (REPL) и REST API на FastAPI |
| Упаковка | Dockerfile и docker-compose.yml (бот и обновление индекса по cron) |

## Где что лежит

| Задание | Результат | Описание |
| --- | --- | --- |
| 1. Исследование моделей и инфраструктуры | docs/task1_research_report.md, docs/cost_calculator.py | Project_template.md |
| 2. База знаний | knowledge_base/, terms_map.json, raw/, scripts/replace_terms.py | docs/README_zadanie2.md |
| 3. Векторный индекс | faiss_index/, src/build_index.py, src/query_index.py, index_report.json, docs/query_example.txt | docs/README_zadanie3.md |
| 4. RAG-бот, few-shot и Chain-of-Thought | src/rag_pipeline.py, src/prompts.py, src/cli.py, src/app.py, examples_dialogs.md | docs/README_zadanie4.md |
| 5. Демонстрация и защита от промпт-инъекций | src/injection_guard.py, src/demo_run.py, logs/demo_log.md, docs/screenshots/ | docs/README_zadanie5.md |
| 6. Ежедневное обновление индекса | src/update_index.py, scripts/run_update.sh, scripts/install_cron.sh, deploy/, docs/diagrams/ | docs/README_zadanie6.md |
| 7. Аналитика покрытия и качества | golden_questions.txt, src/evaluate.py, src/analyze_logs.py, logs/, docs/task7_*.md | docs/README_zadanie7.md |

## Быстрый запуск

Нужны Python 3.11 и доступ в интернет для первой загрузки модели эмбеддингов с Hugging Face (около 470 МБ).

```bash
git clone https://github.com/alexruek/architecture-pro-quantumforge.git
cd architecture-pro-quantumforge
git checkout rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # вписать OPENAI_API_KEY
python src/cli.py
```

Индекс лежит в репозитории, собирать его перед запуском не нужно. Пример вопроса по базе: "Кто такой Хоррак?". Пример вопроса вне базы: "Какая сейчас погода в Хельсинки?".

REST API:

```bash
uvicorn app:app --app-dir src
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "Что такое Директива 66?"}'
```

Docker:

```bash
docker compose up -d bot updater
curl http://127.0.0.1:8000/health
```

Сервис bot отдает REST API на порту 8000, сервис updater раз в сутки обновляет индекс по содержимому knowledge_base/.

## Проверка

```bash
python -m unittest discover -s tests -v     # автотесты, работают без сети и без LLM
python scripts/index_stats.py               # индекс согласован с базой знаний
python src/query_index.py                   # поиск по индексу без LLM
python src/demo_run.py                      # серия из 10 обращений задания 5
python src/evaluate.py                      # золотой набор задания 7
```

## Состояние базы знаний

Для задания 7 из базы намеренно убраны три документа (Ксарн Велгор, Войд-Ядро, Астерра): они перенесены в kb_removed/, индекс собран без них. Вернуть документы: python scripts/make_gaps.py restore, затем python src/update_index.py.
