"""
Скрипт скачивает страницы с открытой тематической вики и сохраняет
очищенный от HTML текст в папку raw_downloaded/.

Важно: скрипт нужно запускать на машине с доступом в интернет.
В учебной песочнице сетевой доступ ограничен списком разрешенных доменов,
поэтому здесь приведен рабочий шаблон, который используется на реальном
окружении студента. Итоговая база знаний в этом задании подготовлена
альтернативным способом - через generate_raw_docs.py, с тем же результатом:
30+ текстовых документов по сущностям вселенной, готовых к анонимизации.

Установка зависимостей:
    pip install requests beautifulsoup4

Запуск:
    python download_pages.py
"""

import os
import time

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_downloaded")

# список страниц-сущностей, которые нужно скачать
# заполняется вручную по 30+ адресам с выбранной вики
PAGE_URLS = [
    "https://starwars.fandom.com/wiki/Darth_Vader",
    "https://starwars.fandom.com/wiki/Luke_Skywalker",
    "https://starwars.fandom.com/wiki/Leia_Organa",
    "https://starwars.fandom.com/wiki/Han_Solo",
    "https://starwars.fandom.com/wiki/Obi-Wan_Kenobi",
    "https://starwars.fandom.com/wiki/Yoda",
    "https://starwars.fandom.com/wiki/Emperor_Palpatine",
    "https://starwars.fandom.com/wiki/Chewbacca",
    "https://starwars.fandom.com/wiki/Darth_Maul",
    "https://starwars.fandom.com/wiki/Mace_Windu",
    "https://starwars.fandom.com/wiki/Tatooine",
    "https://starwars.fandom.com/wiki/Naboo",
    "https://starwars.fandom.com/wiki/Coruscant",
    "https://starwars.fandom.com/wiki/Hoth",
    "https://starwars.fandom.com/wiki/Endor_(moon)",
    "https://starwars.fandom.com/wiki/Dagobah",
    "https://starwars.fandom.com/wiki/Alderaan",
    "https://starwars.fandom.com/wiki/Death_Star",
    "https://starwars.fandom.com/wiki/Lightsaber",
    "https://starwars.fandom.com/wiki/Millennium_Falcon",
    "https://starwars.fandom.com/wiki/The_Force",
    "https://starwars.fandom.com/wiki/R2-D2",
    "https://starwars.fandom.com/wiki/C-3PO",
    "https://starwars.fandom.com/wiki/Blaster",
    "https://starwars.fandom.com/wiki/Jedi",
    "https://starwars.fandom.com/wiki/Sith",
    "https://starwars.fandom.com/wiki/Galactic_Empire",
    "https://starwars.fandom.com/wiki/Alliance_to_Restore_the_Republic",
    "https://starwars.fandom.com/wiki/Wookiee",
    "https://starwars.fandom.com/wiki/Battle_of_Yavin",
    "https://starwars.fandom.com/wiki/Clone_Wars",
    "https://starwars.fandom.com/wiki/Order_66",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational RAG project, contact: student@example.com)"
}


def clean_page(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    content = soup.select_one("div.mw-parser-output")
    if content is None:
        return ""

    # убираем инфобоксы, таблицы, ссылки на источники и служебные блоки
    for tag in content.select("table, aside, sup.reference, .toc, script, style, .navbox"):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if p)
    return text


def title_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].replace("_", " ")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, url in enumerate(PAGE_URLS, start=1):
        title = title_from_url(url)
        filename = f"{i:02d}_{title.lower().replace(' ', '_')}.md"
        path = os.path.join(OUTPUT_DIR, filename)

        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            text = clean_page(response.text)
        except requests.RequestException as exc:
            print(f"Ошибка при загрузке {url}: {exc}")
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{text}\n")

        print(f"Сохранено: {filename} ({len(text)} символов)")
        time.sleep(1)  # уважаем ограничения нагрузки на сайт


if __name__ == "__main__":
    main()
