"""update_index.py

Ежедневное автоматическое обновление векторного индекса (задание 6)

Источник данных - папка базы знаний (по умолчанию knowledge_base/).
Скрипт сравнивает ее содержимое с манифестом faiss_index/manifest.json
(путь файла -> sha256) и находит новые, измененные и удаленные документы.
Для них индекс обновляется точечно:

- удаленные и измененные файлы: их чанки удаляются из FAISS;
- новые и измененные файлы: разбиваются на чанки так же, как в
  build_index.py, получают эмбеддинги и добавляются в индекс;
- неизмененные файлы не трогаются и заново не векторизуются.

Полная пересборка выполняется автоматически, если манифеста нет, не
совпадают модель, размер чанков или backend эмбеддингов, индекс поврежден
или число векторов не сходится с манифестом. Принудительно - флагом --full.

Запуск:
    python src/update_index.py              обычное обновление
    python src/update_index.py --dry-run    только показать, что изменится
    python src/update_index.py --full       полная пересборка

Коды выхода: 0 - успех (или нечего обновлять), 1 - ошибка (в том числе
ошибка по отдельным файлам), 3 - другое обновление уже выполняется

Логи: logs/update_index.log (текст, с ротацией) и logs/update_runs.jsonl
(одна JSON-запись на запуск)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import injection_guard
from embeddings_factory import HashEmbeddings, get_embeddings

try:  # на Windows модуля fcntl нет, блокировка там отключается
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

KB_SUFFIXES = (".md", ".txt")
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".update.lock"
MANIFEST_VERSION = 1
LIST_LIMIT = 20  # сколько имен файлов выводить в лог по каждой категории

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOCKED = 3

LOGGER_NAME = "update_index"


class UpdateError(Exception):
    """Фатальная ошибка обновления: индекс при этом не изменяется"""


class AlreadyRunning(Exception):
    """Блокировка занята: другое обновление уже работает"""


@dataclass
class Settings:
    base_dir: Path
    source_dir: Path
    index_dir: Path
    log_dir: Path
    model_name: str
    backend: str
    chunk_size: int
    chunk_overlap: int
    force_full: bool = False
    dry_run: bool = False
    allow_empty: bool = False
    quarantine: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            base_dir=config.BASE_DIR,
            source_dir=config.KNOWLEDGE_BASE_DIR,
            index_dir=config.INDEX_PATH,
            log_dir=config.BASE_DIR / "logs",
            model_name=config.EMBEDDING_MODEL_NAME,
            backend=os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower(),
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            quarantine=os.getenv("QUARANTINE_INJECTIONS", "false").lower()
            in ("1", "true", "yes"),
        )

    def signature(self) -> dict:
        """Параметры, при смене которых старые векторы нельзя смешивать с новыми"""
        return {
            "model": self.model_name,
            "backend": self.backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


@dataclass
class Plan:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    def has_work(self) -> bool:
        return bool(self.added or self.changed or self.removed)


@dataclass
class RunRecord:
    started_at: str = ""
    finished_at: str = ""
    duration_sec: float = 0.0
    status: str = "ok"  # ok | partial | error | locked | dry-run
    mode: str = "incremental"  # incremental | full | noop
    reason: str = ""
    files_added: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    files_quarantined: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_total: int = 0
    index_size_bytes: int = 0
    embed_time_sec: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- утилиты


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s UTC %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime

    file_handler = RotatingFileHandler(
        log_dir / "update_index.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def source_key(path: Path, settings: Settings) -> str:
    """Ключ файла в манифесте и в метаданных чанков: путь от корня проекта

    Совпадает с metadata["source"], которое записал build_index.py
    (например, knowledge_base/01_ksarn_velgor.md)
    """
    try:
        return path.relative_to(settings.base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_source(settings: Settings, record: RunRecord, log: logging.Logger):
    """Возвращает (ключ -> путь, ключ -> sha256, множество нечитаемых ключей)"""
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    unreadable: set[str] = set()

    if not settings.source_dir.is_dir():
        return paths, hashes, unreadable

    for path in sorted(settings.source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in KB_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        key = source_key(path, settings)
        try:
            hashes[key] = sha256_of(path)
            paths[key] = path
        except OSError as exc:
            unreadable.add(key)
            msg = f"cannot read {key}: {exc}"
            record.errors.append(msg)
            log.error(msg)
    return paths, hashes, unreadable


def load_manifest(index_dir: Path) -> dict | None:
    path = index_dir / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != MANIFEST_VERSION or "files" not in data:
        return None
    return data


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_index_total(index_dir: Path) -> int | None:
    """Число векторов в index.faiss или None, если индекс отсутствует или поврежден"""
    faiss_path = index_dir / "index.faiss"
    pkl_path = index_dir / "index.pkl"
    if not faiss_path.exists() or not pkl_path.exists():
        return None
    try:
        return faiss.read_index(str(faiss_path)).ntotal
    except Exception:
        return None


def index_size_bytes(index_dir: Path) -> int:
    total = 0
    for name in ("index.faiss", "index.pkl"):
        path = index_dir / name
        if path.exists():
            total += path.stat().st_size
    return total


def save_index_atomic(store: FAISS, index_dir: Path) -> None:
    """Сохраняет индекс через временную папку и замену файлов

    Порядок важен: сначала index.pkl, потом index.faiss. Бот следит за
    временем изменения index.faiss и перечитывает индекс только после
    того, как оба файла уже заменены
    """
    tmp_dir = index_dir / ".tmp_save"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    store.save_local(str(tmp_dir))
    for name in ("index.pkl", "index.faiss"):
        os.replace(tmp_dir / name, index_dir / name)
    shutil.rmtree(tmp_dir, ignore_errors=True)


class IndexLock:
    """Не дает двум обновлениям работать одновременно (cron + ручной запуск)"""

    def __init__(self, index_dir: Path) -> None:
        self.path = index_dir / LOCK_NAME
        self._file = None

    def __enter__(self) -> "IndexLock":
        self._file = open(self.path, "w")
        if fcntl is not None:
            try:
                fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._file.close()
                self._file = None
                raise AlreadyRunning("обновление уже выполняется")
        return self

    def __exit__(self, *exc_info) -> None:
        if self._file is not None:
            if fcntl is not None:
                fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()


# ---------------------------------------------------------------- логика


def make_plan(hashes: dict[str, str], manifest_files: dict, unreadable: set[str]) -> Plan:
    plan = Plan()
    for key in sorted(hashes):
        entry = manifest_files.get(key)
        if entry is None:
            plan.added.append(key)
        elif entry.get("sha256") != hashes[key]:
            plan.changed.append(key)
        else:
            plan.unchanged.append(key)
    # нечитаемый файл не считается удаленным: его чанки остаются до следующего запуска
    plan.removed = sorted(k for k in manifest_files if k not in hashes and k not in unreadable)
    return plan


def decide_full_reason(settings: Settings, manifest: dict | None, index_total: int | None) -> str:
    """Возвращает причину полной пересборки или пустую строку, если хватит инкремента"""
    if settings.force_full:
        return "forced by --full"
    if manifest is None:
        return "manifest is missing or corrupted"
    if manifest.get("signature") != settings.signature():
        return "model, backend or chunking parameters changed"
    if index_total is None:
        return "index files are missing or corrupted"
    expected = sum(int(e.get("chunks", 0)) for e in manifest["files"].values())
    if index_total != expected:
        return f"vector count in the index ({index_total}) does not match the manifest ({expected})"
    return ""


def split_file(key: str, text: str, settings: Settings) -> list[Document]:
    """Разбивает один документ на чанки теми же параметрами, что build_index.py"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    stem = Path(key).stem
    document = Document(page_content=text, metadata={"source": key, "title": stem})
    chunks = splitter.split_documents([document])
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{stem}_{idx:03d}"
        chunk.metadata["chunk_index"] = idx
    return chunks


def prepare_chunks(
    keys: list[str],
    paths: dict[str, Path],
    hashes: dict[str, str],
    settings: Settings,
    record: RunRecord,
    log: logging.Logger,
):
    """Читает файлы и режет на чанки. Возвращает (чанки, записи манифеста, пропущенные ключи)"""
    chunks: list[Document] = []
    entries: dict[str, dict] = {}
    skipped: set[str] = set()

    for key in keys:
        try:
            text = paths[key].read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            msg = f"read error {key}: {exc}"
            record.errors.append(msg)
            log.error(msg)
            skipped.add(key)
            continue

        if not text.strip():
            msg = f"file {key} is empty, skipped"
            record.warnings.append(msg)
            log.warning(msg)
            skipped.add(key)
            continue

        triggers = injection_guard.find_triggers(text)
        if triggers:
            labels = ", ".join(triggers)
            if settings.quarantine:
                msg = f"file {key} quarantined, not indexed (rules: {labels})"
                record.files_quarantined += 1
                skipped.add(key)
            else:
                msg = f"file {key} looks like a prompt injection (rules: {labels}), indexed anyway"
            record.warnings.append(msg)
            log.warning(msg)
            if settings.quarantine:
                continue

        file_chunks = split_file(key, text, settings)
        chunks.extend(file_chunks)
        entries[key] = {
            "sha256": hashes[key],
            "size": paths[key].stat().st_size,
            "chunks": len(file_chunks),
            "indexed_at": iso(now_utc()),
        }
    return chunks, entries, skipped


def chunk_ids_for_sources(store: FAISS, sources: set[str]) -> list[str]:
    return [
        doc_id
        for doc_id, doc in store.docstore._dict.items()
        if doc.metadata.get("source") in sources
    ]


def run_update(settings: Settings, log: logging.Logger | None = None) -> RunRecord:
    """Выполняет одно обновление и возвращает запись о запуске (статус - в record.status)"""
    log = log or setup_logging(settings.log_dir)
    record = RunRecord(started_at=iso(now_utc()))
    started = time.time()

    log.info(
        "update started, source=%s, index=%s, model=%s, backend=%s%s",
        source_key(settings.source_dir, settings),
        source_key(settings.index_dir, settings),
        settings.model_name,
        settings.backend,
        ", dry-run" if settings.dry_run else "",
    )
    if settings.backend == "stub":
        log.warning("backend=stub: test embeddings, do not use for the production index")

    settings.index_dir.mkdir(parents=True, exist_ok=True)

    try:
        lock = IndexLock(settings.index_dir)
        with lock:
            _execute(settings, record, log)
    except AlreadyRunning:
        record.status = "locked"
        log.warning("another update is already running, run skipped")
    except UpdateError as exc:
        record.status = "error"
        record.errors.append(str(exc))
        log.error("update aborted: %s", exc)
    except Exception as exc:  # noqa: BLE001 - любая неожиданная ошибка не должна ломать лог
        record.status = "error"
        record.errors.append(f"{type(exc).__name__}: {exc}")
        log.exception("unexpected error, index left unchanged")

    finished = now_utc()
    record.finished_at = iso(finished)
    record.duration_sec = round(time.time() - started, 2)

    if record.status in ("ok", "partial") and record.errors:
        record.status = "partial"

    _write_summary(settings, record, log)
    return record


def _execute(settings: Settings, record: RunRecord, log: logging.Logger) -> None:
    paths, hashes, unreadable = scan_source(settings, record, log)
    manifest = load_manifest(settings.index_dir)
    index_total = read_index_total(settings.index_dir)

    reason = decide_full_reason(settings, manifest, index_total)
    full = bool(reason)
    record.mode = "full" if full else "incremental"
    record.reason = reason

    if full:
        log.info("mode: full rebuild (%s)", reason)
        plan = Plan(added=sorted(hashes))
        old_files: dict = {}
    else:
        old_files = manifest["files"]
        plan = make_plan(hashes, old_files, unreadable)

    if not hashes:
        if full or not settings.allow_empty:
            raise UpdateError(
                f"source {settings.source_dir} has no documents (folder missing, not mounted "
                "or empty). Use --allow-empty to clear the index deliberately"
            )

    record.files_added = len(plan.added)
    record.files_changed = len(plan.changed)
    record.files_removed = len(plan.removed)
    record.files_unchanged = len(plan.unchanged)

    log.info(
        "files found: %d (added %d, changed %d, removed %d, unchanged %d)",
        len(hashes),
        len(plan.added),
        len(plan.changed),
        len(plan.removed),
        len(plan.unchanged),
    )
    for label, keys in (("added", plan.added), ("changed", plan.changed), ("removed", plan.removed)):
        for key in keys[:LIST_LIMIT]:
            log.info("  %s: %s", label, key)
        if len(keys) > LIST_LIMIT:
            log.info("  %s: ... and %d more", label, len(keys) - LIST_LIMIT)

    if settings.dry_run:
        record.status = "dry-run"
        record.chunks_total = index_total or 0
        record.index_size_bytes = index_size_bytes(settings.index_dir)
        return

    if not full and not plan.has_work():
        record.mode = "noop"
        record.chunks_total = index_total or 0
        record.index_size_bytes = index_size_bytes(settings.index_dir)
        return

    to_index = plan.added + plan.changed
    chunks, entries, skipped = prepare_chunks(to_index, paths, hashes, settings, record, log)

    # счетчики отражают реально обработанные файлы, а не только запланированные
    record.files_added = sum(1 for k in plan.added if k in entries)
    record.files_changed = sum(1 for k in plan.changed if k in entries)

    # из-за ошибки чтения измененный файл остается в индексе в старом виде
    changed_ok = [k for k in plan.changed if k not in skipped]
    changed_skipped = [k for k in plan.changed if k in skipped]
    sources_to_drop = set(plan.removed) | set(changed_ok)

    if full and not chunks:
        raise UpdateError("no chunks produced after reading files, index left unchanged")

    # модель нужна только если есть что векторизовать; для чистого удаления
    # хватает заглушки, потому что при загрузке индекса эмбеддинги не вызываются
    embeddings = (
        get_embeddings(settings.backend, settings.model_name) if chunks else HashEmbeddings()
    )

    embed_start = time.time()
    if full:
        store = FAISS.from_documents(chunks, embeddings)
        record.chunks_added = len(chunks)
    else:
        store = FAISS.load_local(
            str(settings.index_dir), embeddings, allow_dangerous_deserialization=True
        )
        drop_ids = chunk_ids_for_sources(store, sources_to_drop)
        if drop_ids:
            store.delete(drop_ids)
        record.chunks_removed = len(drop_ids)
        if chunks:
            store.add_documents(chunks)
        record.chunks_added = len(chunks)
    record.embed_time_sec = round(time.time() - embed_start, 2)

    save_index_atomic(store, settings.index_dir)

    files = {} if full else {k: v for k, v in old_files.items() if k not in sources_to_drop}
    files.update(entries)
    for key in changed_skipped:
        files[key] = old_files[key]
    # неудачно прочитанный файл с прежними чанками остается в манифесте с прежним хешем
    manifest_out = {
        "version": MANIFEST_VERSION,
        "updated_at": iso(now_utc()),
        "signature": settings.signature(),
        "files": dict(sorted(files.items())),
    }
    write_atomic(
        settings.index_dir / MANIFEST_NAME,
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n",
    )

    record.chunks_total = store.index.ntotal
    record.index_size_bytes = index_size_bytes(settings.index_dir)


def _write_summary(settings: Settings, record: RunRecord, log: logging.Logger) -> None:
    if record.status == "locked":
        return

    finished = record.finished_at.replace("T", " ").replace("+00:00", " UTC")
    errors = len(record.errors)

    if record.status == "dry-run":
        log.info(
            "dry-run finished at %s, index not changed: %d files to add, %d to change, %d to remove",
            finished,
            record.files_added,
            record.files_changed,
            record.files_removed,
        )
    elif record.status == "error":
        log.error("update failed at %s, %d errors, index not changed", finished, errors)
    elif record.mode == "noop":
        log.info(
            "index checked at %s, no changes, %d chunks, index size %s, %d errors",
            finished,
            record.chunks_total,
            human_size(record.index_size_bytes),
            errors,
        )
    else:
        log.info(
            "chunks: +%d, -%d, total %d; index size %s; embeddings %.2f s; duration %.2f s",
            record.chunks_added,
            record.chunks_removed,
            record.chunks_total,
            human_size(record.index_size_bytes),
            record.embed_time_sec,
            record.duration_sec,
        )
        log.info(
            "index updated at %s, %d files added, %d changed, %d removed, %d errors",
            finished,
            record.files_added,
            record.files_changed,
            record.files_removed,
            errors,
        )

    if settings.dry_run:
        return
    runs_path = settings.log_dir / "update_runs.jsonl"
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def exit_code_for(record: RunRecord) -> int:
    if record.status in ("ok", "dry-run"):
        return EXIT_OK
    if record.status == "locked":
        return EXIT_LOCKED
    return EXIT_ERROR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обновление векторного индекса базы знаний")
    parser.add_argument("--full", action="store_true", help="полная пересборка индекса")
    parser.add_argument("--dry-run", action="store_true", help="показать план, ничего не менять")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="разрешить обновление, если в источнике не осталось документов",
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="не индексировать файлы, похожие на промпт-инъекцию",
    )
    parser.add_argument("--source-dir", type=Path, help="папка с документами")
    parser.add_argument("--index-dir", type=Path, help="папка индекса FAISS")
    parser.add_argument("--log-dir", type=Path, help="папка логов")
    parser.add_argument("--backend", choices=["huggingface", "stub"], help="backend эмбеддингов")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings.from_env()
    settings.force_full = args.full
    settings.dry_run = args.dry_run
    settings.allow_empty = args.allow_empty
    settings.quarantine = settings.quarantine or args.quarantine
    if args.source_dir:
        settings.source_dir = args.source_dir.resolve()
    if args.index_dir:
        settings.index_dir = args.index_dir.resolve()
    if args.log_dir:
        settings.log_dir = args.log_dir.resolve()
    if args.backend:
        settings.backend = args.backend

    record = run_update(settings)
    return exit_code_for(record)


if __name__ == "__main__":
    sys.exit(main())
