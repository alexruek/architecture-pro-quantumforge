#!/usr/bin/env bash
# Запуск update_index.py с повторными попытками при ошибке (задание 6)
#
# Код 0 (успех) и код 3 (обновление уже выполняется) повторов не требуют.
# Код 1 (ошибка) повторяется до UPDATE_ATTEMPTS раз с паузой
# UPDATE_RETRY_DELAY_SEC секунд. Все аргументы передаются в update_index.py
#
# Переменные окружения:
#   PYTHON                  интерпретатор (по умолчанию .venv/bin/python или python3)
#   UPDATE_ATTEMPTS         число попыток (по умолчанию 3)
#   UPDATE_RETRY_DELAY_SEC  пауза между попытками в секундах (по умолчанию 300)

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p logs

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

ATTEMPTS="${UPDATE_ATTEMPTS:-3}"
DELAY="${UPDATE_RETRY_DELAY_SEC:-300}"
attempt=1

while true; do
  "$PYTHON" src/update_index.py "$@"
  code=$?

  if [ "$code" -ne 1 ]; then
    exit "$code"
  fi

  stamp="$(date -u '+%Y-%m-%d %H:%M:%S')"
  if [ "$attempt" -ge "$ATTEMPTS" ]; then
    echo "$stamp UTC ERROR update failed after $attempt attempts" >> logs/update_index.log
    exit "$code"
  fi

  echo "$stamp UTC WARNING attempt $attempt of $ATTEMPTS failed, retry in ${DELAY}s" >> logs/update_index.log
  attempt=$((attempt + 1))
  sleep "$DELAY"
done
