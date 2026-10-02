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
#   RUN_EVAL_AFTER_UPDATE   true - после успешного обновления прогнать золотой
#                           набор (src/evaluate.py, задание 7), по умолчанию false
#   EVAL_MIN_PASS           минимальная доля прошедших вопросов (по умолчанию 0.7).
#                           Если качество ниже, скрипт завершается с кодом 4,
#                           в logs/update_index.log пишется предупреждение

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
    break
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

# Задание 7: проверка качества обновленного индекса на золотом наборе
if [ "$code" -eq 0 ] && [ "${RUN_EVAL_AFTER_UPDATE:-false}" = "true" ]; then
  MIN_PASS="${EVAL_MIN_PASS:-0.7}"
  "$PYTHON" src/evaluate.py --min-pass "$MIN_PASS" --report logs/eval_last_report.md >> logs/eval_runs.log 2>&1
  eval_code=$?
  stamp="$(date -u '+%Y-%m-%d %H:%M:%S')"
  if [ "$eval_code" -eq 0 ]; then
    echo "$stamp UTC INFO quality check passed (min pass $MIN_PASS), report logs/eval_last_report.md" >> logs/update_index.log
  else
    echo "$stamp UTC WARNING quality check failed with code $eval_code (min pass $MIN_PASS), report logs/eval_last_report.md" >> logs/update_index.log
    exit 4
  fi
fi

exit "$code"
