#!/usr/bin/env bash
# Установка (или удаление) ежедневной cron-задачи обновления индекса (задание 6)
#
#   scripts/install_cron.sh                 установить на 06:00 каждый день
#   scripts/install_cron.sh "30 5 * * *"    установить на другое время
#   scripts/install_cron.sh --remove        удалить задачу
#   scripts/install_cron.sh --show          показать текущую запись
#
# Время задается в часовом поясе системы, где работает cron.
# Повторный запуск не создает дубликатов: запись находится по метке

set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARK="# quantumforge-update-index"

current() { crontab -l 2>/dev/null || true; }

case "${1:-}" in
  --remove)
    current | grep -v "$MARK" | crontab - || true
    echo "Задача удалена"
    exit 0
    ;;
  --show)
    current | grep "$MARK" || echo "Задача не установлена"
    exit 0
    ;;
esac

SCHEDULE="${1:-0 6 * * *}"
mkdir -p "$ROOT/logs"
LINE="$SCHEDULE cd $ROOT && $ROOT/scripts/run_update.sh >> $ROOT/logs/cron.log 2>&1 $MARK"

{ current | grep -v "$MARK" || true; echo "$LINE"; } | crontab -
echo "Установлено: $LINE"
