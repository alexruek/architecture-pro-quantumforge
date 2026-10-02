#!/bin/sh
# Точка входа контейнера updater: сохраняет окружение для cron,
# при RUN_ON_START=true делает одно обновление сразу, затем запускает cron
set -e

python - > /etc/cron.env <<'PY'
import os
import shlex

for key, value in os.environ.items():
    if key.isidentifier():
        print(f"export {key}={shlex.quote(value)}")
PY

crontab /app/deploy/crontab

if [ "${RUN_ON_START:-false}" = "true" ]; then
  /app/scripts/run_update.sh || true
fi

exec cron -f
