#!/bin/bash
set -e
host="$1"
shift
until curl -s "$host" >/dev/null; do
  echo "Waiting for MLflow at $host..."
  sleep 2
done
exec "$@"
