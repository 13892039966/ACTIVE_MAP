#!/usr/bin/env bash

set -euo pipefail

delay_sec="${1:-0}"
shift || true

sleep "${delay_sec}"
exec "$@"
