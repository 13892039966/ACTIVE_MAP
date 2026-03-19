#!/usr/bin/env bash

set -euo pipefail

ENV_ROOT="${1:-/home/hmq/ws/SchoolGymDay/LinuxNoEditor}"
SCENE_MAP="${2:-/Game/SchoolGym/Maps/SchoolGymDay}"
shift $(( $# >= 2 ? 2 : $# ))

GAME_NAME="activegs"
GAME_BIN="${ENV_ROOT}/${GAME_NAME}/Binaries/Linux/${GAME_NAME}"

if [[ ! -x "${GAME_BIN}" ]]; then
  chmod +x "${GAME_BIN}"
fi

exec "${GAME_BIN}" "${GAME_NAME}" "${SCENE_MAP}" "$@"
