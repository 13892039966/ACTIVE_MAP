#!/usr/bin/env bash

set -euo pipefail

ENV_ROOT="${1:-/home/hmq/ws/SchoolGymDay/LinuxNoEditor}"
SCENE_MAP="${2:-/Game/SchoolGym/Maps/SchoolGymDay}"
shift $(( $# >= 2 ? 2 : $# ))

GAME_NAME="activegs"
GAME_BIN="${ENV_ROOT}/${GAME_NAME}/Binaries/Linux/${GAME_NAME}"

if [[ "${AIRSIM_WRITE_SETTINGS:-true}" == "true" || "${AIRSIM_WRITE_SETTINGS:-1}" == "1" ]]; then
  SETTINGS_PATH="${AIRSIM_SETTINGS_PATH:-${HOME}/Documents/AirSim/settings.json}"
  VEHICLE_NAME="${AIRSIM_VEHICLE_NAME:-drone_1}"
  SPAWN_X="${AIRSIM_VEHICLE_SPAWN_X:-0.0}"
  SPAWN_Y="${AIRSIM_VEHICLE_SPAWN_Y:-0.0}"
  SPAWN_Z="${AIRSIM_VEHICLE_SPAWN_Z:--1.0}"
  SPAWN_YAW_DEG="${AIRSIM_VEHICLE_SPAWN_YAW_DEG:-0.0}"

  mkdir -p "$(dirname "${SETTINGS_PATH}")"
  cat > "${SETTINGS_PATH}" <<EOF
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "${VEHICLE_NAME}": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "X": ${SPAWN_X},
      "Y": ${SPAWN_Y},
      "Z": ${SPAWN_Z},
      "Yaw": ${SPAWN_YAW_DEG},
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90,
              "MotionBlurAmount": 0
            },
            {
              "ImageType": 1,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90,
              "MotionBlurAmount": 0
            }
          ],
          "X": 0.35,
          "Y": 0.0,
          "Z": -0.2
        }
      },
      "Sensors": {
        "Imu": {
          "SensorType": 2,
          "Enabled": true
        }
      }
    }
  },
  "SubWindows": [
    {
      "WindowID": 0,
      "ImageType": 0,
      "CameraName": "front_center",
      "Visible": true
    },
    {
      "WindowID": 1,
      "ImageType": 1,
      "CameraName": "front_center",
      "Visible": true
    },
    {
      "WindowID": 2,
      "ImageType": 5,
      "CameraName": "front_center",
      "Visible": true
    }
  ]
}
EOF
  echo "[start_activegs_env] wrote AirSim settings to ${SETTINGS_PATH} with spawn=(${SPAWN_X}, ${SPAWN_Y}, ${SPAWN_Z}), yaw=${SPAWN_YAW_DEG}"
fi

if [[ ! -x "${GAME_BIN}" ]]; then
  chmod +x "${GAME_BIN}"
fi

if [[ -n "${SCENE_MAP}" ]]; then
  exec "${GAME_BIN}" "${GAME_NAME}" "${SCENE_MAP}" "$@"
else
  exec "${GAME_BIN}" "${GAME_NAME}" "$@"
fi
