#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from datetime import datetime


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(path):
    if not os.path.isdir(path):
        return []
    names = []
    for name in sorted(os.listdir(path)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            names.append(name)
    return names


def extract_frame_index(name):
    match = re.search(r"frame_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def read_sparse_images(images_txt_path):
    registered = []
    if not os.path.isfile(images_txt_path):
        return registered

    with open(images_txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            image_name = parts[9]
            registered.append(image_name)
    return registered


def read_next_c_string(file_obj):
    chunks = []
    while True:
        byte = file_obj.read(1)
        if not byte or byte == b"\x00":
            break
        chunks.append(byte)
    return b"".join(chunks).decode("utf-8", errors="replace")


def read_sparse_images_bin(images_bin_path):
    registered = []
    if not os.path.isfile(images_bin_path):
        return registered

    with open(images_bin_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            return registered
        num_images = struct.unpack("<Q", header)[0]
        for _ in range(num_images):
            image_props = f.read(64)
            if len(image_props) != 64:
                break
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack(
                "<idddddddi", image_props
            )
            _ = (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id)
            image_name = read_next_c_string(f)
            registered.append(image_name)

            points2d_count_raw = f.read(8)
            if len(points2d_count_raw) != 8:
                break
            points2d_count = struct.unpack("<Q", points2d_count_raw)[0]
            skip_bytes = points2d_count * 24
            f.seek(skip_bytes, os.SEEK_CUR)

    return registered


def read_sparse_points_count(points3d_txt_path):
    if not os.path.isfile(points3d_txt_path):
        return 0

    with open(points3d_txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("# Number of points:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return 0
    return 0


def read_sparse_points_count_bin(points3d_bin_path):
    if not os.path.isfile(points3d_bin_path):
        return 0

    with open(points3d_bin_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            return 0
        return int(struct.unpack("<Q", header)[0])


def query_scalar(cursor, sql, default=0):
    try:
        row = cursor.execute(sql).fetchone()
    except sqlite3.Error:
        return default
    if row is None or row[0] is None:
        return default
    return row[0]


def inspect_colmap_database(db_path):
    stats = {
        "database_exists": os.path.isfile(db_path),
        "images_in_db": 0,
        "cameras_in_db": 0,
        "keypoint_rows": 0,
        "descriptor_rows": 0,
        "match_pairs": 0,
        "two_view_geometries": 0,
        "keypoints_total": 0,
    }
    if not stats["database_exists"]:
        return stats

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        stats["images_in_db"] = int(query_scalar(cursor, "SELECT COUNT(*) FROM images", 0))
        stats["cameras_in_db"] = int(query_scalar(cursor, "SELECT COUNT(*) FROM cameras", 0))
        stats["keypoint_rows"] = int(query_scalar(cursor, "SELECT COUNT(*) FROM keypoints", 0))
        stats["descriptor_rows"] = int(query_scalar(cursor, "SELECT COUNT(*) FROM descriptors", 0))
        stats["match_pairs"] = int(query_scalar(cursor, "SELECT COUNT(*) FROM matches", 0))
        stats["two_view_geometries"] = int(query_scalar(cursor, "SELECT COUNT(*) FROM two_view_geometries", 0))
        stats["keypoints_total"] = int(
            query_scalar(cursor, "SELECT COALESCE(SUM(rows), 0) FROM keypoints", 0)
        )
    finally:
        conn.close()
    return stats


def snapshot_if_exists(src_path, dst_path):
    if not os.path.exists(src_path):
        return False
    if os.path.isdir(src_path):
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
    else:
        ensure_dir(os.path.dirname(dst_path))
        shutil.copy2(src_path, dst_path)
    return True


def detect_stage(line):
    lowered = line.lower()
    if "feature extractor" in lowered:
        return "feature_extractor"
    if "exhaustive_matcher" in lowered or "matching block" in lowered:
        return "feature_matching"
    if "mapper" in lowered or "registering image" in lowered or "initialized with image pair" in lowered:
        return "mapper"
    if "image_undistorter" in lowered or "undistort" in lowered:
        return "image_undistorter"
    if "mogrify" in lowered or "copying and resizing" in lowered:
        return "resize"
    return None


def build_summary(source_path, report_dir, exit_code, duration_sec, log_stats, db_stats):
    input_dir = os.path.join(source_path, "input")
    images_dir = os.path.join(source_path, "images")
    sparse_dir = os.path.join(source_path, "sparse", "0")
    images_txt = os.path.join(sparse_dir, "images.txt")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points3d_txt = os.path.join(sparse_dir, "points3D.txt")
    points3d_bin = os.path.join(sparse_dir, "points3D.bin")

    input_images = list_images(input_dir)
    output_images = list_images(images_dir)
    registered_images = read_sparse_images(images_txt)
    sparse_model_format = "text"
    if not registered_images:
        registered_images = read_sparse_images_bin(images_bin)
        if registered_images:
            sparse_model_format = "binary"
    if not registered_images:
        sparse_model_format = "missing"
    registered_set = set(registered_images)
    missing_registration = [name for name in input_images if name not in registered_set]
    sparse_points3d_count = read_sparse_points_count(points3d_txt)
    if sparse_points3d_count == 0:
        sparse_points3d_count = read_sparse_points_count_bin(points3d_bin)

    input_indices = [idx for idx in (extract_frame_index(name) for name in input_images) if idx is not None]
    registered_indices = [idx for idx in (extract_frame_index(name) for name in registered_images) if idx is not None]
    missing_indices = sorted(set(input_indices) - set(registered_indices))

    consecutive_missing_ranges = []
    if missing_indices:
        start = prev = missing_indices[0]
        for value in missing_indices[1:]:
            if value == prev + 1:
                prev = value
                continue
            consecutive_missing_ranges.append([start, prev])
            start = prev = value
        consecutive_missing_ranges.append([start, prev])

    registered_gaps = []
    if len(registered_indices) >= 2:
        ordered_registered = sorted(registered_indices)
        for left, right in zip(ordered_registered, ordered_registered[1:]):
            gap = right - left
            if gap > 1:
                registered_gaps.append(
                    {
                        "from": left,
                        "to": right,
                        "gap": gap,
                    }
                )
        max_registered_gap = max((item["gap"] for item in registered_gaps), default=1)
    else:
        max_registered_gap = 0

    summary = {
        "source_path": os.path.abspath(source_path),
        "report_dir": os.path.abspath(report_dir),
        "exit_code": exit_code,
        "duration_sec": duration_sec,
        "input_image_count": len(input_images),
        "undistorted_image_count": len(output_images),
        "registered_image_count": len(registered_images),
        "registered_images": registered_images,
        "unregistered_image_count": len(missing_registration),
        "unregistered_images": missing_registration,
        "sparse_model_format": sparse_model_format,
        "sparse_points3d_count": sparse_points3d_count,
        "sequence": {
            "input_frame_indices": input_indices,
            "registered_frame_indices": registered_indices,
            "missing_frame_indices": missing_indices,
            "missing_ranges": consecutive_missing_ranges,
            "registered_gaps": registered_gaps,
            "max_registered_gap": max_registered_gap,
        },
        "database": db_stats,
        "log": log_stats,
        "snapshots": {
            "database_db": os.path.join(report_dir, "snapshots", "database.db"),
            "sparse_0": os.path.join(report_dir, "snapshots", "sparse", "0"),
        },
    }

    hints = []
    if exit_code != 0:
        hints.append("convert.py returned non-zero exit code; inspect convert_stdout_stderr.log first.")
    if db_stats["database_exists"] and db_stats["images_in_db"] < len(input_images):
        hints.append("Database image count is lower than input image count; feature extraction likely skipped or failed for some images.")
    if db_stats["match_pairs"] == 0:
        hints.append("Database contains zero match pairs; image matching likely failed or did not run.")
    if db_stats["two_view_geometries"] == 0:
        hints.append("No verified two-view geometries; feature matching may exist but geometric verification failed.")
    if len(registered_images) == 0:
        hints.append("Mapper produced zero registered images; pose graph initialization likely failed.")
    elif len(registered_images) < len(input_images):
        hints.append("Only part of the input images were registered; inspect unregistered_images and feature/match coverage.")
        if registered_gaps:
            hints.append("Registered frame sequence contains gaps; inspect sequence.registered_gaps and sequence.missing_ranges for likely break points.")
    if db_stats["keypoints_total"] == 0:
        hints.append("Database contains zero extracted keypoints; feature extraction is the first failure point.")
    if log_stats["error_lines"]:
        hints.append("Error lines were detected in stdout/stderr; inspect error_lines in summary or raw log.")
    summary["hints"] = hints
    return summary


def main():
    parser = argparse.ArgumentParser("Run convert.py and summarize matching failures")
    parser.add_argument("--convert_script", required=True, type=str)
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--python_executable", default=sys.executable, type=str)
    parser.add_argument("--output_root", default="", type=str)
    parser.add_argument("--label", default="", type=str)
    parser.add_argument("convert_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    convert_args = list(args.convert_args)
    if convert_args and convert_args[0] == "--":
        convert_args = convert_args[1:]

    source_path = os.path.abspath(os.path.expanduser(args.source_path))
    output_root = (
        os.path.abspath(os.path.expanduser(args.output_root))
        if args.output_root
        else os.path.join(source_path, "convert_reports")
    )
    ensure_dir(output_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = timestamp if not args.label else f"{timestamp}_{args.label}"
    report_dir = os.path.join(output_root, run_name)
    ensure_dir(report_dir)

    python_executable = args.python_executable.strip() if args.python_executable else ""
    if not python_executable:
        python_executable = sys.executable

    command = [
        python_executable,
        os.path.abspath(os.path.expanduser(args.convert_script)),
        "-s",
        source_path,
    ] + convert_args

    raw_log_path = os.path.join(report_dir, "convert_stdout_stderr.log")
    timeline_path = os.path.join(report_dir, "timeline.csv")
    summary_path = os.path.join(report_dir, "summary.json")

    stage_timestamps = []
    error_lines = []
    warning_lines = []
    current_stage = "startup"
    start_time = time.time()

    with open(raw_log_path, "w", encoding="utf-8") as raw_log, open(
        timeline_path, "w", newline="", encoding="utf-8"
    ) as timeline_file:
        timeline_writer = csv.writer(timeline_file)
        timeline_writer.writerow(["elapsed_sec", "stage", "line"])

        raw_log.write("# command: {}\n".format(" ".join(shlex.quote(part) for part in command)))
        raw_log.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            raw_log.write(line)
            raw_log.flush()

            elapsed = time.time() - start_time
            stripped = line.rstrip("\n")
            detected_stage = detect_stage(stripped)
            if detected_stage is not None:
                current_stage = detected_stage
                stage_timestamps.append({"elapsed_sec": elapsed, "stage": current_stage, "line": stripped})

            timeline_writer.writerow([f"{elapsed:.3f}", current_stage, stripped])

            lowered = stripped.lower()
            if "error" in lowered or "failed" in lowered:
                error_lines.append(stripped)
            elif "warning" in lowered or "warn" in lowered:
                warning_lines.append(stripped)

            print(stripped, flush=True)

        exit_code = process.wait()

    duration_sec = time.time() - start_time

    db_stats = inspect_colmap_database(os.path.join(source_path, "distorted", "database.db"))
    snapshot_if_exists(
        os.path.join(source_path, "distorted", "database.db"),
        os.path.join(report_dir, "snapshots", "database.db"),
    )
    snapshot_if_exists(
        os.path.join(source_path, "sparse", "0"),
        os.path.join(report_dir, "snapshots", "sparse", "0"),
    )
    log_stats = {
        "stage_transitions": stage_timestamps,
        "error_lines": error_lines[-50:],
        "warning_lines": warning_lines[-50:],
    }
    summary = build_summary(source_path, report_dir, exit_code, duration_sec, log_stats, db_stats)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Convert report written to {}".format(report_dir))
    print("Summary: {}".format(summary_path))
    print("Raw log: {}".format(raw_log_path))
    print("Timeline: {}".format(timeline_path))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
