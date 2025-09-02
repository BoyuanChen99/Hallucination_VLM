import pandas as pd
import os
import shutil
import time
import random
import subprocess
import requests
import json
import sys
import logging
from typing import Optional
import tempfile
from pathlib import Path

from transformers import logging as hf_logging


def disable_huggingface_warnings():
    hf_logging.set_verbosity_error()


def download_with_wget(url: str, out_path: str, rate_limit: str = "200k") -> bool:
    """
    Download `url` to `out_path` using wget with retries and timeouts.
    Returns True on success, False otherwise.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already present and non-empty
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] Already exists: {out_path}")
        return True

    # Ensure wget exists
    if shutil.which("wget") is None:
        print("[FAIL] `wget` not found on PATH.")
        return False

    # Download to a temp file first for atomic move on success
    with tempfile.NamedTemporaryFile(
        dir=str(out_path.parent), prefix=out_path.name + ".", suffix=".part", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "wget",
        "--tries=20",
        "--waitretry=5",
        "--retry-on-http-error=429,500,502,503,504",
        "--timeout=30",
        f"--limit-rate={rate_limit}",
        "-O", str(tmp_path),
        url,
    ]

    print(f"[START] Downloading {url} -> {out_path}")
    try:
        # Capture output so failures print nicely; keep return code semantics simple
        result = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(f"[DONE] wget finished with exit code {result.returncode}")
        if result.returncode != 0:
            # Show a compact error message
            err = (result.stderr or "").strip().splitlines()[-1:]  # last line if any
            if err:
                print(f"[FAIL] wget error: {err[0]}")
            else:
                print("[FAIL] wget failed with no stderr.")
            return False

        # Verify file exists and is non-empty
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            print("[FAIL] Download resulted in empty file.")
            return False

        # Atomic move into place
        os.replace(tmp_path, out_path)
        size_kb = out_path.stat().st_size / 1024
        print(f"[OK] Saved {out_path} ({size_kb:.1f} KB)")
        return True

    except FileNotFoundError:
        print("[FAIL] `wget` executable not found during run.")
        return False
    except Exception as e:
        print(f"[FAIL] Exception during download: {e}")
        return False
    finally:
        # Clean up temp file if it’s still around
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass



def load_dataframe(dataset_name, data_dir="../../../data", subset=None, subsplit=None):
    dataset_name = dataset_name.lower()
    if "haloquest" in dataset_name and "eval" in dataset_name:
        dataset_dir = os.path.join(data_dir, "haloquest")
        file_path = os.path.join(dataset_dir, f"haloquest-eval.csv")
        col_prompt = "question"
        col_url = "url"
        col_image = "image_name"
        image_dir = os.path.join(dataset_dir, "images")
        df = pd.read_csv(file_path)
        image_dir = os.path.join(dataset_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        for f in os.listdir(image_dir):
            f_path = os.path.join(image_dir, f)
            if os.path.isfile(f_path) and os.path.getsize(f_path) < 1024:
                os.remove(f_path)
        for idx, row in df.iterrows():
            image_name = row[col_image]
            image_url = row[col_url]
            image_path = os.path.join(image_dir, image_name)
            if not os.path.exists(image_path):
                download_with_wget(image_url, image_path)
        cols_to_remove = [c for c in ["Index", "index"] if c in df.columns]
        df = df.drop(columns=cols_to_remove)
    elif "chair" in dataset_name:
        json_file = os.path.join(data_dir, "chair/chair_1994.json")
        image_dir = os.path.join(data_dir, "coco/val2014")
        # Create an empty dataframe
        df = pd.DataFrame()
        with open(json_file, "r") as f:
            for line in f:
                json_data = json.loads(line)
                if df.empty:
                    df = pd.json_normalize(json_data)
                else:
                    df = pd.concat([df, pd.json_normalize(json_data)], ignore_index=True)
        col_prompt = "text"
        col_image = "image"
    elif "pope" in dataset_name:
        questions = [json.loads(q) for q in open(f"{data_dir}/pope/{subset}/{subset}_pope_{subsplit}.json", "r")]
        df = pd.json_normalize(questions)
        col_prompt = "text"
        col_image = "image"
        image_dir = os.path.join(data_dir, "coco", "val2014")
    return df, col_prompt, col_image, image_dir


def process_response(response, benchmark):
    if "pope" in benchmark:
        response = ''.join(c for c in response if c.isalpha() or c.isspace())
        response = response.lower()
    return response


def concatenate_response(response, row, df_output, col_image):
    # Ensure helper columns exist (nullable integer dtype)
    for c in ("idx_image", "idx_question"):
        if c not in df_output.columns:
            df_output[c] = pd.Series(dtype="Int64")
    # Columns we never want
    NEVER_COLS = {"Index", "index", "Unnamed: 0"}
    # Drop from row (Series -> use labels)
    row = row.drop(labels=[c for c in NEVER_COLS if c in row.index])
    # Drop from df_output (DataFrame -> use columns)
    df_output = df_output.drop(columns=[c for c in NEVER_COLS if c in df_output.columns], errors="ignore")
    # Add response
    row = row.copy()
    row["response"] = response.replace("\n", "\\n")
    if len(df_output) == 0:
        # start counters
        row["idx_image"] = 0
        row["idx_question"] = 0
        df_output = pd.concat([df_output, row.to_frame().T], ignore_index=True)
        # Ensure forbidden columns didn't appear
        df_output = df_output.drop(columns=[c for c in NEVER_COLS if c in df_output.columns], errors="ignore")
        return df_output
    # Compare with last row
    last = df_output.iloc[-1]
    same_image = (
        (col_image in row.index)
        and (col_image in df_output.columns)
        and pd.notna(last.get(col_image))
        and row[col_image] == last[col_image]
    )
    last_idx_image = -1 if pd.isna(last.get("idx_image")) else int(last["idx_image"])
    last_idx_question = -1 if pd.isna(last.get("idx_question")) else int(last["idx_question"])
    if same_image:
        row["idx_image"] = last_idx_image
        row["idx_question"] = last_idx_question + 1
    else:
        row["idx_image"] = last_idx_image + 1
        row["idx_question"] = 0
    df_output = pd.concat([df_output, row.to_frame().T], ignore_index=True)
    # Final guard: ensure "Index" (or variants) never remain
    df_output = df_output.drop(columns=[c for c in NEVER_COLS if c in df_output.columns], errors="ignore")
    return df_output
