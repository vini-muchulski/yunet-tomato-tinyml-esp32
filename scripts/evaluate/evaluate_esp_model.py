import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

from teste_inferencia import (
    CLASS_NAMES,
    ESP32_IP,
    INPUT_INFO_PATH,
    REQUEST_TIMEOUT,
    fnv1a32,
    get_test_samples,
    load_input_info,
    preprocess_image,
    quantize_input,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get(
    "EVALUATION_RESULTS_DIR",
    BASE_DIR.parents[1] / "results" / "esp32s3",
))
SEED = 42
PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
STATUS_URL = f"http://{ESP32_IP}/status"


def select_samples(samples, count, seed):
    if count > len(samples):
        raise ValueError(f"Solicitadas {count} imagens, disponíveis: {len(samples)}")
    rng = np.random.default_rng(seed)
    groups = {
        class_index: [
            index for index, (_, label) in enumerate(samples) if label == class_index
        ]
        for class_index in range(len(CLASS_NAMES))
    }
    exact_counts = {
        class_index: count * len(indices) / len(samples)
        for class_index, indices in groups.items()
    }
    selected_counts = {
        class_index: int(value)
        for class_index, value in exact_counts.items()
    }
    remaining = count - sum(selected_counts.values())
    priority = sorted(
        exact_counts,
        key=lambda class_index: exact_counts[class_index] - selected_counts[class_index],
        reverse=True,
    )
    for class_index in priority[:remaining]:
        selected_counts[class_index] += 1

    selected = []
    for class_index, indices in groups.items():
        chosen = rng.choice(indices, size=selected_counts[class_index], replace=False)
        selected.extend((int(index), *samples[int(index)]) for index in chosen)
    rng.shuffle(selected)
    return selected


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def validate_status(session, input_info):
    response = session.get(STATUS_URL, timeout=(3, 10))
    response.raise_for_status()
    status = response.json()
    expected = {
        "model_initialized": True,
        "input_shape": input_info["shape"],
        "input_dtype": input_info["dtype"],
        "input_bytes": int(np.prod(input_info["shape"])),
        "output_classes": len(CLASS_NAMES),
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise RuntimeError(
                f"Status incompatível: {key}={status.get(key)!r}, esperado={value!r}"
            )
    if not np.isclose(status.get("input_scale", 0.0), input_info["scale"]):
        raise RuntimeError("Escala de entrada do ESP32 difere de input_info.json")
    if status.get("input_zero_point") != input_info["zero_point"]:
        raise RuntimeError("Zero-point de entrada do ESP32 difere de input_info.json")
    return status


def send_tensor(session, tensor):
    started_at = time.perf_counter()
    response = session.post(
        PREDICT_URL,
        data=tensor.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=(5, REQUEST_TIMEOUT),
    )
    response.raise_for_status()
    return response.json(), (time.perf_counter() - started_at) * 1000.0


def evaluate_sample(session, dataset_index, path, true_class, input_info):
    tensor = quantize_input(
        preprocess_image(load_image(path), input_info["size"]),
        input_info,
    )
    input_hash = f"{fnv1a32(tensor):08x}"
    started_at = time.perf_counter()
    try:
        response, request_ms = send_tensor(session, tensor)
        if not response.get("success"):
            raise RuntimeError(response.get("error_message", "Inferência falhou"))
        if response.get("input_fnv1a") != input_hash:
            raise RuntimeError("Hash da entrada divergiu entre PC e ESP32")
        predicted_class = int(response["predicted_class"])
        scores = response["scores"]
        if not 0 <= predicted_class < len(CLASS_NAMES):
            raise RuntimeError(f"Classe predita inválida: {predicted_class}")
        if len(scores) != len(CLASS_NAMES):
            raise RuntimeError(f"Quantidade inválida de scores: {len(scores)}")
        return {
            "dataset_index": dataset_index,
            "image": str(path),
            "true_class": true_class,
            "true_name": CLASS_NAMES[true_class],
            "predicted_class": predicted_class,
            "predicted_name": CLASS_NAMES[predicted_class],
            "correct": int(predicted_class == true_class),
            "success": 1,
            "confidence": float(response["confidence"]),
            "scores": json.dumps(scores),
            "input_fnv1a": input_hash,
            "output_fnv1a": response.get("output_fnv1a", ""),
            "request_ms": request_ms,
            "inference_ms": float(response.get("inference_us", 0)) / 1000.0,
            "total_processing_ms": float(response.get("total_processing_us", 0)) / 1000.0,
            "internal_free": int(response.get("internal_free", -1)),
            "psram_free": int(response.get("psram_free", -1)),
            "error": "",
        }
    except Exception as error:
        return {
            "dataset_index": dataset_index,
            "image": str(path),
            "true_class": true_class,
            "true_name": CLASS_NAMES[true_class],
            "predicted_class": -1,
            "predicted_name": "unknown",
            "correct": 0,
            "success": 0,
            "confidence": 0.0,
            "scores": "[]",
            "input_fnv1a": input_hash,
            "output_fnv1a": "",
            "request_ms": (time.perf_counter() - started_at) * 1000.0,
            "inference_ms": -1.0,
            "total_processing_ms": -1.0,
            "internal_free": -1,
            "psram_free": -1,
            "error": str(error),
        }


def confusion_matrix(results):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int32)
    for result in results:
        if result["success"]:
            matrix[result["true_class"], result["predicted_class"]] += 1
    return matrix


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(run_dir, matrix):
    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row.tolist()])

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predita")
    axis.set_ylabel("Real")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def percentile(values, value):
    return float(np.percentile(values, value)) if values else None


def build_summary(results, status, started_at, finished_at):
    successful = [result for result in results if result["success"]]
    correct = sum(result["correct"] for result in successful)
    request_times = [result["request_ms"] for result in successful]
    inference_times = [result["inference_ms"] for result in successful]
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "esp32_ip": ESP32_IP,
        "seed": SEED,
        "requested_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(results) - len(successful),
        "correct_predictions": correct,
        "accuracy": correct / len(successful) if successful else None,
        "request_ms": {
            "mean": statistics.fmean(request_times) if request_times else None,
            "p50": percentile(request_times, 50),
            "p95": percentile(request_times, 95),
            "p99": percentile(request_times, 99),
        },
        "inference_ms": {
            "mean": statistics.fmean(inference_times) if inference_times else None,
            "p50": percentile(inference_times, 50),
            "p95": percentile(inference_times, 95),
            "p99": percentile(inference_times, 99),
        },
        "device_status": status,
    }


def main():
    started_at = datetime.now().astimezone()
    run_dir = RESULTS_DIR / started_at.strftime("%Y%m%d_%H%M%S")
    input_info = load_input_info(INPUT_INFO_PATH)
    samples = get_test_samples()
    selected_samples = select_samples(samples, len(samples), SEED)
    results = []

    with requests.Session() as session:
        status = validate_status(session, input_info)
        run_dir.mkdir(parents=True, exist_ok=False)
        for dataset_index, path, true_class in tqdm(
            selected_samples, desc="Inferência ESP32", unit="imagem"
        ):
            result = evaluate_sample(session, dataset_index, path, true_class, input_info)
            results.append(result)

    finished_at = datetime.now().astimezone()
    matrix = confusion_matrix(results)
    summary = build_summary(results, status, started_at, finished_at)
    save_csv(run_dir / "predictions.csv", results)
    save_confusion_matrix(run_dir, matrix)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"resultados={run_dir}")


if __name__ == "__main__":
    main()
