import json
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
INPUT_INFO_PATH = Path(os.environ.get("INPUT_INFO_PATH", BASE_DIR / "input_info.json"))
DATASET_DIR = Path(os.environ.get(
    "DATASET_DIR",
    BASE_DIR.parents[2] / "dataset" / "tomato_village" / "tomato_village"
    / "Variant-a(Multiclass Classification)" / "test",
))

#IMAGE_INDEX =  82
#IMAGE_INDEX =  250
IMAGE_INDEX =  180

SHOW_PLOT = True
OUTPUT_PLOT_PATH = Path(os.environ.get(
    "OUTPUT_PLOT_PATH", BASE_DIR / "results" / f"tomato_village_{IMAGE_INDEX}_plot.png"
))
ESP32_IP = os.environ.get("ESP32_IP", "192.168.3.22")
PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
REQUEST_TIMEOUT = 300
CLASS_NAMES = [
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
]


def load_input_info(path):
    with open(path) as f:
        return json.load(f)


def get_test_samples():
    samples = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_DIR / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Classe não encontrada: {class_dir}")
        samples.extend((path, label) for path in sorted(class_dir.rglob("*.jpg")))
    return samples


def get_test_image(index):
    samples = get_test_samples()
    if not 0 <= index < len(samples):
        raise IndexError(f"IMAGE_INDEX deve estar entre 0 e {len(samples) - 1}: {index}")
    path, label = samples[index]
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")), label


def preprocess_image(image_data, size):
    height, width = size
    return np.asarray(Image.fromarray(image_data).convert("RGB").resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def quantize_input(img_u8, info):
    if info["dtype"] != "int8" or info["layout"] != "NCHW" or info["scale"] <= 0:
        raise ValueError("Entrada esperada: int8 NCHW com escala positiva")
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (img_u8.astype(np.float32) / 255.0 - mean) / std
    nchw = normalized.transpose(2, 0, 1)
    quantized = np.round(nchw / float(info["scale"]) + int(info["zero_point"]))
    return np.clip(quantized, -128, 127).astype(np.int8)


def fnv1a32(data):
    value = 2166136261
    for byte in memoryview(np.ascontiguousarray(data)).cast("B"):
        value = ((value ^ int(byte)) * 16777619) & 0xFFFFFFFF
    return value


def send_bin(qimg):
    response = requests.post(PREDICT_URL, data=qimg.tobytes(),
                             headers={"Content-Type": "application/octet-stream"},
                             timeout=(5, REQUEST_TIMEOUT))
    response.raise_for_status()
    result = response.json()
    if not result.get("success"):
        raise RuntimeError(result.get("error_message", "Inference failed"))
    return result


def plot_result(image, result, save_path=OUTPUT_PLOT_PATH, show_plot=SHOW_PLOT):
    import matplotlib

    if show_plot:
        try:
            matplotlib.use("TkAgg", force=True)
        except (ImportError, RuntimeError):
            matplotlib.use("Agg", force=True)
            show_plot = False
    else:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    try:
        fig, ax = plt.subplots()
    except (ImportError, RuntimeError):
        plt.switch_backend("Agg")
        show_plot = False
        fig, ax = plt.subplots()

    ax.imshow(image.astype(np.uint8))
    ax.set_title(
        f"Sent to ESP32\n"
        f"True: {result['true_name']} ({result['true_class']}) | "
        f"Pred: {result['predicted_name']} ({result['predicted_class']})"
    )
    ax.axis("off")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Plot salvo em: {save_path}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    info = load_input_info(INPUT_INFO_PATH)
    image, true_label = get_test_image(IMAGE_INDEX)
    resized_image = preprocess_image(image, info["size"])
    qimg = quantize_input(resized_image, info)
    result = send_bin(qimg)
    if result["input_fnv1a"] != f"{fnv1a32(qimg):08x}":
        raise RuntimeError("PC and ESP32 input hashes differ")
    predicted = int(result["predicted_class"])
    if len(result["scores"]) != len(CLASS_NAMES) or not 0 <= predicted < len(CLASS_NAMES):
        raise RuntimeError("Saída inválida do Tomato Village")
    result["image_index"] = IMAGE_INDEX
    result["true_class"] = true_label
    result["true_name"] = CLASS_NAMES[true_label]
    result["predicted_name"] = CLASS_NAMES[predicted]
    result["correct"] = predicted == true_label
    print(json.dumps(result, indent=2))
    print(f"Inferência: {result['inference_us'] / 1_000_000:.6f} s")
    plot_result(image, result)


if __name__ == "__main__":
    main()
