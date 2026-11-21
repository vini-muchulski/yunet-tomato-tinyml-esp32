import json
import os
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from teste_inferencia import (
    IMAGE_INDEX, CLASS_NAMES, INPUT_INFO_PATH, get_test_image,
    load_input_info, preprocess_image, quantize_input, fnv1a32,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH", BASE_DIR.parent / "data" / "model_int8_avgpool.tflite"
))
RESULTS_DIR = BASE_DIR / "results_pc"
SHOW_PLOT = True


def create_interpreter(model_path=MODEL_PATH, preserve_all_tensors=False):
    interpreter = Interpreter(
        model_path=str(model_path),
        experimental_preserve_all_tensors=preserve_all_tensors,
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if list(inp["shape"]) != [1, 3, 224, 224] or inp["dtype"] != np.int8:
        raise ValueError("Entrada esperada: int8 [1,3,224,224]")
    if list(out["shape"]) != [1, 8] or out["dtype"] != np.int8:
        raise ValueError("Saída esperada: int8 [1,8]")
    if not np.isfinite(out["quantization"][0]) or out["quantization"][0] <= 0:
        raise ValueError("Quantização de saída inválida")
    info = load_input_info(INPUT_INFO_PATH)
    if (info["shape"] != list(inp["shape"]) or
            info["layout"] != "NCHW" or
            info["dtype"] != inp["dtype"].__name__ or
            (info["scale"], info["zero_point"]) != inp["quantization"]):
        raise ValueError("input_info.json difere do modelo; regenere os metadados")
    return interpreter


def prepare_input(image):
    info = load_input_info(INPUT_INFO_PATH)
    return quantize_input(preprocess_image(image, info["size"]), info)


def run_inference(model_path, image, interpreter=None):
    if interpreter is None:
        interpreter = create_interpreter(model_path)
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    quantized = prepare_input(image)
    interpreter.set_tensor(inp["index"], quantized[None])
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])[0]
    scale, zero_point = out["quantization"]
    scores = (raw.astype(np.float32) - zero_point) * scale
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores inválidos")
    predicted = int(np.argmax(scores))
    result = {
        "predicted_class": predicted,
        "predicted_name": CLASS_NAMES[predicted],
        "confidence": float(scores[predicted]),
        "scores": scores.tolist(),
        "input_fnv1a": f"{fnv1a32(quantized):08x}",
        "output_fnv1a": f"{fnv1a32(raw):08x}",
    }
    return result, raw


def plot_result(image, result, save_path, show_plot=SHOW_PLOT):
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
        f"Inference on PC\n"
        f"True: {result['true_name']} ({result['true_class']}) | "
        f"Pred: {result['predicted_name']} ({result['predicted_class']})"
    )
    ax.axis("off")
    fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    image, true_label = get_test_image(IMAGE_INDEX)
    result, raw = run_inference(MODEL_PATH, image)
    result.update(image_index=IMAGE_INDEX, true_class=true_label,
                  true_name=CLASS_NAMES[true_label],
                  correct=result["predicted_class"] == true_label)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"tomato_village_{IMAGE_INDEX}"
    np.save(RESULTS_DIR / f"{stem}_output_int8_pc.npy", raw)
    (RESULTS_DIR / f"{stem}_result_pc.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    plot_result(image, result, RESULTS_DIR / f"{stem}_plot_pc.png")


if __name__ == "__main__":
    main()
