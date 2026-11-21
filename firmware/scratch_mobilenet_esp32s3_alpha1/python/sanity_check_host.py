import sys

from inferencia_pc import MODEL_PATH, create_interpreter, run_inference
from teste_inferencia import IMAGE_INDEX, CLASS_NAMES, get_test_image


def main():
    indices = [int(value) for value in sys.argv[1:]] or [IMAGE_INDEX]
    interpreter = create_interpreter(MODEL_PATH)
    correct = 0
    for index in indices:
        image, true_label = get_test_image(index)
        result, _ = run_inference(MODEL_PATH, image, interpreter=interpreter)
        hit = result["predicted_class"] == true_label
        correct += int(hit)
        print(
            f"Amostra {index}: real={CLASS_NAMES[true_label]} ({true_label}) "
            f"previsão={result['predicted_name']} ({result['predicted_class']}) "
            f"confiança={result['confidence']:.6f} acerto={hit} "
            f"input_hash={result['input_fnv1a']} output_hash={result['output_fnv1a']}"
        )
    print(f"Acertos nas amostras selecionadas: {correct}/{len(indices)}")


if __name__ == "__main__":
    main()
