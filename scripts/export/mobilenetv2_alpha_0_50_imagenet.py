import argparse
import json
from datetime import datetime
from pathlib import Path

import flatbuffers
import litert_torch
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from ai_edge_quantizer import quantizer, recipe
from torchvision import datasets, transforms
from tqdm import tqdm

SEED = 42
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


class MobileNetV2FixedAvgPool(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs):
        outputs = self.backbone.forward_features(inputs)
        outputs = F.avg_pool2d(outputs, kernel_size=(7, 7), stride=1)
        outputs = torch.flatten(outputs, 1)
        return self.backbone.classifier(outputs)


def load_datasets(root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = datasets.ImageFolder(root / "train", transform=transform)
    test = datasets.ImageFolder(root / "test", transform=transform)
    if train.classes != CLASS_NAMES or test.class_to_idx != train.class_to_idx:
        raise ValueError("Classes incompatíveis com o firmware Tomato Village")
    return train, test


def export_fp32(checkpoint, sample, target):
    backbone = timm.create_model(
        "mobilenetv2_050.lamb_in1k", pretrained=False, num_classes=len(CLASS_NAMES), drop_rate=0.3
    )
    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    backbone.eval()
    model = MobileNetV2FixedAvgPool(backbone).eval()
    with torch.no_grad():
        features = backbone.forward_features(sample)
        if tuple(features.shape[-2:]) != (7, 7):
            raise ValueError(f"Feature map incompatível: {tuple(features.shape)}")
        expected = backbone(sample)
        actual = model(sample)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        error = float((actual - expected).abs().max())
    litert_torch.convert(model, (sample,)).export(str(target))
    return error


def quantize_model(source, target, dataset):
    interpreter = Interpreter(model_path=str(source))
    signatures = interpreter.get_signature_list()
    if len(signatures) != 1:
        raise ValueError(f"Esperada uma assinatura: {signatures}")
    signature_name, signature = next(iter(signatures.items()))
    if len(signature["inputs"]) != 1:
        raise ValueError("Esperada uma única entrada")
    input_name = signature["inputs"][0]
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator)[:100].tolist()
    calibration = {
        signature_name: [
            {input_name: dataset[index][0].unsqueeze(0).numpy().astype(np.float32)}
            for index in indices
        ]
    }
    converter = quantizer.Quantizer(str(source))
    converter.load_quantization_recipe(recipe.static_wi8_ai8())
    result = converter.quantize(converter.calibrate(calibration))
    result.export_model(str(target), overwrite=False)


def inline_buffers(source, target):
    contents = source.read_bytes()
    model = schema.Model.GetRootAsModel(contents, 0)
    model_object = schema.ModelT.InitFromObj(model)
    converted = 0
    for index, buffer in enumerate(model_object.buffers):
        original = model.Buffers(index)
        offset, size = original.Offset(), original.Size()
        if not offset and not size:
            continue
        if not offset or not size or offset + size > len(contents):
            raise ValueError(f"Buffer externo inválido: {index}")
        buffer.data = bytearray(contents[offset:offset + size])
        buffer.offset = 0
        buffer.size = 0
        converted += 1
    builder = flatbuffers.Builder(len(contents))
    builder.Finish(model_object.Pack(builder), file_identifier=b"TFL3")
    output = bytes(builder.Output())
    packed = schema.Model.GetRootAsModel(output, 0)
    for index in range(model.BuffersLength()):
        original = model.Buffers(index)
        buffer = packed.Buffers(index)
        if buffer.Offset() or buffer.Size():
            raise ValueError(f"Buffer externo restante: {index}")
        expected = bytes(original.DataAsNumpy()) if original.DataLength() else b""
        if original.Offset():
            expected = contents[original.Offset():original.Offset() + original.Size()]
        actual = bytes(buffer.DataAsNumpy()) if buffer.DataLength() else b""
        if actual != expected:
            raise ValueError(f"Dados alterados no buffer {index}")
    target.write_bytes(output)
    return converted


def validate_model(path, resolver, int8=False):
    interpreter = Interpreter(model_path=str(path), experimental_op_resolver_type=resolver)
    interpreter.allocate_tensors()
    operators = [operation["op_name"] for operation in interpreter._get_ops_details()]
    supported = {
        "TRANSPOSE", "PAD", "CONV_2D", "DEPTHWISE_CONV_2D", "ADD",
        "AVERAGE_POOL_2D", "FULLY_CONNECTED", "RESHAPE",
    }
    if "AVERAGE_POOL_2D" not in operators or set(operators) - supported:
        raise ValueError(f"Operadores incompatíveis: {operators}")
    if int8:
        for details, shape in [(interpreter.get_input_details()[0], [1, 3, 224, 224]),
                               (interpreter.get_output_details()[0], [1, 8])]:
            if (details["shape"].tolist() != shape or details["dtype"] != np.int8
                    or details["quantization"][0] <= 0):
                raise ValueError(f"Tensor incompatível: {details}")
    return interpreter, operators


def evaluate(interpreter, dataset, name):
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    scale, zero_point = input_info["quantization"]
    predictions = []
    correct = 0
    first_output = None
    constant = True
    for image, label in tqdm(dataset, desc=name, total=len(dataset), unit="imagem"):
        values = np.clip(np.round(image.numpy()[None] / scale + zero_point), -128, 127)
        interpreter.set_tensor(input_info["index"], values.astype(np.int8))
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])[0]
        if first_output is None:
            first_output = output.copy()
        elif not np.array_equal(output, first_output):
            constant = False
        prediction = int(np.argmax(output))
        predictions.append(prediction)
        correct += prediction == label
    if constant:
        raise RuntimeError(f"{name}: logits constantes no conjunto de teste")
    return {"accuracy": correct / len(dataset), "correct": correct,
            "samples": len(dataset)}, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    train, test = load_datasets(args.dataset)
    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    fp32 = output_dir / "model_fp32_avgpool.tflite"
    external = output_dir / "model_int8_external.tflite"
    target = output_dir / "model_int8_avgpool.tflite"
    error = export_fp32(args.checkpoint, test[0][0].unsqueeze(0), fp32)
    validate_model(fp32, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    quantize_model(fp32, external, train)
    converted = inline_buffers(external, target)
    report = {"pytorch_max_absolute_error": error, "inlined_buffers": converted,
              "classes": CLASS_NAMES, "model": str(target), "resolvers": {}}
    predictions = []
    for resolver in [OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                     OpResolverType.BUILTIN_REF]:
        interpreter, operators = validate_model(target, resolver, int8=True)
        metrics, predicted = evaluate(interpreter, test, resolver.name)
        report["resolvers"][resolver.name] = metrics
        predictions.append(predicted)
        report["operators"] = sorted(set(operators))
    metrics = list(report["resolvers"].values())
    difference = abs(metrics[0]["accuracy"] - metrics[1]["accuracy"])
    report["accuracy_difference"] = difference
    report["prediction_disagreements"] = sum(a != b for a, b in zip(*predictions))
    report["approved_on_pc"] = difference <= 0.01
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["approved_on_pc"]:
        raise RuntimeError("Diferença de acurácia entre resolvers excede um ponto percentual")
    input_info = interpreter.get_input_details()[0]
    (output_dir / "input_info.json").write_text(json.dumps({
        "shape": input_info["shape"].tolist(), "layout": "NCHW", "size": [224, 224],
        "scale": float(input_info["quantization"][0]),
        "zero_point": int(input_info["quantization"][1]), "dtype": "int8",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Artefatos validados no PC: {output_dir}")


if __name__ == "__main__":
    main()
