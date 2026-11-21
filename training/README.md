# Training experiments

These experiments use the eight-class Tomato-Village dataset with 224 × 224 RGB images. Each run directory contains the original notebook with its saved cell outputs, including training curves and evaluation plots, and the corresponding PyTorch checkpoints and TensorFlow Lite exports.

| Architecture | Experiment | Directory |
| --- | --- | --- |
| YuNet-N | Reference without dropout | [baseline](yunet_n/baseline/) |
| YuNet-N | Dropout 0.1 | [dropout_0_1](yunet_n/dropout_0_1/) |
| YuNet-N | Dropout 0.2 | [dropout_0_2](yunet_n/dropout_0_2/) |
| YuNet-N | Dropout 0.3 | [dropout_0_3](yunet_n/dropout_0_3/) |
| YuNet-N | Five seeds (43–47) without dropout | [multi_seed](yunet_n/multi_seed/) |
| MobileNetV2 | ImageNet fine-tuning, α = 0.50 | [imagenet_alpha_0_50](mobilenet_v2/imagenet_alpha_0_50/) |
| MobileNetV2 | ImageNet fine-tuning, α = 1.00 | [imagenet_alpha_1_00](mobilenet_v2/imagenet_alpha_1_00/) |
| MobileNetV2 | Training from scratch, α = 0.50 | [scratch_alpha_0_50](mobilenet_v2/scratch_alpha_0_50/) |
| MobileNetV2 | Training from scratch, α = 1.00 | [scratch_alpha_1_00](mobilenet_v2/scratch_alpha_1_00/) |

## Saved artifacts

- `tflite_models/` contains the original FP32, INT8 and INT8A16 exports. The [multi-seed run](yunet_n/multi_seed/) includes one checkpoint per seed and a JSON log under `logs/`. Its TFLite models were exported from the seed 47 checkpoint.

See [results](../results/) for benchmark artifacts and deployment export validation.

The notebooks retain their original dataset paths. Update the dataset path in the configuration cell before running a notebook in a different environment.
