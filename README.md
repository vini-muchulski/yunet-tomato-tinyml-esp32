# YuNet for Tomato Leaf Classification on ESP32-S3

This research evaluates an adaptation of YuNet-N for classifying eight tomato leaf conditions in the Tomato-Village dataset. It compares YuNet-N with MobileNetV2 variants trained from scratch or fine-tuned from ImageNet weights.

The experimental workflow trains models in PyTorch, exports quantized INT8 models to TensorFlow Lite, and evaluates inference on the ESP32-S3 with TensorFlow Lite Micro and ESP-NN. A host computer preprocesses the images and sends input tensors to the device over HTTP.
