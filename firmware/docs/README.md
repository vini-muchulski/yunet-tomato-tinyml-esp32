# Documentação do ESP32-S3

| Guia | Conteúdo |
|---|---|
| [Principal](README_principal.md) | Estrutura, submódulos, dependências, hardware, comandos e adaptação para outras tarefas. |
| [YuNet](README_yunet.md) | Modelo, classes, contrato HTTP, avaliação histórica e limitações numéricas. |
| [MobileNetV2 alpha1 e alpha0.5](README_mobilenet.md) | Modelos corrigidos e versões scratch alpha1 e alpha0.5, avaliações no ESP32, comparação com PC e YuNet, tempos, falhas e memória. |
| [Reexportação INT8](README_reexportacao.md) | Pooling explícito, incorporação de buffers, ambiente uv e validação no PC. |
| [OP-TRACE](README_debug.md) | Ativação, captura serial, comparação por operador e aplicação do patch. |

Os caminhos das aplicações são relativos a `esp32-s3-git-espnn-tflite`, salvo indicação diferente. Execute comandos de PlatformIO dentro da aplicação escolhida; execute a reexportação na pasta que contém `reexportar.py`. Os links desta pasta são relativos a `docs/`.

O patch [esp-tflite-micro-optrace.patch](esp-tflite-micro-optrace.patch) preserva a instrumentação local do submódulo. Os documentos em `git_common_components` pertencem às bibliotecas externas e os de `butterflies_antigo` registram experimentos históricos.

Resultados atualizados em 02/10/2026. A aprovação dos modelos no PC não comprova equivalência numérica ou desempenho no ESP32.
