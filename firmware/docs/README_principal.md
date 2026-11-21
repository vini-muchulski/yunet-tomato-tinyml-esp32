# TFLite Micro e ESP-NN no ESP32-S3

Este projeto reúne dependências obtidas por Git e aplicações de classificação Tomato Village: YuNet, MobileNetV2 alpha1 e alpha0.5, além das variantes MobileNetV2 treinadas do zero em `scratch_mobilenet_esp32s3_alpha1` e `scratch_mobilenet_esp32s3_alpha05`. Todas usam PlatformIO, ESP-IDF, TensorFlow Lite Micro e ESP-NN para ESP32-S3. A documentação das aplicações atuais está centralizada em `docs/`; comece pelo [índice](README.md).

## Estrutura

```text
firmware/
├── git_common_components/
│   ├── esp-nn/
│   ├── esp-tflite-micro/
│   ├── esp_littlefs/
│   └── tflite-micro/
├── yunet_esp32s3/
│   ├── CMakeLists.txt
│   ├── platformio.ini
│   ├── sdkconfig.defaults
│   ├── sdkconfig.esp32s3
│   ├── partitions.csv
│   ├── dependencies.lock
│   ├── src/
│   │   ├── main.cpp
│   │   └── CMakeLists.txt
│   ├── data/model_int8.tflite
│   ├── python/
│   │   ├── input_info.json
│   │   ├── get_input_info.py
│   │   ├── teste_inferencia.py
│   │   ├── evaluate_esp_model.py
│   │   ├── inferencia_pc.py
│   │   ├── sanity_check_host.py
│   │   └── trace_operadores_pc.py
│   └── training_code/
├── mobilenet_esp32s3_alpha1/
│   ├── data/model_int8_avgpool.tflite
│   ├── src/main.cpp
│   ├── python/
│   └── training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8/
├── mobilenet_esp32s3_alpha05/
│   ├── data/model_int8_avgpool.tflite
│   ├── src/main.cpp
│   ├── python/
│   └── training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8/
├── scratch_mobilenet_esp32s3_alpha1/
│   ├── data/model_int8_avgpool.tflite
│   ├── python/results_esp32/20261002_102050/
│   └── training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8/
├── scratch_mobilenet_esp32s3_alpha05/
│   ├── python/results_esp32/20261002_110629/
│   └── training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8/
└── docs/
    ├── README.md
    ├── README_principal.md
    ├── README_yunet.md
    ├── README_mobilenet.md
    ├── README_reexportacao.md
    ├── README_debug.md
    └── esp-tflite-micro-optrace.patch
```

O dataset fica na raiz de `ubiquos_tomato_village`, em `dataset/tomato_village/tomato_village/Variant-a(Multiclass Classification)/test`. Ele não fica dentro de `yunet_esp32s3`.

## Dependências e submódulos

| Diretório | Repositório | Uso neste firmware |
|---|---|---|
| `esp-nn` | [espressif/esp-nn](https://github.com/espressif/esp-nn) | Operadores quantizados otimizados para ESP32-S3. |
| `esp-tflite-micro` | [espressif/esp-tflite-micro](https://github.com/espressif/esp-tflite-micro) | Runtime TFLite Micro integrado ao ESP-IDF e à ESP-NN. |
| `esp_littlefs` | [joltwallet/esp_littlefs](https://github.com/joltwallet/esp_littlefs) | Montagem da partição LittleFS que armazena o modelo. |
| `tflite-micro` | [tensorflow/tflite-micro](https://github.com/tensorflow/tflite-micro) | Código original para referência e outras integrações; não é compilado por esta aplicação. |

Os quatro diretórios são submódulos em `firmware/git_common_components/`, registrados no `.gitmodules` da raiz do repositório.

Commits usados na execução de referência de 29/09/2026:

| Dependência | Commit | Versão declarada |
|---|---|---|
| ESP-NN | `d9be5422a79f4115e4275452dcaba3fe363a697d` | `1.4.1` |
| esp-tflite-micro | `dd4085950a119d9e33a6de1d7be1fc7fe29e0cfc` | `1.4.1` |
| esp_littlefs | `c3fe82d4d2794c7db7a9b5c7d9ec151e32fc001b` | `1.14.8` |
| TFLite Micro original | `6add2715939e9266641eed93d42fe23016a7b46e` | Não utilizada pelo firmware. |

Para inicializar as dependências, execute na raiz de `yunet-tomato-tinyml-esp32`:

```bash
git submodule update --init --recursive
git submodule status --recursive
```

Esse comando restaura os commits registrados no projeto. `git submodule update --remote` busca versões das branches remotas e pode alterar a combinação de versões; use apenas quando a intenção for atualizar as dependências. Uma atualização precisa ser validada e sua nova referência registrada no repositório principal.

## Como o firmware encontra os componentes

O `CMakeLists.txt` de cada aplicação acrescenta os componentes usados pelo firmware:

```cmake
list(APPEND EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../git_common_components/esp-tflite-micro"
    "${CMAKE_CURRENT_LIST_DIR}/../git_common_components/esp-nn"
    "${CMAKE_CURRENT_LIST_DIR}/../git_common_components/esp_littlefs")
```

Essa declaração fica antes de incluir `project.cmake` do ESP-IDF. Use `list(APPEND ...)`: o PlatformIO já fornece `src` por `EXTRA_COMPONENT_DIRS`. Substituir a lista com `set(...)` remove o componente da aplicação e causa `Couldn't find the main target of the project!`.

`src/CMakeLists.txt` declara `esp-tflite-micro` em `REQUIRES` e `esp_littlefs` em `PRIV_REQUIRES`. O manifesto do runtime exige `espressif/esp-nn`; o ESP-IDF Component Manager dá prioridade ao componente local `esp-nn` registrado no CMake.

Não acrescente todo o diretório `git_common_components` indiscriminadamente: o TFLite original também está ali, mas não faz parte desta integração.

Outras dependências:

- ESP-IDF e toolchain: fornecidos pelo PlatformIO. A execução de referência usou plataforma `espressif32 6.12.0` e ESP-IDF `5.5.0`.
- LittleFS: fornecido pelo submódulo `esp_littlefs`. A inicialização recursiva também baixa o núcleo LittleFS em `esp_littlefs/src/littlefs/`.
- FlatBuffers, gemmlowp, suporte de profiler ruy e KissFFT: incluídos no componente `esp-tflite-micro`. KissFFT atende ao frontend de áudio, não ao processamento de imagens deste modelo.

O ESP-NN e o esp-tflite-micro exigem ESP-IDF >= 5.1. O Component Manager atualiza o lock ao reconfigurar o projeto.

## Hardware e modelos atuais

| Item | Configuração |
|---|---|
| Placa | ESP32-S3, módulo N16R8 |
| CPU | 240 MHz |
| Flash | 16 MB, QIO, 80 MHz |
| PSRAM | 8 MB, OPI, 80 MHz |
| Framework | ESP-IDF via PlatformIO |
| Entrada | `int8 [1,3,224,224]`, NCHW, RGB |
| Payload HTTP | 150.528 bytes |
| Saída | `int8 [1,8]`, classificação em oito classes |

| Aplicação | Arquivo em data/ | Caminho no dispositivo | Arena configurada |
|---|---|---|---|
| YuNet | `model_int8.tflite` | `/littlefs/model_int8.tflite` | 4 MiB |
| MobileNetV2 alpha1 | `model_int8_avgpool.tflite` | `/littlefs/model_int8_avgpool.tflite` | 5 MiB |
| MobileNetV2 alpha0.5 | `model_int8_avgpool.tflite` | `/littlefs/model_int8_avgpool.tflite` | 5 MiB |
| MobileNetV2 scratch alpha1 e alpha0.5 | `model_int8_avgpool.tflite` | `/littlefs/model_int8_avgpool.tflite` | 5 MiB |

`platformio.ini`, `sdkconfig.defaults` e `sdkconfig.esp32s3` definem a configuração. O nome da placa base do PlatformIO pode aparecer como N8 sem PSRAM no cabeçalho do build; confira as configurações efetivas de flash e PSRAM e os logs de inicialização.

`partitions.csv` reserva `0x600000` bytes para a aplicação e `0x9E0000` bytes para LittleFS. Alterar o tamanho do modelo pode exigir rever essa divisão e a disponibilidade de PSRAM.

O firmware lê o modelo do LittleFS, obtém sua estrutura com `tflite::GetModel()`, registra os operadores, aloca os tensores e executa `MicroInterpreter::Invoke()`.

O resolver das aplicações tem capacidade para 11 operadores. No YuNet, registra Transpose, Pad, Conv2D, DepthwiseConv2D, MaxPool2D, ResizeNearestNeighbor, Add, Sum, Mul, Concatenation e FullyConnected. No MobileNetV2, AveragePool2D substitui Sum; os demais registros foram preservados, embora alguns não sejam usados pelo grafo.

Configuração verificada para os kernels otimizados:

```text
CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN=y
CONFIG_NN_OPTIMIZED=y
CONFIG_IDF_TARGET_ESP32S3=y
```

As implementações específicas do S3 incluem assembly e instruções vetoriais. O caminho utilizado depende do operador e de suas dimensões, tipos e parâmetros; nem toda camada necessariamente executa um kernel otimizado. A confirmação por camada exige instrumentação no dispositivo.

## Compilar e gravar

Em um novo clone, copie o modelo de configuração a partir da raiz do repositório:

```bash
cp firmware/wifi_credentials.example.h firmware/wifi_credentials.local.h
```

Preencha `ssid` e `password` em `firmware/wifi_credentials.local.h`. Os cinco projetos usam esse arquivo local, que é ignorado pelo Git. Abra a aplicação desejada como projeto PlatformIO no VS Code.

Os comandos abaixo são executados dentro da aplicação escolhida:

```bash
pio run -e esp32s3
pio run -e esp32s3 -t upload
pio run -e esp32s3 -t uploadfs
pio device monitor -b 115200
```

`upload` grava o firmware; `uploadfs` grava os arquivos de `data/` no LittleFS. Na primeira instalação e sempre que o modelo mudar, grave também o filesystem. Se precisar escolher a porta, acrescente `--upload-port /dev/ttyUSB0` aos comandos de upload.

O boot deve montar o LittleFS, carregar o modelo, inicializar o interpretador e informar o IP. Uma falha ao abrir o caminho configurado em `kModelPath` indica que o arquivo esperado não está disponível na partição montada.

Os diretórios `.pio` e `managed_components` contêm artefatos gerados. Não reutilize `.pio` copiado de outro projeto: ele pode guardar caminhos absolutos antigos. Se houver cache incompatível após mover o projeto ou atualizar dependências, remova `.pio/build` deste projeto e recompile.

## Inferência e avaliação pelo PC

O PC prepara o tensor e envia a entrada já quantizada. O ESP32 não recebe JPEG nesse fluxo.

| Endpoint | Função |
|---|---|
| `GET /status` | Estado do modelo e contrato de entrada. |
| `POST /predict_bin` | Inferência com corpo binário `int8` e `Content-Type: application/octet-stream`. |

Pré-processamento atual em `teste_inferencia.py`: RGB, resize bilinear para 224x224, divisão por 255, normalização com média `(0.485, 0.456, 0.406)` e desvio `(0.229, 0.224, 0.225)`, transposição para NCHW e quantização usando escala e zero-point de `input_info.json`.

Prepare um ambiente Python e instale as dependências usadas pelos scripts:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install numpy requests Pillow matplotlib tqdm ai-edge-litert
uv pip check
python python/get_input_info.py
```

`get_input_info.py` lê o modelo de `data/` e regenera os metadados. Atualmente valida especificamente a entrada `[1,3,224,224]`. Regenere o JSON quando trocar o modelo, mesmo se as dimensões forem iguais: a quantização pode mudar.

Teste de uma imagem, selecionada por `IMAGE_INDEX` em `teste_inferencia.py`:

```bash
ESP32_IP=192.168.3.22 python python/teste_inferencia.py
```

Avaliação de todas as imagens de teste:

```bash
ESP32_IP=192.168.3.22 python python/evaluate_esp_model.py
```

O avaliador encontra os arquivos `*.jpg` nas oito pastas definidas em `CLASS_NAMES`, seleciona todos uma vez, embaralha com `SEED=42` e apresenta uma barra `tqdm`. A execução é sequencial. A ordem de `CLASS_NAMES` deve corresponder à ordem das saídas usada no treinamento.

O caminho padrão do dataset parte de `BASE_DIR.parents[2]` em `teste_inferencia.py`, que aponta para a raiz de `ubiquos_tomato_village`. Para outro dataset, configure sua pasta de teste:

```bash
DATASET_DIR="/caminho/do/dataset/test" ESP32_IP=192.168.3.22 python python/evaluate_esp_model.py
```

Variáveis disponíveis: `ESP32_IP`, `DATASET_DIR`, `INPUT_INFO_PATH`, `OUTPUT_PLOT_PATH` no teste individual e `EVALUATION_RESULTS_DIR` no avaliador.

### Arquivos de resultado

Cada avaliação cria `python/results_esp32/AAAAMMDD_HHMMSS/`, ou uma pasta sob `EVALUATION_RESULTS_DIR`:

| Arquivo | Conteúdo |
|---|---|
| `predictions.csv` | Resultado por imagem, classe real/predita, acerto, scores, hashes, tempos, memória e erros. |
| `confusion_matrix.csv` | Linhas representam a classe real; colunas, a classe predita. |
| `confusion_matrix.png` | Visualização da matriz de confusão. |
| `summary.json` | Sucessos, falhas, acurácia, tempos médios e P50/P95/P99, datas e status inicial do ESP32. |

A acurácia e a matriz consideram somente inferências bem-sucedidas. Falhas de requisição ou validação são registradas no CSV e contabilizadas no resumo. Recall e F1 por classe não são exportados pelo script atual; podem ser calculados da matriz.

O avaliador valida o hash FNV-1a da entrada retornado pelo dispositivo. O hash de saída é registrado, mas não comparado automaticamente ao PC. `inferencia_pc.py` executa o TFLite localmente; compare o mesmo modelo e a mesma imagem para investigar diferenças numéricas.

### Tempos e execução de referência

Resultados históricos do YuNet em `yunet_esp32s3/python/results_esp32/20260929_195442/`, anteriores à instrumentação OP-TRACE:

| Métrica | Resultado |
|---|---|
| Imagens | 461 |
| Sucessos / falhas | 461 / 0 |
| Acertos / erros | 402 / 59 |
| Acurácia | 87,20% |
| F1 macro calculado da matriz | 85,71% |
| Inferência média / P99 | 737,12 / 740,37 ms |
| Processamento médio no ESP32 | 753,30 ms |
| Requisição média no PC | 1.011,06 ms |
| Duração da avaliação | 8 min 7 s |

`inference_ms` mede a execução do modelo; `total_processing_ms` inclui cópia e pós-processamento no dispositivo; `request_ms` mede a requisição HTTP completa no PC. A diferença média de 257,76 ms entre requisição e processamento inclui transferência e demais custos HTTP. O payload tem 150.528 bytes e os logs mostraram recepção de aproximadamente 195–223 ms.

Na execução, a PSRAM livre permaneceu em 3.896.648 bytes e a memória interna livre variou entre 250.283 e 250.415 bytes, sem evidência de vazamento. A inferência variou entre 735,35 e 742,02 ms.

O menor recall foi `Spotted Wilt Virus` (77,36%). A maior confusão isolada foi `Leaf Miner → Healthy`, com oito imagens. `Pottassium Deficiency` teve somente oito amostras; seus resultados têm suporte limitado. Esses números descrevem este modelo, dataset e configuração, não outros modelos nem garantem equivalência com a inferência no PC.

## Compatibilidade e diagnóstico

O MobileNetV2 original tinha pesos em `Buffer.offset/size` e pooling global convertido em `SUM`. A reexportação substituiu o pooling por `AVERAGE_POOL_2D` e incorporou os pesos em `Buffer.data`. O modelo corrigido foi aprovado nos dois resolvers do PC e avaliado no ESP32 em 01/10/2026. Veja [MobileNetV2](README_mobilenet.md) e [reexportação](README_reexportacao.md).

Na avaliação `mobilenet_esp32s3_alpha1/python/results_esp32/20261001_115908/`, foram enviados 461 exemplos: 454 respostas válidas, sete timeouts e 417 acertos. A acurácia nas respostas válidas foi 91,85%; considerando todas as imagens enviadas, houve 90,46% de acertos. A inferência média foi 2.500,66 ms e a requisição média, 3.789,74 ms. A PSRAM livre ficou constante em 173.384 bytes.

Uma comparação adicional sobre as mesmas 454 entradas confirmou todos os hashes de entrada, 418 acertos no PC e três discordâncias de classe PC–ESP32. Todos os hashes de saída diferiram, portanto não há equivalência byte a byte. As métricas detalhadas e as ressalvas da comparação com YuNet estão no guia MobileNetV2.

Na avaliação da alpha0.5 em `mobilenet_esp32s3_alpha05/python/results_esp32/20261001_130328/`, houve 461 respostas válidas, nenhuma falha e 336 acertos (72,89%). A inferência média foi 1.121,98 ms, a requisição média foi 1.376,46 ms e a PSRAM livre permaneceu em 1.910.088 bytes. A alpha0.5 foi mais rápida que a alpha1, mas teve menor acurácia; frente ao YuNet, apresentou menor acurácia e maior latência. Os detalhes por classe e a comparação com os resolvers do PC estão no [guia MobileNetV2](README_mobilenet.md).

A scratch alpha1 foi avaliada em `scratch_mobilenet_esp32s3_alpha1/python/results_esp32/20261002_102050/`: 399/461 acertos (86,55%), nenhuma falha, inferência média de 2.512,21 ms, requisição média de 3.604,73 ms e PSRAM livre constante em 173.384 bytes. Nos CSVs das 454 entradas compartilhadas com a alpha1 anterior, a scratch teve 393 acertos contra 417. A matriz por classe, as limitações e a comparação com YuNet estão no [guia MobileNetV2](README_mobilenet.md).

A scratch alpha0.5 foi avaliada em `scratch_mobilenet_esp32s3_alpha05/python/results_esp32/20261002_110629/`: 383/461 acertos (83,08%), nenhuma falha, inferência média de 1.125,49 ms, requisição média de 1.390,99 ms e PSRAM livre constante em 1.910.088 bytes. Com hashes de entrada iguais nas 461 imagens, teve 47 acertos a mais que a alpha0.5 anterior e 16 a menos que a scratch alpha1. Foi 55,20% mais rápida na inferência que a scratch alpha1. A matriz por classe e as limitações estão no [guia MobileNetV2](README_mobilenet.md).

O YuNet não tem buffers externos, mas diverge entre os resolvers do PC: para as imagens 82 e 250, `BUILTIN_REF` produziu os mesmos logits. No trace da imagem 250, a primeira divergência apareceu em Conv2D, operador 3, antes do Sum. Isso não prova que essa primeira diferença seja a causa da degeneração; pequenas diferenças de arredondamento também alteram hashes. Não aplique automaticamente o wrapper do MobileNetV2 ao YuNet.

O OP-TRACE está disponível no runtime compartilhado, desabilitado por padrão e preservado no patch em `docs/`. Ele registra as saídas após cada operador e permite comparar PC e ESP32. Consulte o [guia de debug](README_debug.md).

## Adaptar para outro modelo ou tarefa

1. Crie uma aplicação irmã de `yunet_esp32s3`, reutilizando os dois submódulos pelo CMake. Mantenha os commits conhecidos até validar uma atualização das bibliotecas.
2. Exporte um `.tflite` com operadores suportados pelo TFLite Micro. Confirme tipos, dimensões, layout, quantização e ordem das classes. Não basta o arquivo executar no interpretador do PC.
3. Coloque o modelo em `data/`; atualize `kModelPath` se mudar o nome e grave o LittleFS.
4. Em `ModelContext`, adapte dimensões da entrada, canais, classes e arena. Ajuste o `MicroMutableOpResolver` e sua capacidade aos operadores efetivamente presentes no modelo.
5. Atualize as validações, o formato de entrada e o pós-processamento do firmware. Há dimensões fixas também nas respostas HTTP, incluindo `[1,3,224,224]`; mudar somente `ModelContext` não é suficiente.
6. Adapte `get_input_info.py`, `teste_inferencia.py` e `inferencia_pc.py`: contratos fixos, normalização, layout, quantização, nomes e ordem das classes. Regenere `input_info.json`.
7. Ajuste o carregamento do dataset e o avaliador ao novo formato. Detecção, segmentação e regressão exigem pós-processamento e métricas próprios; a matriz atual mede classificação de uma classe por imagem.
8. Verifique carregamento e alocação no dispositivo, teste a mesma entrada no PC e no ESP32 e depois avalie o conjunto completo. Compare acurácia, falhas, latência e memória.

Para Arduino Nano 33 BLE Sense, o runtime TFLite Micro pode ser reutilizado com uma integração para Arduino/Arm. Os componentes `esp-tflite-micro` e ESP-NN pertencem à integração Espressif; essa aplicação e sua configuração de PSRAM não são portáveis diretamente para aquela placa.

## Alterações realizadas nesta etapa

- Clones de ESP-NN, esp-tflite-micro e TFLite Micro original registrados como submódulos.
- Dependências da aplicação conectadas a `git_common_components`, substituindo a referência antiga a `tflite-lib`.
- Registro dos componentes feito com `list(APPEND EXTRA_COMPONENT_DIRS ...)` para preservar o `src` fornecido pelo PlatformIO.
- Caminho padrão do dataset corrigido de `BASE_DIR.parents[3]` para `BASE_DIR.parents[2]`.
- Avaliação ampliada de 100 imagens para todo o conjunto de teste, com progresso via `tqdm`.
- Configuração dos kernels ESP-NN para S3 e resultados da avaliação conferidos.

Os guias específicos e os comandos atuais estão em `docs/`. O conteúdo em `butterflies_antigo` é referência histórica; suas instruções e métricas não descrevem automaticamente estas aplicações.
