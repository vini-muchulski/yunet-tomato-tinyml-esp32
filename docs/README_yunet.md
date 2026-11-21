# Classificação YuNet no ESP32-S3

[Índice](README.md) · [Guia principal](README_principal.md) · [OP-TRACE](README_debug.md)

Projeto ESP-IDF/PlatformIO para executar o classificador YuNet INT8 treinado
no dataset Tomato Village usando TensorFlow Lite Micro e ESP-NN.

## Estrutura

```text
esp32-s3-git-espnn-tflite/
├── git_common_components/
│   ├── esp-nn/
│   ├── esp-tflite-micro/
│   └── tflite-micro/
├── yunet_esp32s3/
│   ├── data/model_int8.tflite
│   ├── python/
│   ├── src/main.cpp
│   ├── CMakeLists.txt
│   ├── partitions.csv
│   ├── platformio.ini
│   ├── sdkconfig.defaults
│   └── sdkconfig.esp32s3
└── docs/
```

O CMake registra `esp-tflite-micro`, `esp-nn` e `esp_littlefs` por `list(APPEND EXTRA_COMPONENT_DIRS ...)`, com caminhos para `../git_common_components`. Não há symlink `components`. O TFLite original não é compilado por esta aplicação.

## Modelo

O arquivo usado pelo firmware é `data/model_int8.tflite`.

| Item | Valor |
|---|---|
| Tamanho | 140008 bytes |
| Entrada | `int8 [1,3,224,224]`, NCHW |
| Escala da entrada | `0.01454248744994402` |
| Zero-point da entrada | `-5` |
| Saída | `int8 [1,8]` |
| Escala da saída | `0.07407154887914658` |
| Zero-point da saída | `-41` |
| Corpo HTTP | 150528 bytes |

A saída contém logits, sem Softmax. O campo `confidence` é o maior logit
desquantizado e serve para comparar a classe vencedora; não é uma probabilidade.

Operadores do grafo:

`TRANSPOSE`, `PAD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `MAX_POOL_2D`,
`RESIZE_NEAREST_NEIGHBOR`, `ADD`, `SUM`, `MUL`, `CONCATENATION` e
`FULLY_CONNECTED`.

## Classes

A ordem deve ser mantida no firmware, no cliente e na avaliação:

| Índice | Classe |
|---:|---|
| 0 | Early_blight |
| 1 | Healthy |
| 2 | Late_blight |
| 3 | Leaf Miner |
| 4 | Magnesium Deficiency |
| 5 | Nitrogen Deficiency |
| 6 | Pottassium Deficiency |
| 7 | Spotted Wilt Virus |

## Pré-processamento

O treinamento e os scripts de inferência usam o mesmo pipeline:

1. Abrir a imagem e converter para RGB.
2. Redimensionar com interpolação bilinear para `224x224`.
3. Converter pixels para float no intervalo `[0,1]`.
4. Aplicar mean `(0.485, 0.456, 0.406)`.
5. Aplicar std `(0.229, 0.224, 0.225)`.
6. Transpor HWC para CHW.
7. Quantizar com `round(valor / scale + zero_point)` e limitar a `[-128,127]`.

O ESP32 não abre JPEG, redimensiona nem normaliza. O endpoint recebe o tensor
INT8 pronto e faz uma cópia direta para o tensor de entrada.

A avaliação embarcada de 29/09/2026 teve 402 acertos em 461 imagens (87,20%), sem falhas. Os resultados estão em `yunet_esp32s3/python/results_esp32/20260929_195442/`. A inferência média foi 737,12 ms e a requisição média, 1.011,06 ms. São métricas históricas desse firmware, sem OP-TRACE.

## Integração TFLite Micro

O resolver em `src/main.cpp` deve registrar todos os operadores presentes no modelo. Sua capacidade deve ser suficiente para o número registrado:

```cpp
tflite::MicroMutableOpResolver<11> op_resolver;
```

Conv2D, DepthwiseConv2D, MaxPool2D, Add, Mul e FullyConnected são registrados
pelas implementações ESP-NN. Transpose, Pad, ResizeNearestNeighbor, Sum e
Concatenation usam os kernels TFLite Micro disponíveis no componente local.

A arena está configurada com 4 MiB. O modelo e a arena são alocados
preferencialmente na PSRAM.

## ESP-NN e ESP32-S3

O projeto usa:

- `CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN=y`, que habilita `-DESP_NN` no CMake do componente;
- `CONFIG_NN_OPTIMIZED=y`;
- kernels assembly específicos do ESP32-S3;
- CPU a 240 MHz;
- PSRAM OPI a 80 MHz;
- cache de dados configurado em 32 KiB.

Os componentes vêm dos submódulos `esp-nn` e `esp-tflite-micro`, ambos com versão declarada 1.4.1 nos commits de referência. Os commits estão registrados no [guia principal](README_principal.md). Não são as cópias antigas de `common_components_tomato_village_yunet`.

A atualização desses componentes deve ser feita separadamente, pois pode
alterar kernels, APIs, uso de arena e resultados numéricos.

O log do PlatformIO pode identificar a placa genérica como `N8 No PSRAM`. Isso
é o nome da definição da placa. A configuração deste projeto espera um módulo
com 16 MB de flash e 8 MB de PSRAM OPI. No boot, confirme:

```text
Found 8MB PSRAM device
```

## Configuração CMake e PlatformIO

Preserve `list(APPEND EXTRA_COMPONENT_DIRS ...)` antes de incluir o `project.cmake` do ESP-IDF. Substituir a lista com `set(...)` remove o `src` fornecido pelo PlatformIO e causa `Couldn't find the main target of the project!`.

A solução antiga baseada em symlink para evitar colisões de objetos não faz parte desta estrutura. Se o cache foi copiado de outro caminho, remova `.pio/build` dentro desta aplicação e recompile. Consulte o [guia principal](README_principal.md) para os comandos de build e gravação.

## LittleFS

O firmware carrega o modelo em:

```text
/littlefs/model_int8.tflite
```

Gravar somente o firmware não atualiza o arquivo do modelo. Sempre que
`data/model_int8.tflite` mudar, grave o filesystem:

```bash
pio run -t uploadfs --upload-port /dev/ttyUSB0
```

O diretório `data/` precisa conter diretamente o arquivo do modelo. Se o
boot informar `Falha ao abrir /littlefs/model_int8.tflite`, a partição gravada
está antiga ou foi criada sem esse arquivo.

## Interface HTTP

`GET /status` retorna o estado de inicialização e o contrato da entrada.

`POST /predict_bin` exige `Content-Type: application/octet-stream` e exatamente
150528 bytes de tensor INT8 NCHW.

A resposta de sucesso inclui:

- `predicted_class` e `confidence`;
- os oito valores de `scores`;
- `input_fnv1a` e `output_fnv1a`;
- tempos de recepção, cópia, inferência e pós-processamento;
- uso da arena e memória livre.

O primeiro diagnóstico de divergência deve comparar `input_fnv1a`. Se o hash da
entrada coincidir, compare `output_fnv1a`, logits e classe.

## Scripts Python

Os scripts devem ser executados na pasta do projeto ESP-IDF:

```bash
cd esp32-s3-git-espnn-tflite/yunet_esp32s3
```

Variáveis úteis:

- `ESP32_IP`: endereço da placa, padrão `192.168.3.22`;
- `DATASET_DIR`: diretório alternativo do teste;
- `MODEL_PATH`: modelo alternativo para a referência no PC;
- `INPUT_INFO_PATH`: metadados alternativos;
`IMAGE_INDEX` é uma constante em `python/teste_inferencia.py`, não uma variável de ambiente. Altere-a nesse arquivo para escolher outra imagem.

O dataset padrão é:

```text
../../dataset/tomato_village/tomato_village/Variant-a(Multiclass Classification)/test
```

O cliente de uma imagem envia uma amostra ao ESP32:

```bash
ESP32_IP=192.168.3.22 .venv/bin/python python/teste_inferencia.py
```

A referência local executa o mesmo modelo no LiteRT:

```bash
.venv/bin/python python/inferencia_pc.py
```

A validação local das imagens selecionadas usa:

```bash
.venv/bin/python python/sanity_check_host.py 57 0 100
```

`evaluate_esp_model.py` avalia todas as imagens `*.jpg` do teste, embaralhadas com semente 42, e mostra uma barra `tqdm`. Consulta `/status`, envia cada tensor ao ESP32 e salva cada execução
em `python/results_esp32/AAAAmmdd_HHMMSS/`:

```bash
ESP32_IP=192.168.3.22 .venv/bin/python python/evaluate_esp_model.py
```

Arquivos gerados:

```text
predictions.csv
summary.json
confusion_matrix.csv
confusion_matrix.png
```

O avaliador considera a imagem correta somente quando a resposta do ESP32 é
válida e o índice predito coincide com o rótulo do dataset. Falhas de rede, contrato ou hash ficam registradas no CSV. A acurácia e a matriz de confusão consideram somente inferências bem-sucedidas.

O script `trace_operadores_pc.py` é uma ferramenta de diagnóstico para comparar
saídas intermediárias com registros `OPTRACE` do firmware. Use-o somente após
confirmar que a entrada e o modelo são os mesmos.

## Fluxo recomendado para um novo modelo

1. Inspecionar o TFLite: shape, layout, dtype, quantização, saída e operadores.
2. Confirmar no código de treinamento o resize, a ordem dos canais e a fórmula de normalização.
3. Escolher a variante quantizada com melhor acurácia e compatibilidade com TFLite Micro.
4. Copiar o modelo para `data/` e atualizar `kModelPath` e `kModelName`.
5. Atualizar shape, tipo, número de classes, tamanho da arena e validações no firmware.
6. Registrar todos os operadores no `MicroMutableOpResolver`.
7. Adaptar o cliente Python e gerar novamente `input_info.json`.
8. Atualizar a ordem de classes em todos os scripts.
9. Validar no host usando o mesmo pré-processamento do treinamento.
10. Gravar o LittleFS e comparar o hash de entrada entre PC e ESP32.
11. Comparar saída, classe, memória e tempo de inferência na placa.
12. Só depois medir desempenho ou trocar ESP-NN/TFLite Micro.

## Limitações conhecidas

- A compilação, gravação e validação final na placa dependem do hardware do desenvolvedor.
- O modelo não tem buffers externos, mas `BUILTIN_REF` produziu logits idênticos para as imagens 82 e 250. Os kernels padrão sem delegates classificaram ambas corretamente.
- No OP-TRACE do PC para a imagem 250, a primeira divergência aparece em `CONV_2D`, operador 3, antes do `SUM`. A causa da degeneração não está isolada; não aplique diretamente o wrapper de pooling do MobileNetV2 ao YuNet.
- Use os dois resolvers no diagnóstico com `trace_operadores_pc.py`; a equivalência numérica com o ESP32 ainda precisa ser comprovada.
- O ESP32-S3 precisa ter PSRAM compatível com a configuração OPI do projeto.
- SSID e senha Wi-Fi estão atualmente em `src/main.cpp` e devem ser alterados antes da distribuição.
