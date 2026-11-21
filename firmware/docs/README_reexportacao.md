# Reexportação do MobileNetV2 INT8 para ESP32-S3

[Índice](README.md) · [MobileNetV2](README_mobilenet.md)

Os arquivos mencionados nesta página ficam em `mobilenet_esp32s3_alpha1/training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8/`, a partir de `esp32-s3-git-espnn-tflite`. Execute os comandos nessa pasta, não em `docs/`.

O script `reexportar.py` carrega o `best__model.pt` da pasta de reexportação, preserva os pesos treinados e gera novos modelos somente em `outputs/<data_hora>/`. Não altera o checkpoint, os modelos de outras pastas, o firmware ou os submódulos. Não realiza treinamento.

## Problemas identificados

O mesmo procedimento está disponível para o alpha 0,5 em `mobilenet_esp32s3_alpha05/training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8/`. Essa pasta contém uma cópia do seu checkpoint `best__model.pt` e um script adaptado para o backbone `mobilenetv2_050.lamb_in1k`. Os artefatos são gerados no `outputs/` dessa pasta; não reutilize o checkpoint ou os modelos do alpha1.

Para preparar e executar o alpha 0,5, a partir da raiz de `ubiquos_tomato_village`:

```bash
cd esp32-s3-git-espnn-tflite/mobilenet_esp32s3_alpha05/training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch torchvision timm litert-torch ai-edge-litert ai-edge-quantizer flatbuffers numpy Pillow tqdm
uv pip check
python reexportar.py
```

O INT8 original do alpha 0,5 contém 55 buffers externos (696.320 bytes) e `SUM`. As imagens 82 e 250 produziram logits idênticos em `BUILTIN_REF`; a versão corrigida ainda precisa ser gerada e validada. As métricas e os comandos de integração do alpha1 nas seções abaixo pertencem àquela aplicação.

## MobileNetV2 alpha1 treinada do zero

O procedimento também está disponível em `scratch_mobilenet_esp32s3_alpha1/training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8/`. A pasta contém `reexportar.py` e uma cópia do checkpoint `best__model.pt` treinado do zero nesta aplicação. O backbone é `mobilenetv2_100`; o script é o mesmo da alpha1, mas carrega esse checkpoint local.

O INT8 original tem 2.797.536 bytes, 71 buffers externos com 2.253.024 bytes e pooling `SUM`. A reexportação incorpora os pesos e substitui o pooling por `AVERAGE_POOL_2D`. A presença de `SUM` foi identificada pela inspeção do grafo; não foi executada inferência para confirmar degeneração nesta versão.

A partir da raiz de `ubiquos_tomato_village`:

```bash
cd esp32-s3-git-espnn-tflite/scratch_mobilenet_esp32s3_alpha1/training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch torchvision timm litert-torch ai-edge-litert ai-edge-quantizer flatbuffers numpy Pillow tqdm
uv pip check
python reexportar.py
```

A execução e a validação ficam a cargo do usuário. Os novos artefatos serão gerados somente em `outputs/<data_hora>/` desta pasta. Após aprovação, copie seu `model_int8_avgpool.tflite` para `scratch_mobilenet_esp32s3_alpha1/data/` e seus metadados para `python/input_info.json` dessa aplicação. O firmware já espera `/littlefs/model_int8_avgpool.tflite` e registra `AddAveragePool2D()`. Grave firmware e LittleFS dentro da aplicação scratch. Não use os artefatos da alpha1 com outros pesos.

A scratch alpha1 foi reexportada em `outputs/20261002_124511_689419/`: 2.794.024 bytes, 71 buffers incorporados, 401/461 acertos no PC padrão e 399/461 no referência. A avaliação no ESP32 em `20261002_102050` registrou 399/461 acertos, nenhuma falha e inferência média de 2.512,21 ms. Veja os detalhes no [guia MobileNetV2](README_mobilenet.md).

## MobileNetV2 alpha0.5 treinada do zero

Pasta: `scratch_mobilenet_esp32s3_alpha05/training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8/`. Contém `reexportar.py` e uma cópia do checkpoint `best__model.pt` desta aplicação. O script usa `mobilenetv2_050`, com `pretrained=False` e `drop_rate=0.3`, conforme o notebook de treinamento desta versão.

O INT8 original tem 1.059.280 bytes, 55 buffers externos com 696.320 bytes e pooling `SUM`. A reexportação substitui o pooling por `AVERAGE_POOL_2D` e incorpora os pesos em `Buffer.data`. A inspeção foi estrutural; não foi executada inferência para confirmar degeneração nesta versão.

A partir da raiz de `ubiquos_tomato_village`:

```bash
cd esp32-s3-git-espnn-tflite/scratch_mobilenet_esp32s3_alpha05/training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch torchvision timm litert-torch ai-edge-litert ai-edge-quantizer flatbuffers numpy Pillow tqdm
uv pip check
python reexportar.py
```

A execução e os testes ficam a cargo do usuário. O script gera somente `outputs/<data_hora>/` nesta pasta, incluindo `model_int8_avgpool.tflite`, `validation.json` e `input_info.json`. Após aprovação, copie o modelo para `scratch_mobilenet_esp32s3_alpha05/data/` e os metadados para `python/input_info.json` dessa aplicação. O firmware já espera `/littlefs/model_int8_avgpool.tflite` e registra `AddAveragePool2D()`. Grave firmware e LittleFS dentro da aplicação scratch alpha0.5. Os pesos desta versão são independentes dos das outras aplicações.

A scratch alpha0.5 foi reexportada em `outputs/20261002_124728_816060/`: 1.055.952 bytes, 55 buffers incorporados, 385/461 acertos no PC padrão e 383/461 no referência. A avaliação no ESP32 em `20261002_110629` registrou 383/461 acertos (83,08%), nenhuma falha e inferência média de 1.125,49 ms. Veja os resultados por classe e a comparação com as demais versões no [guia MobileNetV2](README_mobilenet.md).

### Pooling global convertido em SUM

O MobileNetV2 original exportado contém `SUM`. Na avaliação local, os kernels padrão sem delegates classificaram corretamente as imagens 82 e 250, mas `BUILTIN_REF` produziu os mesmos logits nas duas imagens. Executar corretamente com XNNPACK ou outros kernels do PC não comprova compatibilidade numérica com o TFLite Micro.

A solução reproduz o procedimento de `butterflies_antigo`: um wrapper de exportação aplica `avg_pool2d` com kernel 7x7 sobre o mapa final 7x7, seguido de flatten e do classificador original. O modelo está em modo de avaliação, portanto o dropout original é inativo. Antes da conversão, o script compara numericamente o wrapper com o modelo PyTorch original em uma imagem de teste.

O grafo exportado deve conter `AVERAGE_POOL_2D` e não pode conter `SUM`. O script também rejeita operadores fora do conjunto previsto.

### Pesos em Buffer.offset/size

O modelo original contém 71 buffers externos, com 2.253.024 bytes e `Buffer.data` vazio. O runtime embarcado utilizado acessa `Buffer.data`, portanto esses pesos não são reconhecidos como dados constantes. Isso explica a saturação e pode aumentar a arena necessária ao alocar esses tensores como memória de trabalho.

Após quantizar, o script copia os bytes indicados por `Buffer.offset/size` para `Buffer.data` e zera os campos externos. Confere os limites de cada intervalo e compara os dados de todos os buffers antes e depois da conversão. Os valores dos pesos são preservados.

## Execução

Use um ambiente Python compatível com PyTorch, torchvision, timm, LiteRT Torch e AI Edge Quantizer, preferencialmente o ambiente utilizado na exportação original. As versões não são fixadas aqui; incompatibilidades entre essas ferramentas podem exigir usar as versões do ambiente de treinamento.

Para criar o ambiente local com `uv`, instalar as dependências e executar:

```bash
cd /home/vini_muchulski/dados/experimentos_esp32/ubiquos_tomate/ubiquos_tomato_village/esp32-s3-git-espnn-tflite/mobilenet_esp32s3_alpha1/training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8

uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install torch torchvision timm litert-torch ai-edge-litert ai-edge-quantizer flatbuffers numpy Pillow tqdm

uv pip check

python reexportar.py
```

`uv pip check` verifica a consistência das dependências instaladas; não valida a exportação do modelo. Em execuções posteriores, reative o ambiente com `source .venv/bin/activate`.

O dataset padrão é o Tomato Village dentro da raiz de `ubiquos_tomato_village`, com subpastas `train` e `test`. Para executar em outra máquina:

```bash
python reexportar.py --dataset "/caminho/Variant-a(Multiclass Classification)"
```

O checkpoint é sempre `best__model.pt` da pasta de reexportação. A calibração usa até 100 imagens do treino, sem augmentation, com semente 42. O teste usa todas as imagens e apresenta progresso com `tqdm`.

## Artefatos e validação

| Arquivo em outputs/data_hora | Uso |
|---|---|
| `model_fp32_avgpool.tflite` | Exportação FP32 com pooling explícito. |
| `model_int8_external.tflite` | Quantização intermediária; não usar no firmware. |
| `model_int8_avgpool.tflite` | INT8 com pesos incorporados em Buffer.data. |
| `validation.json` | Operadores, classes, equivalência PyTorch, buffers convertidos e métricas dos dois resolvers. |
| `input_info.json` | Metadados da entrada do novo INT8; gerado após aprovação no PC. |

Cada execução cria uma pasta nova e não sobrescreve os artefatos anteriores. Uma execução que falha pode deixar arquivos intermediários; sua existência não indica aprovação.

O script valida entrada `int8 [1,3,224,224]`, saída `int8 [1,8]`, escalas positivas e classes na ordem do firmware. Avalia o modelo final com `BUILTIN_WITHOUT_DEFAULT_DELEGATES` e `BUILTIN_REF`, rejeita logits constantes no conjunto de teste e exige diferença absoluta de acurácia de no máximo um ponto percentual. Consulte também a acurácia absoluta e a quantidade de predições divergentes em `validation.json`; a aprovação não exige logits idênticos entre resolvers nem certifica a execução no ESP32.

O pré-processamento mantém resize bilinear 224x224, RGB e normalização ImageNet do treinamento. O script usa os transforms do torchvision para avaliação; o cliente embarcado usa PIL e NumPy. Compare os hashes da entrada ao validar a mesma imagem no PC e no ESP32.

## Integração posterior no firmware

Estas etapas são manuais; o script não as executa:

1. Confira `approved_on_pc: true` e as acurácias em `validation.json`.
2. Copie somente o modelo final para `mobilenet_esp32s3_alpha1/data/model_int8_avgpool.tflite`, nome atualmente esperado pelo firmware.
3. Copie o novo `input_info.json` para `mobilenet_esp32s3_alpha1/python/input_info.json` ou regenere com `python/get_input_info.py` após copiar o modelo.
4. Confirme `op_resolver.AddAveragePool2D()` em `src/main.cpp`; essa substituição já foi aplicada nesta aplicação. Se os operadores reportados incluírem `RESHAPE`, registre também `AddReshape()` e confira a capacidade do resolver.
5. Mantenha ESP-NN habilitada. Recompile, grave o firmware e grave o LittleFS com o novo modelo.
6. Confira `AllocateTensors` e a arena usada, compare imagens conhecidas e execute a avaliação completa no ESP32.

A arena de 5 MiB foi escolhida para o artefato anterior. O modelo com pesos incorporados pode exigir menos memória; determine o consumo real no dispositivo antes de alterar esse limite.

## Artefato validado em 01/10/2026

A execução `outputs/20261001_143152_063478/` gerou um modelo final de 2.794.024 bytes, sem buffers externos e com os dados dos buffers preservados. A equivalência do wrapper PyTorch teve erro absoluto máximo de `4,76837158203125e-07`.

No teste completo: 425/461 acertos (92,19%) nos kernels padrão sem delegates e 424/461 (91,97%) em `BUILTIN_REF`, diferença de 0,22 ponto percentual e três predições divergentes. As imagens 82 e 250 foram classificadas corretamente por ambos. A validação no PC não substitui a avaliação no ESP32.

## Artefato alpha0.5 validado em 01/10/2026

A execução da alpha0.5 em `outputs/20261001_153647_622727/` gerou `model_int8_avgpool.tflite` com 1.055.952 bytes. Os 55 buffers externos foram incorporados sem alteração dos dados; o arquivo final não contém buffers externos nem `SUM`, e contém `AVERAGE_POOL_2D`. O erro absoluto máximo de equivalência do wrapper PyTorch foi `5,960464477539063e-08`.

O relatório registra 337/461 acertos (73,10%) nos kernels padrão sem delegates e 336/461 (72,89%) em `BUILTIN_REF`, com 12 discordâncias de classe e diferença de 0,22 ponto percentual. A aprovação verifica a proximidade entre resolvers e a ausência de logits constantes; não comprova preservação da acurácia do checkpoint PyTorch.

A avaliação no ESP32 em `mobilenet_esp32s3_alpha05/python/results_esp32/20261001_130328/` registrou 336/461 acertos (72,89%), nenhuma falha e inferência média de 1.121,98 ms. Consulte as métricas por classe, tempos e memória no [guia MobileNetV2](README_mobilenet.md).
