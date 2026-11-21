# OP-TRACE do TFLite Micro

[Índice](README.md) · [Guia principal](README_principal.md)

A instrumentação está em `git_common_components/esp-tflite-micro`, no laço de execução de `MicroInterpreterGraph::InvokeSubgraph`. Não depende da arquitetura do modelo nem altera a seleção dos kernels ESP-NN.

## Ativar

Dentro da aplicação PlatformIO, execute:

```bash
pio run -e esp32s3 -t menuconfig
```

Em `ESP TFLite-Micro`, habilite `Trace operator output tensors`, salve e recompile/grave o firmware. A opção correspondente em `sdkconfig.esp32s3` é:

```text
CONFIG_ESP_TFLITE_MICRO_OPTRACE=y
```

O trace é desabilitado por padrão. Para desativar, desmarque a opção no menuconfig e recompile. Não use os tempos medidos com trace para avaliar desempenho.

## Capturar e comparar

Dentro de `yunet_esp32s3` ou `mobilenet_esp32s3_alpha1`, crie a pasta de logs e inicie o monitor:

```bash
mkdir -p python/results
pio device monitor -b 115200 | tee python/results/optrace_esp32.log
```

Em outro terminal, envie exatamente uma imagem com `python python/teste_inferencia.py`. Pare o monitor depois de receber o resultado. O log de comparação deve conter somente uma inferência. Se reutilizar o arquivo, o `tee` acima substitui seu conteúdo.

Use o mesmo modelo, `input_info.json` e `IMAGE_INDEX` no PC e no dispositivo. Confirme o hash da entrada no resultado antes de interpretar divergências.

```bash
python python/trace_operadores_pc.py python/results/optrace_esp32.log
```

Os scripts das duas aplicações executam os resolvers `BUILTIN_WITHOUT_DEFAULT_DELEGATES` e `BUILTIN_REF` preservando os tensores intermediários. Salvam os traces em `python/results_pc/`, comparam os resolvers entre si e com o ESP32, e mostram a primeira diferença de hash e de estatísticas. Sem argumento, comparam somente os dois resolvers no PC. O código de saída 1 indica divergência.

## Formato

```text
OPTRACE op=0 name=TRANSPOSE tensor=1 bytes=150528 hash=12345678 min=-128 max=127 sum=123 subgraph=0
```

Cada linha representa um tensor de saída de um operador executado com sucesso. O hash FNV-1a é calculado sobre os bytes, imediatamente após o operador, antes da reutilização da memória pela arena. As estatísticas são calculadas para INT8, UINT8, INT16 e INT32, com acumulador de 64 bits. Para outros tipos, apenas bytes e hash são informativos; min, max e sum ficam em zero.

A instrumentação identifica o subgrafo no log. Os scripts atuais comparam somente o subgrafo principal e rejeitam registros de outros subgrafos. Também rejeitam logs incompletos ou com inferências repetidas.

Diferenças de hash podem representar pequenas diferenças de arredondamento; não demonstram por si só uma falha grave. Use a primeira divergência, min/max/sum, a saída final e as métricas para distinguir arredondamento de saturação ou pesos inválidos. O trace não grava os valores completos dos tensores nem hashes dos pesos.

## Preservar a instrumentação

As alterações no runtime são locais ao submódulo e não estão incluídas automaticamente em um commit do repositório principal. `esp-tflite-micro-optrace.patch`, nesta pasta, preserva essas alterações sem exigir commits ou modificações no GitHub.

Em um checkout com o submódulo original, execute a partir de `esp32-s3-git-espnn-tflite`:

```bash
git -C git_common_components/esp-tflite-micro apply --check ../../docs/esp-tflite-micro-optrace.patch
git -C git_common_components/esp-tflite-micro apply ../../docs/esp-tflite-micro-optrace.patch
```

Não reaplique sobre um submódulo já instrumentado. Se atualizar o commit do runtime, confira se o patch continua aplicável e revise o laço de execução antes de adaptar.
