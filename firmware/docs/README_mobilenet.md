# MobileNetV2 alpha1 e alpha0.5 no ESP32-S3

[Índice](README.md) · [Guia principal](README_principal.md) · [Reexportação](README_reexportacao.md) · [OP-TRACE](README_debug.md)

Aplicação: `mobilenet_esp32s3_alpha1`. O checkpoint treinado e os artefatos de exportação ficam em `training_code/tomato_village_dgx_MobileNetv2_alpha1/`, dentro da aplicação.

## Modelo e configuração atuais

| Item | Valor |
|---|---|
| Projeto CMake | `tomato_village_mobilenetv2_alpha1_esp32s3` |
| Identificação | `Tomato Village MobileNetV2 alpha1` |
| Modelo local | `data/model_int8_avgpool.tflite` |
| Caminho no dispositivo | `/littlefs/model_int8_avgpool.tflite` |
| Tamanho do arquivo validado | 2.794.024 bytes |
| Entrada | `int8 [1,3,224,224]`, NCHW, RGB |
| Quantização de entrada | Escala `0.01454248744994402`, zero-point `-5` |
| Saída | `int8 [1,8]` |
| Payload | 150.528 bytes |
| Arena configurada | 5 MiB, preferencialmente em PSRAM |
| ESP-NN | Habilitada |

O pré-processamento e a ordem das oito classes são os mesmos descritos no [guia YuNet](README_yunet.md). A saída contém scores desquantizados; `confidence` é o maior score, não uma probabilidade Softmax.

O modelo corrigido usa `TRANSPOSE`, `PAD`, `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `AVERAGE_POOL_2D` e `FULLY_CONNECTED`. O firmware já registra `AddAveragePool2D()` no lugar de `AddSum()`. O resolver mantém capacidade 11 e quatro registros extras do firmware original; não precisam ser removidos para executar este grafo.

O CMake registra `esp-tflite-micro`, `esp-nn` e `esp_littlefs` em `../git_common_components` por `list(APPEND EXTRA_COMPONENT_DIRS ...)`. O OP-TRACE compartilhado é opcional e não precisa estar ativado para inferência.

## Problemas corrigidos na exportação

O artefato original tinha dois problemas independentes:

1. Pooling global convertido em `SUM`, com divergência nos kernels de referência: imagens diferentes geravam logits idênticos no PC com `BUILTIN_REF`.
2. 71 buffers de pesos em `Buffer.offset/size`, sem `Buffer.data`, incompatíveis com a forma como o runtime embarcado acessa os pesos.

A reexportação preserva o checkpoint e substitui o pooling por `avg_pool2d` 7x7, equivalente para o mapa final 7x7, gerando `AVERAGE_POOL_2D`. Após quantizar INT8, incorpora os pesos em `Buffer.data` e confere que os bytes foram preservados. Os detalhes e comandos uv estão no [guia de reexportação](README_reexportacao.md).

## Validação no PC

Artefato de referência: `training_code/tomato_village_dgx_MobileNetv2_alpha1/reexportar_tflite_int8/outputs/20261001_143152_063478/model_int8_avgpool.tflite`.

| Resolver | Acertos | Acurácia |
|---|---:|---:|
| `BUILTIN_WITHOUT_DEFAULT_DELEGATES` | 425/461 | 92,19% |
| `BUILTIN_REF` | 424/461 | 91,97% |

Diferença de 0,22 ponto percentual e três predições divergentes. As imagens 82 e 250 foram classificadas corretamente em ambos. O arquivo final não possui buffers externos, contém `AVERAGE_POOL_2D` e não contém `SUM`. Os metadados locais foram regenerados com esse modelo.

Esses resultados são do PC. A versão corrigida também foi avaliada no ESP32, conforme a seção seguinte. Os logs de saturação em classe 0 pertenciam ao modelo anterior.

## Avaliação no ESP32 em 01/10/2026

Resultados em `mobilenet_esp32s3_alpha1/python/results_esp32/20261001_115908/`.

| Métrica | Resultado |
|---|---:|
| Imagens enviadas | 461 |
| Inferências válidas / falhas | 454 / 7 |
| Acertos / erros de classificação | 417 / 37 |
| Acurácia nas inferências válidas | 91,85% |
| Acertos sobre todas as imagens enviadas | 90,46% |
| F1 macro calculado da matriz | 91,30% |
| Inferência média / P99 | 2.500,66 / 2.504,10 ms |
| Processamento médio no dispositivo | 2.516,77 ms |
| Requisição média / P95 / P99 | 3.789,74 / 6.310,05 / 7.873,03 ms |
| Duração total | 29 min 45 s |

A matriz e a acurácia do resumo excluem as sete falhas. Contar uma falha como ausência de acerto resulta em 417/461, ou 90,46%; essa razão não é a acurácia de classificação exportada pelo avaliador.

| Classe | Acertos / inferências válidas | Recall |
|---|---:|---:|
| Early_blight | 47/48 | 97,92% |
| Healthy | 20/22 | 90,91% |
| Late_blight | 87/92 | 94,57% |
| Leaf Miner | 88/103 | 85,44% |
| Magnesium Deficiency | 89/92 | 96,74% |
| Nitrogen Deficiency | 36/37 | 97,30% |
| Pottassium Deficiency | 8/8 | 100,00% |
| Spotted Wilt Virus | 42/52 | 80,77% |

A maior confusão foi `Leaf Miner → Healthy`, com nove imagens. A precisão de `Healthy` foi 66,67%. O resultado de `Pottassium Deficiency` tem suporte de apenas oito imagens.

### Comparação das mesmas entradas no PC

Uma verificação adicional executou o modelo corrigido no PC com `BUILTIN_WITHOUT_DEFAULT_DELEGATES`, usando exatamente as 454 imagens com respostas válidas. Todos os hashes de entrada coincidiram com os registros do ESP32.

- PC: 418/454 acertos; ESP32: 417/454.
- Três discordâncias de classe, aproximadamente 0,66% das entradas comparadas.
- Os 454 hashes de saída diferiram: há proximidade nas classificações, mas não equivalência byte a byte.

| Índice da imagem | Classe real | Predição PC | Predição ESP32 |
|---:|---|---|---|
| 256 | Leaf Miner | Leaf Miner | Early_blight |
| 421 | Spotted Wilt Virus | Leaf Miner | Spotted Wilt Virus |
| 113 | Late_blight | Late_blight | Spotted Wilt Virus |

Essa comparação adicional foi feita durante a análise; não é executada automaticamente por `evaluate_esp_model.py` nem está gravada em seu `summary.json`.

### Tempos, falhas e memória

A inferência variou entre 2.497,80 e 2.507,15 ms. A requisição completa chegou a 9.552,04 ms, mostrando variação principalmente fora da execução do modelo. A diferença média entre requisição e processamento foi 1.272,96 ms, incluindo transferência e outros custos HTTP.

As falhas foram timeouts de transporte nos índices 286, 259, 278, 19, 49, 329 e 439, após aproximadamente 5–5,9 segundos. O CSV não determina sozinho a causa desses timeouts; não foram registradas como erros de `Invoke()`.

A PSRAM livre permaneceu em 173.384 bytes. A memória interna livre variou entre 250.183 e 250.287 bytes, sem queda progressiva observada. A margem de PSRAM é pequena.

Comparado à avaliação histórica do YuNet, a acurácia nas respostas válidas passou de 87,20% para 91,85%, e a inferência média ficou 3,39 vezes mais lenta. Os conjuntos válidos são diferentes devido aos sete timeouts no MobileNetV2, portanto essa diferença de acurácia não é uma comparação sobre exatamente as mesmas imagens.

## MobileNetV2 alpha0.5: avaliação em 01/10/2026

Aplicação: `mobilenet_esp32s3_alpha05`. Resultados em `mobilenet_esp32s3_alpha05/python/results_esp32/20261001_130328/`.

O modelo `data/model_int8_avgpool.tflite` tem 1.055.952 bytes, nenhum buffer externo e usa `AVERAGE_POOL_2D`, sem `SUM`. Entrada, quantização e saída mantêm o contrato descrito acima. A arena está configurada em 5 MiB. A identificação do firmware é `Tomato Village MobileNetV2 alpha0.5`; o nome CMake permanece `tomato_village_mobilenetv2_alpha1_esp32s3`, herdado da cópia da aplicação.

O relatório da reexportação em `training_code/tomato_village_dgx_MobileNetv2_alpha0.50/reexportar_tflite_int8/outputs/20261001_153647_622727/validation.json` registra 337/461 acertos (73,10%) com `BUILTIN_WITHOUT_DEFAULT_DELEGATES` e 336/461 (72,89%) com `BUILTIN_REF`, com 12 discordâncias de classe entre os resolvers. Esses números são da validação do artefato no PC.

| Métrica no ESP32 | Resultado |
|---|---:|
| Imagens enviadas / inferências válidas / falhas | 461 / 461 / 0 |
| Acertos / erros de classificação | 336 / 125 |
| Acurácia | 72,89% |
| F1 macro calculado da matriz | 67,66% |
| Recall macro calculado da matriz | 68,65% |
| Inferência média / P95 / P99 | 1.121,98 / 1.124,59 / 1.127,09 ms |
| Processamento médio no dispositivo | 1.138,01 ms |
| Requisição média / P95 / P99 | 1.376,46 / 1.567,15 / 1.816,70 ms |
| Duração total | 10 min 58 s |

| Classe | Acertos / imagens | Recall |
|---|---:|---:|
| Early_blight | 37/50 | 74,00% |
| Healthy | 14/22 | 63,64% |
| Late_blight | 68/92 | 73,91% |
| Leaf Miner | 83/104 | 79,81% |
| Magnesium Deficiency | 86/95 | 90,53% |
| Nitrogen Deficiency | 29/37 | 78,38% |
| Pottassium Deficiency | 5/8 | 62,50% |
| Spotted Wilt Virus | 14/53 | 26,42% |

A maior limitação é `Spotted Wilt Virus`: 39 das 53 imagens foram classificadas incorretamente, principalmente como `Early_blight` (16) e `Late_blight` (14). `Late_blight → Early_blight` ocorreu em 12 imagens. A precisão de `Early_blight` foi 48,05%; o recall de `Magnesium Deficiency` foi o maior, 90,53%. A classe `Pottassium Deficiency` tem apenas oito exemplos.

A inferência variou entre 1.119,39 e 1.128,87 ms. A diferença média entre requisição e processamento foi 238,45 ms, incluindo transferência e custos HTTP. A PSRAM livre permaneceu em 1.910.088 bytes; a memória interna livre variou entre 250.259 e 250.423 bytes, sem queda progressiva observada. O CSV não registra a arena usada.

### Comparação adicional das mesmas entradas

Uma análise adicional, realizada antes da atualização desta documentação, confirmou os hashes das 461 entradas. O PC com `BUILTIN_REF` produziu as mesmas classes que o ESP32 em todas as imagens, com 446/461 hashes de saída idênticos. Com `BUILTIN_WITHOUT_DEFAULT_DELEGATES`, houve 337 acertos, 12 discordâncias de classe frente ao ESP32 e nenhum hash de saída idêntico.

Essa comparação não faz parte do `summary.json` nem é executada automaticamente pelo avaliador. Ela não indica degeneração exclusiva do ESP32 nesta execução: o resultado de classificação coincide com o resolver de referência no PC, embora 15 saídas não sejam idênticas byte a byte. A validação disponível compara resolvers; não mede a perda de acurácia frente ao checkpoint PyTorch.

### Comparação entre os modelos embarcados

| Modelo / execução | Acertos / respostas válidas | Falhas | Acurácia | Inferência média | Requisição média | PSRAM livre |
|---|---:|---:|---:|---:|---:|---:|
| YuNet / `20260929_195442` | 402/461 | 0 | 87,20% | 737,12 ms | 1.011,06 ms | 3.896.648 bytes |
| MobileNetV2 alpha1 / `20261001_115908` | 417/454 | 7 | 91,85% | 2.500,66 ms | 3.789,74 ms | 173.384 bytes |
| MobileNetV2 alpha0.5 / `20261001_130328` | 336/461 | 0 | 72,89% | 1.121,98 ms | 1.376,46 ms | 1.910.088 bytes |
| MobileNetV2 scratch alpha1 / `20261002_102050` | 399/461 | 0 | 86,55% | 2.512,21 ms | 3.604,73 ms | 173.384 bytes |
| MobileNetV2 scratch alpha0.5 / `20261002_110629` | 383/461 | 0 | 83,08% | 1.125,49 ms | 1.390,99 ms | 1.910.088 bytes |

A alpha0.5 reduziu o tempo médio de inferência em 55,13% frente à alpha1, com mais PSRAM livre e nenhuma falha nesta execução. A diferença entre as acurácias publicadas foi de 18,96 pontos percentuais; a alpha1 exclui sete timeouts. Contra o YuNet, ambos com 461 respostas válidas, a alpha0.5 teve 66 acertos a menos e inferência aproximadamente 52,21% mais lenta. Nas execuções disponíveis, o YuNet apresentou maior acurácia e menor latência que a alpha0.5. Os tempos de requisição incluem condições de transporte específicas de cada execução.

## MobileNetV2 scratch alpha1: avaliação em 02/10/2026

Aplicação treinada do zero: `scratch_mobilenet_esp32s3_alpha1`. Resultados em `scratch_mobilenet_esp32s3_alpha1/python/results_esp32/20261002_102050/`. O modelo reexportado em `outputs/20261002_124511_689419/` tem 2.794.024 bytes, pesos incorporados e pooling `AVERAGE_POOL_2D`. O relatório da reexportação registra 401/461 acertos (86,98%) no PC padrão sem delegates e 399/461 (86,55%) no resolver de referência.

| Métrica no ESP32 | Resultado |
|---|---:|
| Imagens enviadas / inferências válidas / falhas | 461 / 461 / 0 |
| Acertos / erros de classificação | 399 / 62 |
| Acurácia | 86,55% |
| F1 macro calculado da matriz | 84,43% |
| Recall macro calculado da matriz | 84,77% |
| Inferência média / P95 / P99 | 2.512,21 / 2.515,23 / 2.517,55 ms |
| Processamento médio no dispositivo | 2.528,34 ms |
| Requisição média / P95 / P99 | 3.604,73 / 5.341,34 / 6.478,96 ms |
| Duração total | 28 min 4 s |

| Classe | Acertos / imagens | Recall |
|---|---:|---:|
| Early_blight | 41/50 | 82,00% |
| Healthy | 16/22 | 72,73% |
| Late_blight | 83/92 | 90,22% |
| Leaf Miner | 90/104 | 86,54% |
| Magnesium Deficiency | 90/95 | 94,74% |
| Nitrogen Deficiency | 35/37 | 94,59% |
| Pottassium Deficiency | 7/8 | 87,50% |
| Spotted Wilt Virus | 37/53 | 69,81% |

A maior confusão foi `Spotted Wilt Virus → Late_blight`, com dez imagens, seguida de `Leaf Miner → Healthy`, com sete. `Healthy` teve precisão de 61,54%. O recall de `Spotted Wilt Virus` foi o menor; `Pottassium Deficiency` tem apenas oito exemplos.

Os 461 hashes de saída foram distintos: não há saída constante nesta execução. A acurácia agregada coincide com o relatório do PC de referência, mas isso não comprova igualdade das predições individuais ou dos logits. Esta análise leu os resultados existentes; não executou novas inferências no PC.

A inferência variou entre 2.509,82 e 2.527,34 ms. A requisição chegou a 7.839,33 ms; a diferença média entre requisição e processamento foi 1.076,39 ms, incluindo transferência e custos HTTP. A PSRAM livre ficou constante em 173.384 bytes. A memória interna livre variou entre 250.195 e 250.287 bytes e terminou no mesmo valor inicial, sem queda progressiva observada. A margem de PSRAM é pequena; o CSV não informa a arena usada.

### Comparação pelos CSVs existentes

Os hashes de entrada coincidiram nas 454 imagens com respostas válidas da alpha1 anterior: a scratch acertou 393/454 (86,56%), contra 417/454 (91,85%), diferença de 24 acertos ou 5,29 pontos percentuais. A inferência média foi semelhante, 2.512,21 ms contra 2.500,66 ms. A scratch não apresentou falhas nesta execução; a alpha1 anterior teve sete timeouts. Essas observações não isolam a causa da diferença de acurácia nem garantem que os timeouts tenham sido eliminados.

Contra o YuNet, os 461 hashes de entrada coincidiram: a scratch acertou 399 imagens, contra 402, e teve inferência média de 2.512,21 ms contra 737,12 ms, aproximadamente 3,41 vezes o tempo. Nestas execuções, a scratch alpha1 não apresentou vantagem de acurácia ou latência frente ao YuNet.

## MobileNetV2 scratch alpha0.5: avaliação em 02/10/2026

Aplicação treinada do zero: `scratch_mobilenet_esp32s3_alpha05`. Resultados em `scratch_mobilenet_esp32s3_alpha05/python/results_esp32/20261002_110629/`. O artefato reexportado em `outputs/20261002_124728_816060/` tem 1.055.952 bytes, sem buffers externos, e usa `AVERAGE_POOL_2D`, sem `SUM`. Seu relatório registra 385/461 acertos (83,51%) no PC padrão sem delegates e 383/461 (83,08%) no resolver de referência.

| Métrica no ESP32 | Resultado |
|---|---:|
| Imagens enviadas / inferências válidas / falhas | 461 / 461 / 0 |
| Acertos / erros de classificação | 383 / 78 |
| Acurácia | 83,08% |
| F1 macro calculado da matriz | 81,84% |
| Recall macro calculado da matriz | 82,20% |
| Inferência média / P95 / P99 | 1.125,49 / 1.127,41 / 1.129,33 ms |
| Processamento médio no dispositivo | 1.141,63 ms |
| Requisição média / P95 / P99 | 1.390,99 / 1.527,55 / 1.798,06 ms |
| Duração total | 11 min 9 s |

| Classe | Acertos / imagens | Recall |
|---|---:|---:|
| Early_blight | 36/50 | 72,00% |
| Healthy | 17/22 | 77,27% |
| Late_blight | 77/92 | 83,70% |
| Leaf Miner | 87/104 | 83,65% |
| Magnesium Deficiency | 89/95 | 93,68% |
| Nitrogen Deficiency | 34/37 | 91,89% |
| Pottassium Deficiency | 7/8 | 87,50% |
| Spotted Wilt Virus | 36/53 | 67,92% |

A maior confusão foi `Leaf Miner → Healthy`, com dez imagens. `Late_blight → Spotted Wilt Virus` e o sentido inverso ocorreram oito vezes cada. `Healthy` teve precisão de 56,67%; `Spotted Wilt Virus` teve o menor recall. A precisão de `Pottassium Deficiency` foi 100%, mas com apenas oito imagens reais e sete predições dessa classe.

Os 461 hashes de saída foram distintos, sem saída constante nesta execução. A acurácia agregada coincide com o relatório do PC de referência; isso não comprova igualdade por imagem ou por logit. Esta análise utilizou somente os arquivos existentes, sem executar novas inferências.

A inferência variou entre 1.123,85 e 1.131,63 ms. A requisição chegou a 2.570,78 ms; a diferença média entre requisição e processamento foi 249,36 ms, incluindo transferência e custos HTTP. A PSRAM livre permaneceu em 1.910.088 bytes. A memória interna livre variou entre 250.167 e 250.295 bytes e terminou no mesmo valor inicial, sem queda progressiva observada. O CSV não informa a arena usada.

### Comparação pelos CSVs existentes

Os hashes das 461 entradas coincidiram com os registros da alpha0.5 anterior, da scratch alpha1 e do YuNet. As comparações de acurácia abaixo usam as mesmas entradas:

- Contra a alpha0.5 anterior: 383 contra 336 acertos, ganho de 47 acertos ou 10,20 pontos percentuais, com inferência média semelhante (1.125,49 contra 1.121,98 ms).
- Contra a scratch alpha1: 383 contra 399 acertos, perda de 16 acertos ou 3,47 pontos percentuais, com redução de 55,20% no tempo médio de inferência e mais PSRAM livre.
- Contra o YuNet: 383 contra 402 acertos, perda de 19 acertos ou 4,12 pontos percentuais. A inferência média foi 1.125,49 contra 737,12 ms; o YuNet também apresentou mais PSRAM livre.

A scratch alpha0.5 melhorou o resultado frente à alpha0.5 anterior nesta avaliação. Os registros não isolam a causa dessa melhora. Frente à scratch alpha1, oferece menor latência e menor ocupação observada de PSRAM, com perda de acurácia. Frente ao YuNet, não apresentou vantagem de acurácia, latência ou memória livre nas execuções disponíveis. Os tempos de requisição incluem as condições de transporte de cada execução.

## Gravar e testar

A partir da raiz de `ubiquos_tomato_village`:

```bash
cd esp32-s3-git-espnn-tflite/mobilenet_esp32s3_alpha1
python python/get_input_info.py
pio run -e esp32s3 -t upload
pio run -e esp32s3 -t uploadfs
pio device monitor -b 115200
```

Configure a rede em `src/main.cpp`. Use o IP informado no boot e execute em outro terminal, dentro da mesma aplicação e com o ambiente Python ativo:

```bash
ESP32_IP=192.168.3.22 python python/teste_inferencia.py
ESP32_IP=192.168.3.22 python python/evaluate_esp_model.py
```

O avaliador percorre todas as imagens de teste com `tqdm` e cria `python/results_esp32/<data_hora>/`. Matriz, CSV e resumo são descritos no [guia principal](README_principal.md). Para comparar com o PC, execute `python python/sanity_check_host.py 82 250`.

## Arena e diagnóstico

Com o artefato anterior, a arena de 4 MiB falhava com uma solicitação de 4.703.328 bytes para a região não persistente. Foi aumentada para 5 MiB; naquele teste, o dispositivo reportou 4.850.656 bytes usados. A avaliação da versão corrigida também registrou 173.384 bytes de PSRAM livre. O CSV dessa avaliação não informa a arena usada; preserve a distinção entre esses registros.

O tamanho do arquivo do modelo não determina a arena: pesos, ativações, metadados e scratch dos kernels ocupam regiões distintas. Confirme `Arena usada` e memória livre no boot da versão corrigida antes de reduzir a arena ou comparar desempenho.

Se a entrada estiver correta mas a saída divergir, use [OP-TRACE](README_debug.md) com a mesma imagem e modelo. `success: true` indica que `Invoke()` terminou sem erro; não garante resultados numéricos corretos.
