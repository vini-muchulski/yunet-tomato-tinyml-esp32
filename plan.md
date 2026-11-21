# Plano do repositório

O repositório acompanha o estudo multiclasse da YuNet-N e da MobileNetV2 no ESP32-S3. Cada etapa será tratada separadamente.

- [x] Criar o `README.md` e este plano.
- [ ] Selecionar e copiar os notebooks de treinamento usados no artigo para `notebooks/`, com as configurações de cada experimento.
- [ ] Registrar a origem do Tomato-Village, a ordem das classes e as divisões de treino, validação e teste em `data/`.
- [ ] Copiar os scripts de exportação e avaliação para `scripts/` e identificar os checkpoints e modelos TFLite em `models/`.
- [ ] Copiar os projetos PlatformIO usados nas medições para `firmware/esp32s3/` e registrar as versões das dependências.
- [ ] Copiar as evidências do artigo para `results/desktop/` e `results/esp32s3/`. Vincular cada execução ao modelo correspondente pelo SHA-256 e indicar os arquivos que sustentam cada tabela e figura.
- [ ] Documentar como regenerar os resultados publicados e repetir o benchmark no ESP32-S3 em `docs/reproduction.md`.
- [ ] Preparar a licença, o `CITATION.cff` e a versão de referência do artigo quando houver autorização para publicar.

Os experimentos multirrótulo e os protótipos Arduino ficam no repositório de pesquisa.
