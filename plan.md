# Plano do repositório

O repositório acompanha o estudo multiclasse da YuNet-N e da MobileNetV2 no ESP32-S3. As etapas são tratadas uma de cada vez.

## Concluído

- [x] Criar o `README.md` geral com figuras do dataset e exemplos de inferência no ESP32-S3.
- [x] Organizar os nove experimentos em `training/`, com notebooks originais, saídas salvas, checkpoints e exports TFLite.
- [x] Adicionar o notebook de download do Tomato-Village e as figuras do dataset em `data/`.
- [x] Organizar os scripts de avaliação embarcada e reexportação da MobileNetV2 em `scripts/`.
- [x] Organizar as cinco avaliações embarcadas em `results/esp32s3/` e as quatro reexportações com validação no computador em `results/desktop/`.
- [x] Adicionar os cinco projetos PlatformIO em `firmware/` e ligar o LittleFS ao submódulo compartilhado.
- [x] Mover o SSID e a senha dos firmwares para uma configuração local ignorada pelo Git.

## Pendente

- [ ] Validar a configuração e o build dos cinco firmwares com os submódulos, conferir os `dependencies.lock` atualizados e testar o carregamento do modelo e a inferência no ESP32-S3.
- [ ] Registrar o SHA-256 do modelo implantado em cada avaliação e indicar os arquivos que sustentam as tabelas e figuras do artigo.
- [ ] Documentar a versão do dataset, a ordem das classes e como gerar novamente as tabelas e repetir o benchmark em `docs/reproduction.md`.
- [ ] Revisar as licenças dos artefatos e preparar `LICENSE`, `CITATION.cff` e a versão citável do repositório para o artigo.

As pastas atuais em `firmware/` ficam como estão por enquanto. O notebook em `data/` baixa o dataset do Kaggle.

Os experimentos multirrótulo e os protótipos Arduino ficam no repositório de pesquisa.
