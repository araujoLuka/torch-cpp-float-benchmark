# torch-cpp-float-benchmark

Este projeto é parte do Trabalho de Conclusão de Curso (TCC) do estudante **Lucas Araujo**, graduando em Bacharelado em Ciência da Computação pela **UFPR**. A pesquisa explora o impacto de diferentes representações de ponto flutuante em redes neurais convolucionais (CNN), com foco na área de Aprendizagem de Máquina (Machine Learning), utilizando **LibTorch** (PyTorch em C++) para avaliar desempenho computacional, consumo de energia e uso de memória.

## 🎯 Objetivo

Avaliar o impacto de diferentes representações de ponto flutuante (`float16`, `float32`, `float64`) em redes neurais convolucionais (CNN) utilizando **LibTorch** (PyTorch em C++) para:

- Desempenho computacional (tempo de inferência)
- Consumo de energia
- Uso de memória

O experimento utiliza o **PKLot Dataset**, voltado à **classificação de vagas de estacionamento** com imagens.

## 🧠 Tecnologias e ferramentas

- C++17
- LibTorch (PyTorch em C++)
- LIKWID (benchmarking e métricas de energia/FLOPS)
- Dataset PKLot
- Modularização orientada a objetos

## 🗂️ Estrutura

```
.
├── main.cpp                 # Execução principal
├── net/                     # Arquitetura da rede neural
├── data/                    # Loader do dataset PKLot
├── include/                 # Funções auxiliares
├── benchmark/               # Código para medir desempenho
├── scripts/                 # Scripts de automação (ex: LIKWID)
├── CMakeLists.txt           # Build via CMake
```

## 📊 Métricas analisadas

- Tempo de execução
- Consumo de energia (via LIKWID)
- MFLOPS e GFLOPS por tipo de dado
- Acurácia do modelo com float32 (referência)

## 🚀 Como compilar

``` bash
mkdir build && cd build
cmake ..
make
```

## 🔎 Como executar benchmarks

``` bash
./scripts/run_likwid.sh
```

---

## 👨‍💻 Autor

**Lucas Araujo**  
Graduando em Ciência da Computação - UFPR  
Contato: [GitHub](https://github.com/araujoLuka) | [LinkedIn](https://linkedin.com/in/lukaraujo)
