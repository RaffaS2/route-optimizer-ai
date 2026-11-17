IA26_P01_G03 - Projeto 01 de Inteligencia Artificial (Grupo 03)
Otimizacao de cadeias de abastecimento de comida local (recolha e entrega com janelas temporais)

Grupo 03
  Joao Soares    26593
  Edgar Casal    31026
  Joao Machado   31510
  Rafael Soares  24869

Estrutura
  data_tiny/                          dataset tiny (CSV)
  data_small/                         dataset small (CSV)
  src/model.py                        entidades, leitura dos CSVs, funcao de custo e verificador de rotas
  src/informed.py                     pesquisa informada: Greedy Best-First e A* (2 heuristicas)
  src/local_search.py                 pesquisa local: Hill Climbing (steepest e first improvement)
  src/csp.py                          CSP: backtracking, forward checking e AC-3
  Notebooks/01_Project01_G03.ipynb    notebook principal (codigo + documentacao + resultados)
  Notebooks/02_eda_tiny.ipynb         analise exploratoria dos dados
  requirements.txt                    bibliotecas necessarias

Como correr
  1. pip install -r requirements.txt
  2. Abrir Notebooks/01_Project01_G03.ipynb (Jupyter ou VS Code) e correr todas as celulas.
     O notebook principal demora cerca de 20 segundos.
  Para usar outro dataset com a mesma estrutura, basta criar a pasta (ex.: data_medium/)
  e carregar com Problem(os.path.join(ROOT, "data_medium")).
