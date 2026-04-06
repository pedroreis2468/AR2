# 🏎️ Navegação Autónoma na Formula Student: Uma Abordagem Computacional Baseada em Aprendizagem por Reforço


![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Aprendizagem por Reforço** | Mestrado em Inteligência Artificial | Universidade do Minho | 2025/26

Agente de Reinforcement Learning que aprende a conduzir um carro de Formula Student Driverless numa pista delimitada por cones, seguindo as regras oficiais da competição FS-AI. O ambiente de simulação implementa penalizações realistas: cones derrubados aplicam penalizações de tempo (+2s cada), o carro pode ultrapassar os limites da pista (com penalizações graduais), e o episódio só termina em situações de DOO (*Did Not Operate*) — como demasiados cones derrubados ou desvio extremo do percurso.

---

### Regras FS-AI implementadas

- **Cones azuis** (fronteira esquerda) e **amarelos** (fronteira direita), **laranja** no (começo e fim)
- Cones derrubados = **penalização de 2s** por cone (não terminam o episódio)
- Cones laranja = penalidade agravada (4s)
- **DOO** (terminação) apenas se: ≥10 cones derrubados, off-course >5m, ou off-course >2s consecutivos
- Cones derrubados são **removidos** da perceção do agente (simulando remoção física)

## Arquitetura

```
AR/
├── agent/                  # Implementação do agente RL
│   ├── sac.py              # Soft Actor-Critic (custom, PyTorch)
│   ├── networks.py         # Redes Actor-Critic (Gaussian Actor + Twin Q)
│   └── replay_buffer.py    # Replay buffer circular
├── env/                    # Ambiente de simulação Gymnasium
│   ├── racing_env.py       # Ambiente principal (FSRacingEnv)
│   ├── car_model.py        # Modelo cinemático de bicicleta
│   ├── track_generator.py  # Gerador procedural de pistas
│   ├── cone_sensor.py      # Simulação de sensores (LIDAR/câmara)
│   └── renderer.py         # Visualização PyGame
├── config/                 # Ficheiros de configuração
│   └── default.yaml        # Parâmetros por defeito
├── docs/                   # Documentação adicional
│   ├── planeamento.pdf     # Relatório de planeamento (LNCS)
│   ├── planeamento.tex     # Fonte LaTeX do relatório
│   └── pacsim_setup.md     # Setup do PacSim 3D (objetivo secundário)
├── train.py                # Script de treino (SAC custom + SB3)
├── evaluate.py             # Script de avaliação e visualização
└── requirements.txt        # Dependências Python
```

## Componentes principais

- **Ambiente 2D** (`FSRacingEnv`) — modelo cinemático de bicicleta, pistas procedurais fechadas, cones com cores FS-AI, sistema de penalizações realista
- **Observação** — vetor de 20 dimensões (estado do veículo + 3 cones azuis/amarelos mais próximos + distância às fronteiras + erro de heading)
- **SAC** (Soft Actor-Critic) como algoritmo principal — adequado para espaço de ações contínuo (steering + throttle)
- **PPO** como baseline de comparação via Stable-Baselines3
- **Domain randomization** — variação de massa, atrito e ruído sensorial para robustez

### Extensão exploratória: PacSim (3D)

Como objetivo secundário, está a ser explorada a integração com o [PacSim](https://github.com/PacSim/pacsim), um simulador 3D baseado em ROS 2. A ideia é investigar a transferência de políticas treinadas no ambiente 2D para o simulador 3D (*sim-to-sim transfer*), validando a robustez do agente num cenário mais realista. Esta componente é exploratória e não condiciona o trabalho principal. Notas de instalação em [`docs/pacsim_setup.md`](docs/pacsim_setup.md).

## Instalação

```bash
git clone https://github.com/pedroreis2468/AR.git
cd AR
pip install -r requirements.txt
```

### Dependências principais

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Gymnasium ≥ 0.29
- PyGame ≥ 2.5
- Stable-Baselines3 ≥ 2.1 (opcional, para modo SB3)

## Utilização

### Testar o ambiente (agente aleatório)

```bash
python evaluate.py --random
```

### Treinar o agente

```bash
# SAC custom (implementação educativa)
python train.py --mode custom --total-steps 500000

# SB3 SAC (produção)
python train.py --mode sb3 --algo sac --total-steps 1000000

# SB3 PPO (baseline)
python train.py --mode sb3 --algo ppo --total-steps 1000000
```

### Avaliar modelo treinado

```bash
# Modelo SAC custom
python evaluate.py --model runs/<run_dir>/best_model.pt --mode custom

# Modelo SB3
python evaluate.py --model runs/<run_dir>/final_model.zip --mode sb3
```

### Monitorizar treino

```bash
tensorboard --logdir runs/
```

---

**Grupo 1:**
| Nº | Nome |
|---------|------|
| PG60390 | Luís Miguel Pereira Silva |
| PG59908 | Pedro Miguel Soares de Albergaria Urbano dos Reis |

---

## Referências

- Kabzan, J. et al. (2020). [AMZ Driverless: The Full Autonomous Racing System](https://doi.org/10.1002/rob.21977). *Journal of Field Robotics*, 37(7), 1267–1294.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. ISBN 978-0-262-03924-6.
- Kong, J., Pfeiffer, M., Schildbach, G., & Borrelli, F. (2015). [Kinematic and Dynamic Vehicle Models for Autonomous Driving Control Design](https://doi.org/10.1109/IVS.2015.7225830). *IEEE Intelligent Vehicles Symposium*, 1094–1099.
- Haarnoja, T. et al. (2018). [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905). arXiv:1812.05905.
- Schulman, J. et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
- Tobin, J. et al. (2017). [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://doi.org/10.1109/IROS.2017.8202133). *IEEE/RSJ IROS*, 23–30.
- Ulrich, F., & Wehrli, T. (2024). [End-to-End Deep Reinforcement Learning for Autonomous Racing Dynamics](https://www.zhaw.ch/storage/engineering/institute-zentren/cai/studentische_arbeiten/Spring_2024/BA_FS24_Fabian_Ulrich_Tobias_Wehrli_End-to-End_Deep_Reinforcement_Learning_for_Autonomous_Racing_Dynamics.pdf). Bachelor Thesis, ZHAW.
