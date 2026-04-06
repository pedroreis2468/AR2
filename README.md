# 🏎️ Formula Student Driverless — Navegação Autónoma com Aprendizagem por Reforço

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green)
![SAC](https://img.shields.io/badge/Algorithm-SAC-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Aprendizagem por Reforço** | Mestrado em Inteligência Artificial | Universidade do Minho | 2025/26

Agente de Reinforcement Learning que aprende a conduzir um carro de Formula Student Driverless numa pista delimitada por cones, seguindo as regras oficiais da competição **FS-AI**. O ambiente 2D simula sensores, física cinemática e penalizações realistas, treinando com pistas reais de competição em formato YAML.

---

## 🎯 Objetivos

* Treinar um agente SAC capaz de completar voltas em pistas FS sem derrubar cones
* Implementar regras FS-AI realistas: penalizações por cone (+2s), DOO, off-course gradual
* Suportar pistas reais de competição (FSG19, FSE22, FSE24, etc.) via ficheiros YAML
* Transferir políticas treinadas para o simulador 3D PacSim (objetivo exploratório)

---

## 🏗️ Arquitetura

```
                    ┌─────────────────┐
                    │   train.py      │  SAC custom ou SB3
                    │   evaluate.py   │  Avaliação + visualização
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  FSRacingEnv    │  Gymnasium environment
                    │  (racing_env)   │
                    └──┬─────┬─────┬──┘
                       │     │     │
              ┌────────▼┐ ┌──▼───┐ ┌▼──────────┐
              │ CarModel│ │Sensor│ │TrackLoader │
              │ (bicicl)│ │(cone)│ │ (YAML/proc)│
              └─────────┘ └──────┘ └────────────┘
```

### Componentes

| Componente | Ficheiro | Descrição |
|-----------|----------|-----------|
| **FSRacingEnv** | `env/racing_env.py` | Ambiente Gymnasium — observação, reward, FS-AI rules |
| **KinematicBicycleModel** | `env/car_model.py` | Modelo cinemático de bicicleta (250kg, slicks) |
| **ConeSensor** | `env/cone_sensor.py` | Perceção de cones no ref. do carro (simula YOLO+depth) |
| **YAMLTrackLoader** | `env/track_loader.py` | Carrega pistas reais de competição FS |
| **TrackGenerator** | `env/track_generator.py` | Gerador procedural de pistas (fallback) |
| **FSRenderer** | `env/renderer.py` | Visualização PyGame em tempo real |
| **SACAgent** | `agent/sac.py` | Soft Actor-Critic custom (PyTorch) |
| **Networks** | `agent/networks.py` | Gaussian Actor + Twin Q-Networks |

---

## 🔧 Regras FS-AI Implementadas

| Regra | Implementação |
|-------|--------------|
| 🔵 Cones azuis (esquerda) | Fronteira esquerda da pista |
| 🟡 Cones amarelos (direita) | Fronteira direita da pista |
| 🟠 Cones laranja (start/finish) | Penalidade agravada (×2) |
| Cone derrubado | **+2s penalização** (não termina episódio) |
| Cone derrubado — perceção | **Removido do FOV** (persistent knockdown) |
| DOO: ≥10 cones | Terminação imediata |
| DOO: off-course >5m | Terminação imediata |
| DOO: off-course >2s | Terminação imediata |
| Off-course gradual | Penalização quadrática progressiva |
| Volta completa | Bónus `200 - cones×5` |

---

## 👁️ Espaço de Observação (27 dims)

| Índices | Componente | Descrição |
|:-------:|-----------|-----------|
| 0 | `vx` | Velocidade longitudinal normalizada |
| 1 | `vy` | Velocidade lateral normalizada |
| 2 | `ω` | Yaw rate normalizado |
| 3 | `δ` | Ângulo de steering normalizado |
| 4 | `ax` | Aceleração longitudinal normalizada |
| 5 | `ay` | Aceleração lateral normalizada |
| 6–11 | `blue_cones` | 3 cones azuis mais próximos (x, y) |
| 12–17 | `yellow_cones` | 3 cones amarelos mais próximos (x, y) |
| 18–23 | `orange_cones` | 3 cones laranja mais próximos (x, y) |
| 24 | `dist_left` | Distância à fronteira esquerda |
| 25 | `dist_right` | Distância à fronteira direita |
| 26 | `heading_err` | Erro de heading relativo à pista |

> Flag `--legacy-obs` reduz para 21 dims (sem orange cones) para compatibilidade com modelos antigos.

---

## 🎁 Reward Shaping

| Componente | Peso | Descrição |
|-----------|:----:|-----------|
| `r_progress` | ×2.0 | Distância percorrida ao longo da centerline (com sinal) |
| `r_alignment` | ×0.4 | `v × cos(heading_error)` — andar rápido E alinhado |
| `r_smooth` | ×0.05 | Penaliza mudanças de steering > dead-band de 0.1 |
| `r_lateral` | ×0.08 | Linear dentro da pista, quadrática fora |
| `r_time` | -0.005 | Custo fixo por step (encorajar velocidade) |
| `cone_hit` | -8.0 | Por cone azul/amarelo derrubado |
| `orange_hit` | -16.0 | Por cone laranja derrubado |
| `off-course` | quad. | Penalização progressiva fora dos limites |
| `stagnation` | -5.0 | Se <2m de progresso em 300 steps → DOO |

---

## 📂 Estrutura do Repositório

```
AR/
├── train.py                       # Script de treino (SAC custom + SB3)
├── evaluate.py                    # Avaliação e visualização de modelos
├── debug_reward.py                # Debug de componentes do reward
├── test_steering.py               # Teste de steering com modelo treinado
├── requirements.txt               # Dependências Python
│
├── agent/                         # Implementação do agente RL
│   ├── sac.py                     #   Soft Actor-Critic (PyTorch)
│   ├── networks.py                #   Gaussian Actor + Twin Q-Networks
│   └── replay_buffer.py           #   Replay buffer circular
│
├── env/                           # Ambiente de simulação Gymnasium
│   ├── racing_env.py              #   FSRacingEnv (ambiente principal)
│   ├── car_model.py               #   Modelo cinemático de bicicleta
│   ├── track_generator.py         #   Gerador procedural de pistas
│   ├── track_loader.py            #   Carregador de pistas YAML (FS reais)
│   ├── cone_sensor.py             #   Simulação de sensores (perceção de cones)
│   └── renderer.py                #   Visualização PyGame
│
├── config/                        # Configuração
│   └── default.yaml               #   Parâmetros por defeito (veículo, pista, SAC, PPO)
│
├── docs/                          # Documentação
│   ├── planeamento.pdf            #   Relatório de planeamento (LNCS)
│   ├── planeamento.tex            #   Fonte LaTeX
│   └── pacsim_setup.md            #   Setup do PacSim 3D
│
└── runs/                          # Modelos treinados (gerado pelo treino)
    └── sb3_sac_<timestamp>/
        ├── best/best_model.zip
        ├── checkpoints/
        ├── vecnormalize.pkl
        └── tb/                    # TensorBoard logs
```

---

## 🚀 Reprodução

### Pré-requisitos

* **Python ≥ 3.10** (via Anaconda/Miniconda)
* **Pistas YAML** — clonar o repositório de pistas para `../pistas/`

### Passos

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/pedroreis2468/AR.git
   cd AR
   ```

2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Obter pistas (se ainda não tiver):**
   ```bash
   cd .. && git clone <repo-pistas> pistas && cd AR
   ```

4. **Testar o ambiente (agente aleatório):**
   ```bash
   python evaluate.py --random --tracks-dir ../pistas/tracks
   ```

5. **Treinar:**
   ```bash
   # SAC custom (educativo)
   python train.py --mode custom --total-steps 500000 --tracks-dir ../pistas/tracks

   # SB3 SAC (produção, recomendado)
   python train.py --mode sb3 --algo sac --total-steps 2500000 --tracks-dir ../pistas/tracks

   # SB3 PPO (baseline)
   python train.py --mode sb3 --algo ppo --total-steps 1000000 --tracks-dir ../pistas/tracks
   ```

6. **Avaliar modelo treinado:**
   ```bash
   # SB3 (auto-detecta VecNormalize)
   python evaluate.py --model runs/<run_dir>/best/best_model.zip --mode sb3 --tracks-dir ../pistas/tracks

   # Pista específica
   python evaluate.py --model runs/<run_dir>/best/best_model.zip --mode sb3 --track FSG19
   ```

7. **Monitorizar treino:**
   ```bash
   tensorboard --logdir runs/
   ```

### Debug

```bash
# Verificar componentes do reward numa pista
python debug_reward.py --tracks-dir ../pistas/tracks --steps 300

# Testar steering com modelo treinado
python test_steering.py
```

---

## ⚙️ Configuração

Parâmetros editáveis em `config/default.yaml`:

| Secção | Parâmetros |
|--------|-----------|
| **environment** | `dt`, `action_repeat`, `max_episode_steps`, `randomize_track` |
| **vehicle** | `mass`, `wheelbase`, `max_speed`, `max_steering`, `mu` |
| **track** | `track_width`, `cone_spacing`, `min_radius`, `arena_size` |
| **penalties** | `cone_penalty_reward`, `doo_cone_limit`, `oc_time_limit` |
| **sensors** | `max_range`, `fov`, `n_closest_per_side`, `noise_range` |
| **sac** | `hidden_dims`, `lr_actor`, `gamma`, `tau`, `buffer_size` |
| **training** | `total_steps`, `n_envs`, `eval_freq`, `checkpoint_freq` |

---

## 🧪 Extensão Exploratória: PacSim (3D)

Como objetivo secundário, está a ser explorada a integração com o [PacSim](https://github.com/PacSim/pacsim), um simulador 3D baseado em ROS 2, para investigar *sim-to-sim transfer* de políticas treinadas no ambiente 2D. Notas de instalação em [`docs/pacsim_setup.md`](docs/pacsim_setup.md).

---

## 📚 Referências

* Kabzan, J. et al. (2020). [AMZ Driverless: The Full Autonomous Racing System](https://doi.org/10.1002/rob.21977). *J. Field Robotics*, 37(7).
* Haarnoja, T. et al. (2018). [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905). arXiv:1812.05905.
* Schulman, J. et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv:1707.06347.
* Kong, J. et al. (2015). [Kinematic and Dynamic Vehicle Models for Autonomous Driving](https://doi.org/10.1109/IVS.2015.7225830). *IEEE IV*.
* Tobin, J. et al. (2017). [Domain Randomization for Sim-to-Real Transfer](https://doi.org/10.1109/IROS.2017.8202133). *IEEE/RSJ IROS*.
* Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## 👥 Grupo 1

| Nº | Nome | Email |
|----|------|-------|
| PG60390 | Luís Miguel Pereira Silva | pg60390@alunos.uminho.pt |
| PG59908 | Pedro Miguel S. A. Urbano dos Reis | pg59908@alunos.uminho.pt |

---

## 📜 Licença

Este trabalho é de cariz estritamente académico. Universidade do Minho, Escola de Engenharia, Departamento de Informática.