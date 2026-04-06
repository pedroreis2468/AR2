# Navegação Autónoma na Formula Student: Uma Abordagem Computacional Baseada em Aprendizagem por Reforço

**UC:** Aprendizagem por Reforço — Mestrado em Inteligência Artificial, Universidade do Minho, 2025/26

**Grupo 1:**
| Nº | Nome |
|---------|------|
| PG60390 | Luís Miguel Pereira Silva |
| PG59908 | Pedro Miguel Soares de Albergaria Urbano dos Reis |

---

## Descrição

Agente de Reinforcement Learning que aprende a conduzir um carro de Formula Student Driverless numa pista delimitada por cones, seguindo as regras oficiais da competição FS-AI. O ambiente de simulação implementa penalizações realistas: cones derrubados aplicam penalizações de tempo (+2s cada), o carro pode ultrapassar os limites da pista (com penalizações graduais), e o episódio só termina em situações de DOO (*Did Not Operate*) — como demasiados cones derrubados ou desvio extremo do percurso.

### Regras FS-AI implementadas

- **Cones azuis** (fronteira esquerda) e **amarelos** (fronteira direita), **laranja** no start/finish
- Cones derrubados = **penalização de 2s** por cone (não terminam o episódio)
- Cones laranja = penalidade agravada (4s)
- **DOO** (terminação) apenas se: ≥10 cones derrubados, off-course >5m, ou off-course >2s consecutivos
- Cones derrubados são **removidos** da perceção do agente (simulando remoção física)

## Arquitetura

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

### Monitorizar treino

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