# 🏎️ Formula Student Driverless — Navegação Autónoma com Aprendizagem por Reforço

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green)
![SB3](https://img.shields.io/badge/Stable--Baselines3-2.1+-purple)
![Algorithms](https://img.shields.io/badge/Algorithms-SAC%20%7C%20PPO-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> **Aprendizagem por Reforço** · Mestrado em Inteligência Artificial · Universidade do Minho · 2025/26

Agente de Reinforcement Learning que aprende a conduzir um carro de Formula Student Driverless em pistas delimitadas por cones, seguindo as regras oficiais **FS-AI**. O ambiente 2D simula sensores de perceção, física cinemática e penalizações realistas, sobre **14 pistas reais de competição** em formato YAML (9 de treino, 5 de teste inéditas).

---

## 📊 Resultados (TL;DR)

Comparação entre os melhores modelos no test split FS-AI (5 pistas inéditas, 10 episódios por pista, política determinística, com domain randomization).

| Modelo                | Lap rate | Cones derrubados | Velocidade |
|-----------------------|---------:|-----------------:|-----------:|
| **SAC** (3M, 3 seeds) | **66%**  | 2.4              | 76 km/h    |
| PPO (6M, 3 seeds)     | 68%      | 4.8              | 79 km/h    |
| **SAC + Mirror Aug.** | **72%**  | **2.9**          | 67 km/h    |

**Principais descobertas:**

- ✅ **SAC supera PPO em consistência** (3× menos cones derrubados) apesar de PPO ter ligeiramente mais lap rate em algumas seeds
- 🔍 **Identificámos viés direcional** via avaliação em pistas espelhadas: lap rate cai de 66% → 40% (SAC) e 68% → 45% (PPO)
- 🎯 **Mirror augmentation** elimina o viés (74% em espelhadas) e melhora generalização global (+6pp em pistas originais)
- 👁️ **Robustez de perceção assimétrica:** o agente tolera *flicker* do sensor (≤10% de deteções perdidas/frame ≈ sem efeito) mas **não** cones fisicamente em falta — a remoção persistente colapsa o lap rate já a 5–10%
- 🛡️ **Augmentation dirigida corrige e compõe-se:** treinar com *cone-dropout* recupera os cones em falta e **juntar mirror + dropout é super-aditivo** (17%→48% de lap rate a 10% de cones removidos), sem custo no baseline

---

## 📦 Entrega e Avaliação

Para facilitar a avaliação do trabalho, a entrega inclui três pontos de entrada:

| Artefacto | Ficheiro | O que é |
|-----------|----------|---------|
| 📄 **Relatório final** | [`docs/report.pdf`](docs/report.pdf) | Relatório final (PDF) com metodologia, resultados, ablações e discussão. |
| 📓 **Notebook de avaliação** | [`notebooks/overview.ipynb`](notebooks/overview.ipynb) | Reúne os resultados principais (tabelas + figuras) e permite **correr avaliação ao vivo** dos modelos treinados. Já vem com os outputs executados. |
| 📊 **Resultados agregados** | [`results/aggregated/`](results/aggregated/) | CSVs e tabelas LaTeX prontas (SAC vs PPO, per-track, ablações, viés direcional). |

**Abrir o notebook** (não precisa de GPU; as tabelas e curvas correm em segundos, os GIFs ficam em cache após a 1.ª geração):

```bash
pip install -r requirements.txt
jupyter notebook notebooks/overview.ipynb   # depois: Kernel → Restart & Run All
```

---

## 🎯 Objetivos

- Treinar um agente SAC/PPO capaz de completar voltas em pistas FS sem derrubar cones
- Implementar regras FS-AI realistas: penalizações por cone (+2s), DOO, off-course gradual
- Suportar pistas reais de competição (FSG19, FSE22, FSE24, etc.) via ficheiros YAML
- **Avaliar generalização e robustez** via splits canónicos, pistas espelhadas e cones em falta (held-out)

---

## 🏗️ Arquitetura

```
       ┌─────────────────┐
       │   train.py      │  SB3 SAC/PPO
       │   evaluate.py   │  Avaliação + visualização PyGame
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

### Componentes principais

| Componente               | Ficheiro                  | Descrição                                          |
|--------------------------|---------------------------|----------------------------------------------------|
| **FSRacingEnv**          | `env/racing_env.py`       | Ambiente Gymnasium — observação, reward, FS-AI    |
| **KinematicBicycleModel**| `env/car_model.py`        | Modelo cinemático de bicicleta (250kg, slicks)    |
| **ConeSensor**           | `env/cone_sensor.py`      | Perceção de cones no ref. do carro (FOV/range)    |
| **YAMLTrackLoader**      | `env/track_loader.py`     | Carrega pistas reais FS via YAML                   |
| **TrackGenerator**       | `env/track_generator.py`  | Gerador procedural de pistas (fallback)            |
| **FSRenderer**           | `env/renderer.py`         | Visualização PyGame em tempo real                  |
| **track_splits.py**      | `env/track_splits.py`     | Split canónico train (9) / test (5)                |

### Scripts auxiliares (`scripts/`)

| Script                          | Função                                                       |
|---------------------------------|--------------------------------------------------------------|
| `eval_per_track.py`             | Avaliação per-pista (gera CSV + tabela LaTeX)               |
| `aggregate_seeds.py` / `aggregate_ablations.py` | Agregam seeds/ablações → CSV + tabelas LaTeX |
| `make_report_figures.py`        | Gera as figuras do relatório/notebook a partir dos CSVs      |
| `make_reward_animation.py`      | Anima a recompensa a evoluir enquanto o carro conduz (GIF)   |
| `reward_iterations.py`          | Ablação de reward: jornada de 6 fixes (treina N seeds + avalia + figuras) |
| `plot_learning_curves.py`       | Curvas de aprendizagem agregadas (SAC vs PPO)                |
| `mirror_tracks.py`              | Espelha pistas em Y (gera test held-out de viés direcional) |
| `eval_perception_robustness.py` | Sweep de robustez de perceção: *flicker* transiente (`--mode perception`) / remoção estrutural de cones (`--mode structural`) → CSV + figura |
| `make_perception_failure_gif.py`| GIF + contraste do carro a falhar por falta de cones (modo *perception*/*structural*) |
| `augmentation_matrix.py`        | Matriz 2×2 de robustez (espelhado × cones em falta) nos 4 modelos do design factorial |
| `debug_reward.py`               | Debug dos componentes da recompensa                          |
| `test_steering.py`              | Teste de inferência de ações                                 |
| `run_pacsim_viz.sh` + `test_pacsim_*.py` | Exploração sim-to-sim com PacSim 3D                 |

---

## 🔧 Regras FS-AI Implementadas

| Regra                              | Implementação                                  |
|------------------------------------|------------------------------------------------|
| 🔵 Cones azuis (esquerda)          | Fronteira esquerda da pista                    |
| 🟡 Cones amarelos (direita)        | Fronteira direita da pista                     |
| 🟠 Cones laranja (start/finish)    | Penalidade agravada (×2)                       |
| Cone derrubado                     | **+2s** penalização (não termina episódio)     |
| Cone derrubado — perceção          | **Removido do FOV** (persistent knockdown)     |
| DOO: ≥10 cones                     | Terminação imediata                            |
| DOO: off-course >5m                | Terminação imediata                            |
| DOO: off-course >2s                | Terminação imediata                            |
| Off-course gradual                 | Penalização quadrática progressiva             |
| Volta completa                     | Bónus +200 − cones×5                           |

---

## 👁️ Espaço de Observação (24 dims)

> **Nota:** O agente é avaliado estritamente pela sua perceção local (visão dos cones), sem acesso à *centerline* ideal. O planeamento global emerge do comportamento reativo.

| Índices | Componente   | Descrição                                        |
|---------|--------------|--------------------------------------------------|
| 0       | vx           | Velocidade longitudinal normalizada              |
| 1       | vy           | Velocidade lateral normalizada                   |
| 2       | ω            | Yaw rate normalizado                             |
| 3       | δ            | Ângulo de steering normalizado                   |
| 4       | ax           | Aceleração longitudinal normalizada              |
| 5       | ay           | Aceleração lateral normalizada                   |
| 6–11    | blue_cones   | 3 cones azuis mais próximos (x, y) no ref. carro |
| 12–17   | yellow_cones | 3 cones amarelos mais próximos (x, y)            |
| 18–23   | orange_cones | 3 cones laranja mais próximos (x, y)             |

> A flag `--legacy-obs` reduz para 18 dims (sem cones laranja) para compatibilidade com modelos antigos.

---

## 🎁 Reward Shaping

| Componente   | Peso      | Descrição                                                     |
|--------------|-----------|---------------------------------------------------------------|
| `r_progress` | ×2.0      | Distância percorrida ao longo da centerline ideal (com sinal) |
| `r_alignment`| ×0.4      | v × cos(heading_error) — andar rápido E alinhado              |
| `r_smooth`   | ×0.05     | Penaliza mudanças de steering > dead-band de 0.1              |
| `r_lateral`  | ×0.08     | Linear dentro da pista, quadrática fora                       |
| `r_time`     | −0.005    | Custo fixo por step (encorajar velocidade)                    |
| `cone_hit`   | −8.0      | Por cone azul/amarelo derrubado                               |
| `orange_hit` | −16.0     | Por cone laranja derrubado                                    |
| `off-course` | quad.     | Penalização progressiva fora dos limites                      |
| `stagnation` | −5.0      | Se <2m de progresso em 300 steps → DOO                        |
| `lap_bonus`  | +200      | Por volta completa (descontado pelos cones)                   |

---

## 🔬 Protocolo Experimental

### Splits canónicos (`env/track_splits.py`)

| Split  | N° pistas | Pistas                                                                |
|--------|-----------|-----------------------------------------------------------------------|
| Train  | 9         | FSG19, FSG21, FSG23, FSE22, FSE22_test, FSE23, FSO20, FSS19, FSS22_V1 |
| Test   | 5         | FSG24, FSI24, FSCZ24, FSE24, FSS22_V2                                 |
| Mirror | 14        | Versões espelhadas em Y (held-out para viés direcional)               |

### Modelos treinados

| Família       | Algoritmo | Steps | Seeds | Observações                          |
|---------------|-----------|-------|-------|--------------------------------------|
| Baseline      | SAC       | 3M    | 3     | `runs/sac_seed{0,1,2}`               |
| Baseline      | PPO       | 6M    | 3     | `runs/ppo_seed{0,1,2}`               |
| Ablações      | SAC       | 2M    | 1     | 15 variantes (`runs/abl_sac_*`)      |
| Mirror Aug.   | SAC       | 3M    | 1     | `runs/sac_mirror_aug`                |
| Cone-dropout Aug.   | SAC | 3M  | 1     | `runs/sac_dropout_aug`               |
| Mirror+Dropout Aug. | SAC | 3M  | 1     | `runs/sac_mirror_dropout_aug`        |

### Ablações realizadas

- **Recompensa**: `--no-alignment`, `--no-persistent-knockdown`
- **Domain Randomization**: `--no-dr`
- **Sensor**: `--no-sensor-noise`, `--cone-fov-deg {90, 360}`, `--cone-range 8`
- **Dinâmica**: `--max-speed 16.7`
- **Observação**: `--legacy-obs`
- **Hiperparâmetros**: `--lr 1e-4`, `--buffer-size 1000000`, `--batch-size 512`, `--target-entropy {-2.0, -1.0, -0.2}`
- **Data Augmentation**: `--mirror-augment`, `--cone-dropout-augment`

### ⚠️ Nota metodológica: avaliação estocástica

> A avaliação per-track corre **N=10 episódios por pista com _domain randomization_ ativa**: cada episódio sorteia massa (±15%), atrito (±20%), arrasto (±10%), posição/velocidade iniciais e ruído de perceção. Os valores reportados são, portanto, **estimativas da performance média sob variação** — não medições determinísticas.
>
> Estas fontes de aleatoriedade usam o gerador **global** do NumPy, que o avaliador não semeia; o `seed` passado a cada episódio (`reset(seed=...)`) só afeta o `np_random` do Gymnasium, que o ambiente não usa para esta variabilidade. Logo, **re-execuções produzem valores ligeiramente diferentes** (variação observada ≈ ±5–8 pp no _lap rate_ com N=10). Implicações para a leitura dos resultados:
>
> - As médias de 3 seeds são estáveis em tendência e suportam as conclusões.
> - **Diferenças de poucos pontos percentuais entre modelos/ablações estão dentro do ruído amostral** e não devem ser sobre-interpretadas.
> - Os efeitos grandes — consistência SAC vs PPO (≈2× menos cones), viés direcional (≈−25 pp) e _mirror augmentation_ (+34 pp em espelhadas) — situam-se muito acima desse ruído e são robustos.

---

## 📂 Estrutura do Repositório

```
AR2/
├── agent/                    # Implementação custom (SAC, redes, buffer)
├── config/                   # default.yaml — parâmetros base
├── docs/                     # report.pdf, planeamento.pdf, apresentação, pacsim_setup.md
│   ├── report.pdf            # ← RELATÓRIO FINAL (PDF, adicionar antes da entrega)
│   └── planeamento.tex/.pdf  # Relatório de planeamento
├── notebooks/                # ← NOTEBOOK DE AVALIAÇÃO
│   └── overview.ipynb        # Resultados + avaliação ao vivo (executado)
├── env/                      # Ambiente de simulação Gymnasium
│   ├── racing_env.py         # FSRacingEnv (ambiente principal)
│   ├── car_model.py          # Modelo cinemático de bicicleta
│   ├── track_generator.py    # Gerador procedural
│   ├── track_loader.py       # Carregador YAML + allowed_tracks
│   ├── track_splits.py       # Splits canónicos train/test
│   ├── cone_sensor.py        # Perceção de cones
│   └── renderer.py           # Visualização PyGame
├── results/                  # Tabelas LaTeX, CSV per-track, figuras
│   ├── aggregated/           # Resumos agregados (CSV + LaTeX)
│   ├── perception_*/ · aug_matrix/   # Robustez de perceção + matriz de augmentation (CSV)
│   └── figures/              # Figuras do relatório/notebook (PDF + PNG)
├── runs/                     # Checkpoints e logs TensorBoard
├── scripts/                  # eval_per_track, plot, mirror, make_report_figures, ...
├── tracks/                   # Pistas oficiais FS-AI (.yaml)
│   └── mirrored/             # Versões espelhadas em Y
├── .gitignore
├── evaluate.py               # Avaliação + visualização PyGame
├── train.py                  # Script principal de treino
├── requirements.txt
└── README.md
```

---

## 🚀 Reprodução

### Pré-requisitos

- **Python ≥ 3.10** (via Anaconda/Miniconda recomendado)
- GPU NVIDIA (testado em RTX 5070 Ti)

### Setup

```bash
git clone https://github.com/pedroreis2468/AR2.git
cd AR2
pip install -r requirements.txt
```

### Testar ambiente (agente aleatório)

```bash
python evaluate.py --random
```

### Treino

```bash
# SAC baseline — 3 seeds × 3M steps
python train.py --mode sb3 --algo sac --total-steps 3000000 \
    --split train --seed 0 --n-envs 4 --device cuda --run-name sac_seed0

# PPO baseline — 3 seeds × 6M steps
python train.py --mode sb3 --algo ppo --total-steps 6000000 \
    --split train --seed 0 --n-envs 8 --device cuda --run-name ppo_seed0

# SAC com mirror augmentation (resolve viés direcional)
python train.py --mode sb3 --algo sac --total-steps 3000000 \
    --split train --seed 0 --n-envs 4 --device cuda --mirror-augment \
    --run-name sac_mirror_aug

# Exemplo de ablação (sem domain randomization)
python train.py --mode sb3 --algo sac --total-steps 2000000 \
    --split train --seed 0 --n-envs 4 --device cuda --no-dr \
    --run-name abl_sac_no_dr
```

### Avaliação

```bash
# Visualização PyGame numa pista específica
python evaluate.py --mode sb3 \
    --model runs/sac_seed0/best/best_model.zip \
    --track FSG24 --n-episodes 3

# Avaliação per-track (CSV + tabela LaTeX) no test split
python scripts/eval_per_track.py \
    --model runs/sac_seed2/best/best_model.zip --algo sac \
    --split test --n-episodes 10 \
    --output-dir results/sac_seed2

# Avaliação em pistas espelhadas (held-out)
python scripts/eval_per_track.py \
    --model runs/sac_seed2/best/best_model.zip --algo sac \
    --tracks-dir tracks/mirrored \
    --tracks-list mirrored_FSG24 mirrored_FSI24 mirrored_FSCZ24 mirrored_FSE24 mirrored_FSS22_V2 \
    --n-episodes 10 \
    --output-dir results/sac_seed2_mirrored

# Curvas de aprendizagem agregadas
python scripts/plot_learning_curves.py \
    --sac-runs runs/sac_seed0 runs/sac_seed1 runs/sac_seed2 \
    --ppo-runs runs/ppo_seed0 runs/ppo_seed1 runs/ppo_seed2 \
    --output results/learning_curves.pdf --smooth 5
```

### Monitorização

```bash
tensorboard --logdir runs/
```

### Testes

```bash
pytest -q     # suite unitária: ambiente, modelo de veículo, splits (16 testes)
```

### Debug

```bash
python scripts/debug_reward.py --steps 300   # decomposição da reward
python scripts/test_steering.py              # teste de inferência
```

---

## 🔍 Findings principais

### 1. SAC vs PPO: trade-off velocidade vs segurança

Embora SAC e PPO apresentem lap rates comparáveis no test split, **SAC é claramente mais conservador** (2.4 vs 4.8 cones derrubados em média). PPO é ligeiramente mais rápido mas mais agressivo.

### 2. FOV do sensor tem um sweet spot (em qualidade)

Tanto FOV reduzido (90°) como super-amplo (360°) **sobem o lap rate** face ao default (180°: 60% → 84%), mas **à custa de muitos mais cones derrubados** (9.2 e 17.4 vs 3.2). O *sweet spot* de 180° é em **qualidade/segurança**, não em *lap rate* bruto — um trade-off perceção-aprendizagem que **não é monotónico**.

### 3. Viés direcional sistemático

Avaliando os modelos em **pistas espelhadas em Y** (mesma geometria, direção invertida), verificámos:

- SAC cai de 66% → 40% (média 3 seeds, **−26pp**)
- PPO cai de 68% → 45% (média 3 seeds, **−23pp**)
- Padrão **bimodal** no SAC (10% ou 100% por pista)
- Padrão **degradação uniforme** no PPO

Isto sugere que o test split FS-AI tradicional **subestima o overfit estrutural** dos modelos.

### 4. Mirror augmentation elimina o viés

Treinando 1 seed SAC com **espelhamento Y aleatório (50%) em cada reset**, obtemos:

| Métrica          | SAC baseline | SAC + Mirror Aug. |
|------------------|-------------:|------------------:|
| Train lap rate   | (não medido) | **100%**          |
| Test lap rate    | 66%          | **72%** (+6pp)    |
| Mirror lap rate  | 40%          | **74%** (+34pp)   |
| Cones (test)     | 2.4          | **2.9**           |

O modelo aumentado é **igualmente bom** em pistas originais e espelhadas, e até **melhora** o desempenho em ambas. Isto demonstra que o problema era estrutural ao treino, não ao algoritmo.

### 5. Robustez de perceção: o *flicker* não dói, os cones em falta sim

Sondámos a perceção de **duas** formas — e dão resultados **opostos**, o que é o achado mais interessante desta análise:

- **Falha transiente** (cada cone tem probabilidade de não ser detetado *por frame*, via `detection_prob`): **tolerada**. A ≤10% de deteções perdidas o lap rate mantém-se dentro do ruído amostral — um cone falhado num frame reaparece no seguinte, e a política média esse ruído ao longo dos steps.
- **Cones fisicamente em falta** (remoção **persistente** de uma fração dos cones, mantendo a *centerline* intacta): **colapso**. Já a 5–10% o lap rate desaba.

| Cones perdidos (a 10%) | Transiente (*flicker*) | Persistente (em falta) |
|------------------------|-----------------------:|-----------------------:|
| **SAC**                | 63%                    | **17%**                |
| **PPO**                | 62%                    | **10%**                |
| **SAC + Mirror Aug.**  | 80%                    | **30%**                |

**Não é *quantos* cones se perdem — é *por quanto tempo*.** Um cone falhado num frame é ruído de curto prazo; um cone que desaparece deixa um **buraco de informação** que não há como preencher. Mantêm-se os padrões das secções anteriores: **SAC degrada-se melhor que PPO**, e o **`SAC + Mirror Aug.` é o mais robusto** em ambos os regimes.

### 6. Augmentation dirigida corrige as fraquezas — e compõe-se

As fraquezas dos achados 3 e 5 são **estruturais ao treino**, não ao algoritmo: corrigem-se com *data augmentation* dirigida, **sem mexer na recompensa**. Treinando o design factorial (SAC 3M, 1 seed por augmentation) e avaliando nos dois *probes*:

| Modelo | Test normal | Espelhado | Cones em falta (10%) |
|--------|------------:|----------:|---------------------:|
| Base        | 63 % | 39 % | 17 % |
| + Mirror    | 86 % | **82 %** | 30 % |
| + Dropout   | 62 % | 60 % | 28 % |
| **+ Ambos** | 82 % | **78 %** | **48 %** |

Cada augmentation **arruma sobretudo o seu eixo** (mirror→espelhado, dropout→cones em falta), e **juntá-las é super-aditivo nos cones em falta (48 %, acima dos 30/28 individuais)** — sem custo no *baseline* (até sobe). O `--cone-dropout-augment` remove 0–15 % dos cones físicos a cada *reset*, mantendo a *centerline* intacta. **Ressalva:** nem o `+Ambos` salva o regime catastrófico (≥30 % removido) — a política é reativa, sem memória; o caminho aí seria dar-lhe memória (*LSTM* / *frame-stacking*).

---

## 🧪 Trabalho Exploratório: PacSim (sim-to-sim)

Em paralelo, explorámos a integração com o **PacSim** (simulador 3D ROS 2 da Formula Student) para investigar *sim-to-sim transfer* de políticas treinadas no ambiente 2D. O setup encontra-se em:

- `scripts/run_pacsim_viz.sh` — RViz2 helper
- `scripts/test_pacsim_env.py` — bridge ambiente → ROS 2 topics
- `scripts/test_pacsim_live.py` — teste em pacote PacSim em execução
- `docs/pacsim_setup.md` — notas de instalação

---

## 📚 Referências

- Kabzan, J. et al. (2020). *AMZ Driverless: The Full Autonomous Racing System*. J. Field Robotics, 37(7).
- Haarnoja, T. et al. (2018). *Soft Actor-Critic Algorithms and Applications*. arXiv:1812.05905.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Kong, J. et al. (2015). *Kinematic and Dynamic Vehicle Models for Autonomous Driving*. IEEE IV.
- Tobin, J. et al. (2017). *Domain Randomization for Sim-to-Real Transfer*. IEEE/RSJ IROS.
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

## 👥 Grupo 1

| Nº       | Nome                                | Email                        |
|----------|-------------------------------------|------------------------------|
| PG60390  | Luís Miguel Pereira Silva           | pg60390@alunos.uminho.pt     |
| PG59908  | Pedro Miguel S. A. Urbano dos Reis  | pg59908@alunos.uminho.pt     |

---

## 📜 Licença

Trabalho de cariz estritamente académico. Universidade do Minho, Escola de Engenharia, Departamento de Informática.
