# Instalação - Formula Student RL Racing

## Requisitos

- Python 3.10+
- pip

## Setup

```bash
cd fs_racing_rl

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Para PyTorch CPU-only (mais leve):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Testar

```bash
# Verificar que tudo funciona com agente aleatório
python evaluate.py --random

# Treino rápido (SAC custom)
python train.py --mode custom --total-steps 500000

# Treino com Stable-Baselines3
python train.py --mode sb3 --algo sac --total-steps 1000000

# Treino PPO (baseline)
python train.py --mode sb3 --algo ppo --total-steps 1000000

# Avaliar modelo treinado
python evaluate.py --model runs/<pasta>/best_model.pt --mode custom
```

## Estrutura

```
fs_racing_rl/
├── env/
│   ├── car_model.py           # Modelo cinemático de bicicleta
│   ├── track_generator.py     # Gerador procedural de pistas FS
│   ├── cone_sensor.py         # Simulação de perceção (cones)
│   ├── racing_env.py          # Ambiente Gymnasium (FSRacing-v0)
│   └── renderer.py            # Visualização PyGame
├── agent/
│   ├── sac.py                 # SAC (Soft Actor-Critic)
│   ├── networks.py            # Redes Actor-Critic (PyTorch)
│   └── replay_buffer.py       # Experience replay buffer
├── train.py                   # Script de treino
├── evaluate.py                # Avaliação e visualização
└── requirements.txt
```
