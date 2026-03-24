# AR - Aprendizagem por Reforço 🏎️

Projeto de Reinforcement Learning aplicado a condução autónoma Formula Student Driverless.

**Curso:** Sensoriação e Ambiente (SA2026) — Mestrado em IA, Universidade do Minho

## Projeto: FS Racing RL

Agente de RL (SAC / PPO) que aprende a conduzir um carro de Formula Student numa pista delimitada por cones azuis (esquerda) e amarelos (direita).

### Componentes principais

- **Ambiente 2D** com modelo cinemático de bicicleta e cones procedurais
- **Observação** de 20 dimensões (ego state + cones + boundary info)
- **SAC** (Soft Actor-Critic) como algoritmo principal, **PPO** como baseline
- **Domain randomization** para robustez (massa, atrito, ruído sensorial)
- **Visualização PyGame** em tempo real

### Quick Start

```bash
cd fs_racing_rl
pip install -r requirements.txt
python evaluate.py --random          # testar ambiente
python train.py --mode custom --total-steps 500000  # treinar SAC
```

Ver [instal.md](instal.md) para instruções detalhadas.

## Referências

- Sutton & Barto (2020) — *Reinforcement Learning: An Introduction*
- Balaji et al. (2019) — *DeepRacer: Autonomous Racing Platform* ([arXiv](https://arxiv.org/pdf/1905.05150))
- Ulrich & Wehrli (2024) — *End-to-End Deep RL for Autonomous Racing Dynamics* ([ZHAW](https://www.zhaw.ch/storage/engineering/institute-zentren/cai/studentische_arbeiten/Spring_2024/BA_FS24_Fabian_Ulrich_Tobias_Wehrli_End-to-End_Deep_Reinforcement_Learning_for_Autonomous_Racing_Dynamics.pdf))
- Haarnoja et al. (2018) — *Soft Actor-Critic Algorithms and Applications*
