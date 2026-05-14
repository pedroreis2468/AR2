#!/usr/bin/env bash
# Arranca PacSim + rviz2 (config pre-feita em config/pacsim.rviz).
# Usa-se da raiz do AR2.
#
# Fluxo recomendado:
#   Terminal A: bash scripts/run_pacsim_viz.sh
#   Terminal B (dentro de 60 s, vê o "PacSim pronto"):
#     bash scripts/run_pacsim_viz.sh eval runs/sac_seed1/final_model.zip
#
# Notas:
#   - example.launch.py tem timeout de "primeira passagem na linha de start"
#     em 60 s; se o carro não cruzar a meta nesse tempo, fecha.
#   - /dev/shm/fastrtps_* é limpo antes do launch (estado stale do FastRTPS
#     causa SIGSEGV no pacsim_node em arranques consecutivos).

set -euo pipefail

cd "$(dirname "$0")/.."
AR2_ROOT="$PWD"
RVIZ_CFG="$AR2_ROOT/config/pacsim.rviz"
MODE="${1:-viz}"

ros_env() {
    unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE RMW_IMPLEMENTATION
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    # shellcheck disable=SC1091
    source ~/pacsim_ws/install/setup.bash
}

wait_perception() {
    local i
    for i in $(seq 1 12); do
        local r
        r=$(timeout 2 ros2 topic hz /pacsim/perception/livox_front/landmarks 2>&1 | head -3 || true)
        if echo "$r" | grep -q "average rate"; then
            echo "[viz] PacSim pronto (perceção a publicar)."
            return 0
        fi
        sleep 1
    done
    echo "[viz] PacSim não publicou em 12 s. Vê /tmp/pacsim.log." >&2
    return 1
}

# Modo "eval": apenas chama o evaluate.py com o env sourced (assume PacSim já a correr).
if [[ "$MODE" == "eval" ]]; then
    MODEL="${2:-runs/sac_seed1/final_model.zip}"
    STEPS="${3:-200}"
    ros_env
    /usr/bin/python3.12 evaluate.py \
        --model "$MODEL" --mode sb3 --env pacsim \
        --n-episodes 1 --max-steps "$STEPS" \
        --pacsim-step-timeout 1.5 --device cpu
    exit $?
fi

# Modo default: arranca PacSim + rviz2.
echo "[viz] A limpar shm do FastRTPS..."
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true

echo "[viz] A iniciar PacSim em background..."
( ros_env && ros2 launch pacsim example.launch.py ) > /tmp/pacsim.log 2>&1 &
PACSIM_PID=$!
trap 'echo "[viz] Cleanup..."; kill -INT "$PACSIM_PID" 2>/dev/null || true; sleep 1; kill -KILL "$PACSIM_PID" 2>/dev/null || true; rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true' EXIT

# Esperar pelo tópico de perceção (usa o mesmo env sourced)
( ros_env && wait_perception ) || { echo "[viz] Aborto."; exit 1; }

echo
echo "[viz] Abre noutro terminal:"
echo "    bash scripts/run_pacsim_viz.sh eval runs/sac_seed1/final_model.zip 200"
echo "[viz] (ou diretamente: source ROS, depois python3.12 evaluate.py …)"
echo

echo "[viz] A iniciar rviz2 (Ctrl+C para parar TUDO)..."
ros_env
exec rviz2 -d "$RVIZ_CFG"
