"""
Train/test splits canónicos para as pistas YAML Formula Student.

Estes splits são FIXOS e devem ser referenciados consistentemente pelo treino,
avaliação e relatório. Não alterar sem actualizar também o relatório/avaliação.

Disponíveis: 14 pistas circuito (excluindo skidpad/acceleration/gripMap).
Treino: 9 pistas (variedade de organizações: FSG, FSE, FSO, FSS, FSE_test).
Teste : 5 pistas mantidas COMPLETAMENTE inéditas durante o treino.
"""

# 9 pistas para treino (variadas em geometria e organização)
TRAIN_TRACKS = [
    'FSG19',
    'FSG21',
    'FSG23',
    'FSE22',
    'FSE22_test',
    'FSE23',
    'FSO20',
    'FSS19',
    'FSS22_V1',
]

# 5 pistas inéditas — apenas para avaliação final
TEST_TRACKS = [
    'FSG24',
    'FSI24',
    'FSCZ24',
    'FSE24',
    'FSS22_V2',
]

# Sanity check
_overlap = set(TRAIN_TRACKS) & set(TEST_TRACKS)
assert not _overlap, f"Overlap entre train e test: {_overlap}"
