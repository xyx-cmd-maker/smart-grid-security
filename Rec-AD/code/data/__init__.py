"""Data layer: contract, reference FDIA generator, loaders."""

from data.contract import (  # noqa: F401
    ATTACK_LOAD_REDIST,
    ATTACK_NAMES,
    ATTACK_NORMAL,
    ATTACK_RANDOM,
    ATTACK_STEALTHY,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    ContractError,
    describe_contract,
    load_contract,
    save_contract,
    validate_contract,
)
from data.dataset import (  # noqa: F401
    FDIADataset,
    NormStats,
    build_loaders,
    build_spec,
    compute_norm_stats,
    loader_arrays,
)
