from .motus_policy import MotusPolicy


def get_model(cfg, torch_dtype=None):
    return MotusPolicy(cfg, torch_dtype=torch_dtype)
