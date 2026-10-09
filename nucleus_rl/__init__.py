"""Small, auditable components for the nucleus-prompt RL pilot."""

__all__ = [
    "ObjectPrompt", "ParseResult", "evaluate_prediction", "foreground_iou",
    "instance_metrics", "masks_to_instances", "parse_completion",
]


def __getattr__(name):
    # Keep command-line help usable before installing numerical/GPU packages.
    if name in __all__:
        from . import rewards
        return getattr(rewards, name)
    raise AttributeError(name)
