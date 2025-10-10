# Avoid heavy imports when running mock tests without PL
try:
    from .vilt_missing_aware_prompt_module import ViLTransformerSS  # noqa: F401
except Exception:
    ViLTransformerSS = None
