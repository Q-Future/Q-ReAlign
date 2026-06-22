"""
ms-swift startup shim — loaded via ``--custom_register_path`` (so it runs during
arg-parse, before the trainer is built). It:
  1. installs the packed-image training cache hook (if the blob exists), and
  2. imports qalign.callback, which registers the ``qalign_eval`` callback.

Config (cache dir, etc.) is read from the YAML at ``$QALIGN_CFG``. Everything is
best-effort: a missing cache or swift import just falls back to normal behavior,
never breaking training.
"""
import os

try:
    from qalign.config import Config
    cfg = Config.from_yaml(os.environ["QALIGN_CFG"]) if os.environ.get("QALIGN_CFG") else Config()
    try:
        from qalign.cache import install_train_hook
        install_train_hook(cfg.paths.cache_dir)
    except Exception as e:
        print(f"[qalign-register] train cache hook off ({type(e).__name__}: {e})", flush=True)
    # importing the callback registers it into swift's callbacks_map
    import qalign.callback  # noqa: F401
except Exception as e:
    print(f"[qalign-register] shim error ({type(e).__name__}: {e})", flush=True)
