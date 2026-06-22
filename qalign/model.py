"""
Model loading — the single point of ms-swift coupling.

``load(cfg)`` returns ``(model, template, tokenizer)`` for any swift-registered VL
model. Everything model-specific (the path, the swift ``model_type`` and
``template`` names, the thinking toggle, max length) comes from the config, so
switching backbones is a YAML edit. swift/torch are imported here lazily.
"""


def load(cfg, model_path=None, load_model=True):
    """Load (model, template, tokenizer) per the config.

    Args:
        cfg: a qalign.config.Config.
        model_path: override cfg.model.path (e.g. a checkpoint dir to evaluate).
        load_model: False to load only the processor/template (rarely needed).
    """
    from swift import get_model_processor, get_template

    m = cfg.model
    path = model_path or m.path
    if not path:
        raise ValueError("model path is empty — set model.path in the config "
                         "(or pass --model)")

    model, processor = get_model_processor(path, model_type=m.model_type,
                                           load_model=load_model)
    if model is not None:
        model.eval()
    template = get_template(processor, template_type=m.template,
                            max_length=m.max_length,
                            enable_thinking=m.enable_thinking)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return model, template, tok
