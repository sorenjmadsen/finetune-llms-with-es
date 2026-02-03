import torch
from transformers import AutoModelForCausalLM  # or AutoModel, AutoModelForSeq2SeqLM, etc.

def load_self_weights_from_disk(base_model_name_or_path: str, weights_path: str, *, device="cuda"):
    # 1) Build base model (must match the architecture you fine-tuned)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype="auto",
        device_map=None,  # keep it simple; move after loading
    )

    # 2) Load your saved tensors
    sd = torch.load(weights_path, map_location="cpu")

    # 3) Handle common DDP prefix just in case (your save code doesn't add it, but harmless)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # 4) Load (strict=False prevents minor head/vocab mismatches from hard-crashing)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        print(f"[load] Missing keys (not found in .pth): {len(missing)}")
        print("  e.g.", missing[:20])
    if unexpected:
        print(f"[load] Unexpected keys (in .pth but not in model): {len(unexpected)}")
        print("  e.g.", unexpected[:20])

    # 5) Move to device after loading
    if device is not None:
        model.to(device)

    model.eval()
    return model
