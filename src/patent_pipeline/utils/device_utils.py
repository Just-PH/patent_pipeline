import os, torch

def get_device():
    env = os.getenv("DEVICE")
    if env in {"cuda","mps","cpu"}: return env
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def print_device_info():
    d = get_device()
    print(f"🧠 Using device: {d}")
    if d == "cuda":
        try: print(f"→ GPU: {torch.cuda.get_device_name(0)}")
        except Exception: pass
    elif d == "mps":
        print("→ Apple Metal (MPS)")
    else:
        print("→ CPU fallback")
    return d
