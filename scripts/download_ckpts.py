from huggingface_hub import snapshot_download

# Download only best_model.pth and config.json to a local directory
snapshot_download(
    repo_id="danhtran2mind/Viet-Glow-TTS-finetuning",
    local_dir="./ckpts",
    local_dir_use_symlinks=False,
    allow_patterns=["best_model.pth", "config.json"]
)