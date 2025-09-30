from huggingface_hub import snapshot_download

# Download the full snapshot to a local directory
snapshot_download(
    repo_id="danhtran2mind/Viet-Glow-TTS-finetuning",
    local_dir="./viet-glow-tts",  # Optional: Specify where to save (creates folder if it doesn't exist)
    local_dir_use_symlinks=False  # Ensures full files are copied, not symlinks
)