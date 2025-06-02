from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="./models/mistral",
    local_dir_use_symlinks=False  # Avoids symlink issues on Windows
)