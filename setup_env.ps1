# setup_env.ps1
# Creates a local Python venv and installs all dependencies.
# Run from C:\AI\audio_ml:
#   .\setup_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== Audio ML — Environment Setup ===" -ForegroundColor Cyan

# Create venv
if (-Not (Test-Path "venv")) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "`nvenv already exists, skipping creation." -ForegroundColor Green
}

$pip = ".\venv\Scripts\pip.exe"
$py  = ".\venv\Scripts\python.exe"

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
& $pip install --upgrade pip --quiet

# Install PyTorch with CUDA 12.6
Write-Host "`nInstalling PyTorch (CUDA 12.6)..." -ForegroundColor Yellow
& $pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
Write-Host "`nInstalling other dependencies..." -ForegroundColor Yellow
& $pip install librosa soundfile numpy matplotlib tqdm jupyter ipykernel

# Register Jupyter kernel
Write-Host "`nRegistering Jupyter kernel..." -ForegroundColor Yellow
& $py -m ipykernel install --user --name audio_ml --display-name "Python (audio_ml)"

# Verify
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
& $py -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
& $py -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"
& $py -c "import librosa; print('librosa:', librosa.__version__)"

Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
Write-Host "Activate with:  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Then run:       python download_data.py" -ForegroundColor Cyan
