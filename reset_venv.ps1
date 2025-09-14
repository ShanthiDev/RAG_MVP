# reset_venv.ps1
# --------------------------
# Optional: Map a short drive letter for clean paths
# subst R: "$PSScriptRoot"
# To remove again later: subst R: /D

# Always work from repo root
Set-Location $PSScriptRoot

Write-Host "Removing old venv..."
if (Test-Path .venv) {
    Remove-Item -Recurse -Force .venv
}

Write-Host "Creating new venv (Python 3.12)..."
py -3.12 -m venv .venv

Write-Host "Activating venv..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing requirements..."
python -m pip install -r requirements.txt

Write-Host "✅ Done. Environment reset and ready."
