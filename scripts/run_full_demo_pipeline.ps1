<#
.SYNOPSIS
Runs the complete synthetic demo pipeline for GeoPoverty AI on Windows.

.DESCRIPTION
This script deletes the existing database and sequentially runs all the 
scripts required to generate demo data, populate the database, train the models, 
and precompute predictions and explanations.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $ProjectRoot

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   GeoPoverty AI - Full Demo Pipeline" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Clean old database
$DbPath = Join-Path $ProjectRoot "database\geopoverty.db"
if (Test-Path $DbPath) {
    Write-Host "`n[*] Removing old database..." -ForegroundColor Yellow
    Remove-Item -Force $DbPath
}

# 2. Init database
Write-Host "`n[*] 1/5 Initializing database..." -ForegroundColor Green
python database/init_db.py

# 3. Generate synthetic data
Write-Host "`n[*] 2/5 Generating synthetic dataset (GeoJSON boundaries)..." -ForegroundColor Green
python scripts/generate_demo_data.py

# 4. Load demo data
Write-Host "`n[*] 3/5 Loading demo data into database..." -ForegroundColor Green
python scripts/load_demo_to_db.py

# 5. Train Models
Write-Host "`n[*] 4/5 Training ML Models..." -ForegroundColor Green
python training/train_models.py --data data/demo/demo_training_data.csv

# 6. Precompute predictions and SHAP/Grad-CAM
Write-Host "`n[*] 5/5 Precomputing predictions & explanations..." -ForegroundColor Green
python scripts/precompute_predictions.py --data data/demo/demo_training_data.csv

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "Pipeline Complete! You can now start the API:" -ForegroundColor Cyan
Write-Host "python backend/app.py" -ForegroundColor Yellow
