# =============================================================================
# setup_automation_windows.ps1
# Sets up automated daily pipeline runs on Windows
# =============================================================================

Write-Host "🤖 Setting up ML Pipeline Automation (Windows)" -ForegroundColor Cyan
Write-Host ""

# Get the project directory
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Project directory: $ProjectDir"

# Task Scheduler settings
$TaskName = "ML_Pipeline_Daily_Run"
$TaskDescription = "Automated ML pipeline for binary classification"
$TriggerTime = "2:00AM"

# Python executable path (adjust if needed)
$PythonPath = "python"  # or full path: "C:\Python310\python.exe"

# Script to run
$ScriptPath = Join-Path $ProjectDir "auto_pipeline.py"

# Log directory
$LogDir = Join-Path $ProjectDir "logs\cron"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Check if task already exists
$TaskExists = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($TaskExists) {
    Write-Host "⚠️  Task already exists. Removing old task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create the action
$Action = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument "$ScriptPath --source api" `
    -WorkingDirectory $ProjectDir

# Create the trigger (daily at 2 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At $TriggerTime

# Create task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

# Register the task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Description $TaskDescription `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -RunLevel Highest

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ AUTOMATION SETUP COMPLETE" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "What happens now:"
Write-Host "  ✓ Every day at 2:00 AM, the pipeline will:" -ForegroundColor White
Write-Host "    1. Fetch live data from your API/database"
Write-Host "    2. Run full pipeline (clean, validate, train)"
Write-Host "    3. Auto-deploy if model performance is good"
Write-Host "    4. Email you ONLY if something fails"
Write-Host ""
Write-Host "To test immediately:" -ForegroundColor Yellow
Write-Host "  python auto_pipeline.py --source api" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view scheduled tasks:" -ForegroundColor Yellow
Write-Host "  taskschd.msc  (opens Task Scheduler)" -ForegroundColor Cyan
Write-Host ""
Write-Host "To remove automation:" -ForegroundColor Yellow
Write-Host "  Unregister-ScheduledTask -TaskName `"$TaskName`"" -ForegroundColor Cyan
Write-Host ""
