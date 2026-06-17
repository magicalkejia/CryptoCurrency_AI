$ErrorActionPreference = "Stop"

$profileDir = Join-Path (Resolve-Path ".").Path "data_storage\browser_profiles\chrome_cdp"
Write-Host $profileDir
if (Test-Path $profileDir) {
    Get-ChildItem -Force $profileDir | Select-Object Name, Mode, LastWriteTime | Format-Table -AutoSize
} else {
    Write-Host "Profile directory does not exist yet."
}
