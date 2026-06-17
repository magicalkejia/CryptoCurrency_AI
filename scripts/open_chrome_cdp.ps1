$ErrorActionPreference = "Stop"

$candidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)

$chrome = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) {
    throw "Chrome executable not found. Please edit scripts/open_chrome_cdp.ps1 and set the Chrome path manually."
}

$profileDir = Join-Path (Resolve-Path ".").Path "data_storage\browser_profiles\chrome_cdp"
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null

Write-Host "Starting Chrome with CDP on http://127.0.0.1:9222"
Write-Host "Profile: $profileDir"
Write-Host "Log in to CoinDesk in this Chrome window, then run: task coindesk_archive"

$arguments = @(
    "--remote-debugging-port=9222",
    "--user-data-dir=`"$profileDir`"",
    "https://www.coindesk.com/"
)

Start-Process -FilePath $chrome -ArgumentList $arguments

Start-Sleep -Seconds 2
try {
    $version = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:9222/json/version" -TimeoutSec 3
    Write-Host "Chrome CDP is ready."
    Write-Host $version.Content
} catch {
    Write-Host "[WARN] Chrome started, but CDP is not reachable at http://127.0.0.1:9222"
    Write-Host "Close all Chrome windows/processes, then run: task chrome_cdp"
}
