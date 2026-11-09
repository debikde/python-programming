param(
    [string]$RunnerRoot = 'C:\actions-runner'
)

$ErrorActionPreference = 'Stop'
$version = '2.338.2'
$expectedHash = 'a1f17aa987fc8158bc7521d284912267a0d01b7c2c4684b33c86783b5e8221cc'
$archiveName = "actions-runner-win-x64-$version.zip"
$downloadUrl = "https://github.com/actions/runner/releases/download/v$version/$archiveName"
$runnerPath = [System.IO.Path]::GetFullPath($RunnerRoot)
$archivePath = Join-Path $runnerPath $archiveName

if (Test-Path (Join-Path $runnerPath 'config.cmd')) {
    Write-Host "Runner files already exist in $runnerPath"
    exit 0
}

if ((Test-Path $runnerPath) -and (Get-ChildItem -LiteralPath $runnerPath -Force | Select-Object -First 1)) {
    throw "Directory is not empty: $runnerPath"
}

New-Item -ItemType Directory -Path $runnerPath -Force | Out-Null
Invoke-WebRequest -Uri $downloadUrl -OutFile $archivePath
$actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) {
    throw "Runner archive checksum does not match the official release: $archivePath"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($archivePath, $runnerPath)
Remove-Item -LiteralPath $archivePath
Write-Host "Runner files are ready in $runnerPath"
Write-Host 'Register it only on an isolated Windows machine with a fresh GitHub token and the gait-lab label.'
