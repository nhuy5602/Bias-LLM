param(
    [switch]$DryRun,
    [switch]$Status,
    [switch]$NewRun,
    [switch]$RetryUnavailable,
    [ValidateRange(1,600)][int]$TimeoutSeconds = 240,
    [ValidateRange(1,86400)][int]$MaxSeconds = 14400,
    [ValidateRange(1,80)][int]$MaxNewLaunches = 80
)
$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction Stop
foreach ($name in @('GEMINI_API_KEY','GOOGLE_API_KEY')) {
    if (-not [Environment]::GetEnvironmentVariable($name,'Process')) {
        $value = [Environment]::GetEnvironmentVariable($name,'User')
        if ($value) { [Environment]::SetEnvironmentVariable($name,$value,'Process') }
    }
}
$argsList = @((Join-Path $PSScriptRoot 'scripts/semantic_label_pilot.py'),
    '--timeout', "$TimeoutSeconds", '--max-seconds', "$MaxSeconds",
    '--max-new-launches', "$MaxNewLaunches")
if ($DryRun) { $argsList += '--dry-run' }
if ($Status) { $argsList += '--status' }
if ($NewRun) { $argsList += '--new-run' }
if ($RetryUnavailable) { $argsList += '--retry-unavailable' }
& $python.Source @argsList
exit $LASTEXITCODE
