param(
    [switch]$DryRun,
    [switch]$Status,
    [switch]$NewRun,
    [ValidateRange(1,2)][int]$MaxRounds = 2,
    [ValidateRange(1,28)][int]$MaxCalls = 12,
    [ValidateRange(1,600)][int]$TimeoutSeconds = 240,
    [ValidateRange(1,3600)][int]$MaxSeconds = 1200
)
$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction Stop
# Refresh user-scoped keys if this terminal predates their installation. Never print values.
foreach ($name in @('GEMINI_API_KEY','GOOGLE_API_KEY')) {
    if (-not [Environment]::GetEnvironmentVariable($name,'Process')) {
        $value = [Environment]::GetEnvironmentVariable($name,'User')
        if ($value) { [Environment]::SetEnvironmentVariable($name,$value,'Process') }
    }
}
$argsList = @((Join-Path $PSScriptRoot 'scripts/collab.py'), '--max-rounds', "$MaxRounds",
    '--max-calls', "$MaxCalls", '--timeout', "$TimeoutSeconds", '--max-seconds', "$MaxSeconds")
if ($DryRun) { $argsList += '--dry-run' }
if ($Status) { $argsList += '--status' }
if ($NewRun) { $argsList += '--new-run' }
& $python.Source @argsList
exit $LASTEXITCODE

