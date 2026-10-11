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
# A plain invocation resumes the latest run with its original budget, never a fresh one.
$latestPath = Join-Path $PSScriptRoot 'collab/latest.json'
if (-not $NewRun -and (Test-Path -LiteralPath $latestPath)) {
    $latest = Get-Content -Raw -LiteralPath $latestPath | ConvertFrom-Json
    if ($latest.id -notmatch '^\d{8}-\d{6}-\d{6}$') { throw 'Invalid run ID' }
    $manifestPath = Join-Path $PSScriptRoot "collab/runs/$($latest.id)/manifest.json"
    $saved = (Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json).signature
    if (-not $PSBoundParameters.ContainsKey('MaxRounds')) { $MaxRounds = $saved.rounds }
    if (-not $PSBoundParameters.ContainsKey('MaxCalls')) { $MaxCalls = $saved.max_calls }
    if (-not $PSBoundParameters.ContainsKey('TimeoutSeconds')) { $TimeoutSeconds = $saved.timeout }
    if (-not $PSBoundParameters.ContainsKey('MaxSeconds')) { $MaxSeconds = $saved.max_seconds }
}
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

