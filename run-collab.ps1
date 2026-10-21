param(
    [switch]$DryRun,
    [switch]$Status,
    [switch]$NewRun,
    [switch]$ContinueLatest,
    [switch]$ExtendBudget,
    [switch]$UnlimitedBudget,
    [string]$ResumeRunId,
    [switch]$RepairGemini,
    [switch]$RepairGeminiSynthesis,
    [ValidateRange(60,900)][int]$GeminiRepairTimeout = 600,
    [ValidateRange(1,2)][int]$MaxRounds = 2,
    [ValidateRange(1,28)][int]$MaxCalls = 12,
    [ValidateRange(1,28)][int]$MemberMaxCalls = 4,
    [ValidateRange(0,3600)][int]$MemberMaxSeconds = 0,
    [ValidateRange(0,100)][double]$MemberMaxUsd = 0.0,
    [ValidateRange(1,600)][int]$TimeoutSeconds = 240,
    [ValidateRange(1,3600)][int]$MaxSeconds = 1200,
    [switch]$RotateRoles,
    [int]$RunIndex = -1,
    [switch]$SectionFailover,
    [switch]$Duo
)
$ErrorActionPreference = 'Stop'
$DuoExplicit = $PSBoundParameters.ContainsKey('Duo')
if (($NewRun -and $ContinueLatest) -or ($NewRun -and $ExtendBudget) -or ($ContinueLatest -and $ExtendBudget)) {
    throw '-NewRun, -ContinueLatest, and -ExtendBudget cannot be used together'
}
if ($DuoExplicit -and ($RepairGemini -or $RepairGeminiSynthesis -or $SectionFailover)) {
    throw '-Duo is manual-only: cannot combine with -RepairGemini* or -SectionFailover'
}
if ($DuoExplicit -and -not ($NewRun -or $ContinueLatest -or $ExtendBudget)) {
    throw '-Duo is a manual backup: use with -NewRun, -ContinueLatest, or -ExtendBudget'
}
$python = Get-Command python -ErrorAction Stop
# A plain invocation resumes the latest run with its original budget, never a fresh one.
$latestPath = Join-Path $PSScriptRoot 'collab/latest.json'
if (-not $NewRun -and -not $ResumeRunId -and (Test-Path -LiteralPath $latestPath)) {
    $latest = Get-Content -Raw -LiteralPath $latestPath | ConvertFrom-Json
    if ($latest.id -notmatch '^\d{8}-\d{6}-\d{6}$') { throw 'Invalid run ID' }
    $manifestPath = Join-Path $PSScriptRoot "collab/runs/$($latest.id)/manifest.json"
    $saved = (Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json).signature
    if (-not $PSBoundParameters.ContainsKey('MaxRounds') -and $null -ne $saved.rounds) { $MaxRounds = $saved.rounds }
    if (-not $PSBoundParameters.ContainsKey('MaxCalls') -and $null -ne $saved.max_calls) { $MaxCalls = $saved.max_calls }
    if (-not $PSBoundParameters.ContainsKey('MemberMaxCalls') -and $null -ne $saved.member_max_calls) { $MemberMaxCalls = $saved.member_max_calls }
    if (-not $PSBoundParameters.ContainsKey('TimeoutSeconds') -and $null -ne $saved.timeout) { $TimeoutSeconds = $saved.timeout }
    if (-not $PSBoundParameters.ContainsKey('MaxSeconds') -and $null -ne $saved.max_seconds) { $MaxSeconds = $saved.max_seconds }
    if ($saved.unlimited_budget) { $UnlimitedBudget = $true }
    if (-not $PSBoundParameters.ContainsKey('RotateRoles') -and $saved.rotate_roles) { $RotateRoles = [bool]$saved.rotate_roles }
    if (-not $PSBoundParameters.ContainsKey('SectionFailover') -and $saved.section_failover) { $SectionFailover = [bool]$saved.section_failover }
}
# Refresh user-scoped keys if this terminal predates their installation. Never print values.
foreach ($name in @('GEMINI_API_KEY','GOOGLE_API_KEY')) {
    if (-not [Environment]::GetEnvironmentVariable($name,'Process')) {
        $value = [Environment]::GetEnvironmentVariable($name,'User')
        if ($value) { [Environment]::SetEnvironmentVariable($name,$value,'Process') }
    }
}
$argsList = @((Join-Path $PSScriptRoot 'scripts/collab.py'), '--max-rounds', "$MaxRounds",
    '--max-calls', "$MaxCalls", '--member-max-calls', "$MemberMaxCalls",
    '--timeout', "$TimeoutSeconds", '--max-seconds', "$MaxSeconds")
if ($MemberMaxSeconds -gt 0) { $argsList += @('--member-max-seconds', "$MemberMaxSeconds") }
if ($MemberMaxUsd -gt 0) { $argsList += @('--member-max-usd', "$MemberMaxUsd") }
if ($RotateRoles) { $argsList += '--rotate-roles' }
if ($RunIndex -ge 0) { $argsList += @('--run-index', "$RunIndex") }
if ($SectionFailover) { $argsList += '--section-failover' }
if ($Duo) { $argsList += '--duo' }
if ($ContinueLatest) { $argsList += '--continue-latest' }
if ($ExtendBudget) { $argsList += '--extend-budget' }
if ($UnlimitedBudget) { $argsList += '--unlimited-budget' }
if ($ResumeRunId) {
    if (-not $ExtendBudget) { throw '-ResumeRunId requires -ExtendBudget' }
    $argsList += @('--resume-run-id', $ResumeRunId)
}
if ($RepairGemini) {
    $argsList += '--repair-gemini'
    $argsList += @('--gemini-repair-timeout', "$GeminiRepairTimeout")
}
if ($RepairGeminiSynthesis) { $argsList += '--repair-gemini-synthesis' }
if ($DryRun) { $argsList += '--dry-run' }
if ($Status) { $argsList += '--status' }
if ($NewRun) { $argsList += '--new-run' }
& $python.Source @argsList
exit $LASTEXITCODE

