param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$seriesPath = Join-Path $root 'collab/series/20261012-012214/ledger.json'
$launcher = Join-Path $root 'run-collab.ps1'
$ledger = Get-Content -Raw -LiteralPath $seriesPath | ConvertFrom-Json -AsHashtable
$completed = [int]$ledger.completed_rounds
$remaining = [int]$ledger.requested_rounds - $completed

if ($remaining -le 0) {
    Write-Output "Series already complete: $completed/$($ledger.requested_rounds) rounds."
    exit 0
}

Write-Output "Series: $completed/$($ledger.requested_rounds) rounds complete; $remaining remain."
if ($DryRun) { exit 0 }

for ($i = 0; $i -lt $remaining; $i++) {
    $latestPath = Join-Path $root 'collab/latest.json'
    $latestId = (Get-Content -Raw -LiteralPath $latestPath | ConvertFrom-Json).id
    $latestFolder = Join-Path $root "collab/runs/$latestId"
    $latestState = Get-Content -Raw -LiteralPath (Join-Path $latestFolder 'state.json') | ConvertFrom-Json -AsHashtable
    if ($latestState.status -and $latestState.status -ne 'COMPLETED') {
        $failedSynthesis = $latestState.tasks['synthesis-gemini']
        if ($failedSynthesis -and $failedSynthesis.status -in @('FAILED', 'MEMBER_BUDGET_EXHAUSTED') `
                -and $failedSynthesis.attempts -ge 4) {
            $runOutput = & $launcher -ExtendBudget -UnlimitedBudget -ResumeRunId $latestId `
                -RepairGeminiSynthesis -MaxRounds 1 -TimeoutSeconds 600 2>&1
        } else {
            $runOutput = & $launcher -ExtendBudget -UnlimitedBudget -ResumeRunId $latestId `
                -MaxRounds 1 -TimeoutSeconds 600 2>&1
        }
    } else {
        $runOutput = & $launcher -ContinueLatest -UnlimitedBudget -MaxRounds 1 -TimeoutSeconds 600 2>&1
    }
    $runOutput | ForEach-Object { Write-Output $_ }
    if ($LASTEXITCODE -notin @(0, 2)) {
        throw "Controller failed with exit code $LASTEXITCODE; stopping series without starting another round."
    }

    $latest = Get-Content -Raw -LiteralPath (Join-Path $root 'collab/latest.json') | ConvertFrom-Json
    $runId = $latest.id
    $runFolder = Join-Path $root "collab/runs/$runId"
    $state = Get-Content -Raw -LiteralPath (Join-Path $runFolder 'state.json') | ConvertFrom-Json -AsHashtable
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $runFolder 'manifest.json') | ConvertFrom-Json -AsHashtable
    $ledger = Get-Content -Raw -LiteralPath $seriesPath | ConvertFrom-Json -AsHashtable
    $existing = @($ledger.rounds | Where-Object { $_.run_id -eq $runId })
    $oldCalls = if ($existing.Count) { [double]$existing[0].calls } else { 0.0 }
    $oldClaudeCost = if ($existing.Count) { [double]$existing[0].member_stats.claude.cost_usd } else { 0.0 }
    $callDelta = [double]$state.calls - $oldCalls
    $claudeCostDelta = [double]$state.member_stats.claude.cost_usd - $oldClaudeCost

    $entry = @{
        sequence = if ($existing.Count) { [int]$existing[0].sequence } else { $ledger.rounds.Count + 1 }
        run_id = $runId
        continuation_from = $state.continuation_from
        continuation_round = $state.continuation_round
        status = $state.status
        calls = [double]$state.calls
        budget_mode = if ($manifest.signature.unlimited_budget) { 'unlimited' } else { 'bounded' }
        member_stats = $state.member_stats
    }
    if ($existing.Count) {
        $ledger.rounds = @($ledger.rounds | Where-Object { $_.run_id -ne $runId }) + @($entry)
    } else {
        $ledger.rounds += $entry
    }
    $ledger.total_cli_launches = [double]$ledger.total_cli_launches + $callDelta
    $ledger.reported_claude_cost_usd = [double]$ledger.reported_claude_cost_usd + $claudeCostDelta
    $ledger.completed_rounds = @($ledger.rounds | Where-Object { $_.status -eq 'COMPLETED' }).Count
    $ledger.stopped_after = $ledger.rounds.Count
    $ledger.updated = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    $ledger.stop_reason = if ($ledger.completed_rounds -ge $ledger.requested_rounds) {
        "All $($ledger.requested_rounds) requested continuation rounds are complete."
    } elseif ($state.status -eq 'COMPLETED') {
        "$($ledger.completed_rounds) of $($ledger.requested_rounds) requested rounds are complete."
    } else {
        "Stopped after run $runId ended with status $($state.status); inspect that run before continuing."
    }

    $tempPath = "$seriesPath.tmp"
    $ledger | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $tempPath -Encoding utf8
    Move-Item -LiteralPath $tempPath -Destination $seriesPath -Force
    Write-Output "Recorded round $($state.continuation_round): $($state.status); series $($ledger.completed_rounds)/$($ledger.requested_rounds)."

    if ($state.status -ne 'COMPLETED') {
        throw "Run $runId did not complete; stopping before launching another continuation."
    }
}

Write-Output "Series complete: $($ledger.completed_rounds)/$($ledger.requested_rounds) rounds."
