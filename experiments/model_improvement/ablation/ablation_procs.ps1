# Helper for watchdog_ablation.sh: list running ablation jobs, or stop one job's python processes.
#   powershell -NoProfile -File ablation_procs.ps1 -Mode list
#   powershell -NoProfile -File ablation_procs.ps1 -Mode kill -Job B7_seed_42
param([string]$Mode = 'list', [string]$Job = '')
$procs = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_ablation\.py --arm (B\d+|LITE) --seed \d+' })
if ($Mode -eq 'list') {
    $procs | ForEach-Object { if ($_.CommandLine -match '--arm (B\d+|LITE) --seed (\d+)') { '{0}_seed_{1}' -f $Matches[1], $Matches[2] } } | Sort-Object -Unique
} elseif ($Mode -eq 'kill') {
    if ($Job -notmatch '^(B\d+|LITE)_seed_(\d+)$') { 'bad job name'; exit 2 }
    $pattern = '--arm {0} --seed {1}(\s|$)' -f $Matches[1], $Matches[2]
    $procs | Where-Object { $_.CommandLine -match $pattern } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        'stopped pid {0}' -f $_.ProcessId
    }
}
