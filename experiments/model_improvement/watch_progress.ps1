# BiSurfGNN v2 补充实验进度（只读，不改动任何运行）。在任意目录的 PowerShell 里运行：
#   & 'C:\LHZ\research\bisurfgnn-model-experiments\experiments\model_improvement\watch_progress.ps1'
# 每 60 秒刷新一次；Ctrl-C 只关闭这个查看页，训练不受影响。加 -Once 只显示一次。
# 文件须保存为带 BOM 的 UTF-8，否则 Windows PowerShell 5.1 会把中文读错。
param([switch]$Once)

$V2 = (Resolve-Path (Join-Path $PSScriptRoot '..\..\runs\model_improvement\bisurfgnn_v2_dev')).Path
$Cold = Join-Path $V2 'cold_drug'
$Abl = Join-Path $V2 'ablation'

function Tail-Lines([string]$path, [int]$n) {
    if (Test-Path -LiteralPath $path) { return @(Get-Content -LiteralPath $path -Tail $n -Encoding utf8 -ErrorAction SilentlyContinue) }
    return @()
}

function New-Run([string]$name, [string]$dir, [string]$log, [string]$job = '') {
    return [pscustomobject]@{ Name = $name; Dir = $dir; Log = $log; Job = $job }
}

function Progress-Text([string]$kind, $run, [int]$max) {
    $line = (Tail-Lines $run.Log 1) -join ''
    $start = (Get-Item -LiteralPath $run.Log).CreationTime.ToString('HH:mm')
    switch ($kind) {
        'v2' {
            if ($line -match 'epoch=(\d+)') { return "第 $([int]$Matches[1] + 1)/$max 轮，开始于 $start" }
            return "准备数据中，开始于 $start"
        }
        'dd' {
            if ($line -match 'combo epoch (\d+).*val_rmse% ([\d.]+)') { return "联用网络第 $([int]$Matches[1] + 1) 轮，验证集 RMSE $($Matches[2])" }
            if ($line -match 'mono epoch (\d+)') { return "单药网络第 $([int]$Matches[1] + 1) 轮" }
            return "准备中，开始于 $start"
        }
        'dsb' {
            $csv = Get-ChildItem -LiteralPath $run.Dir -Recurse -Filter 'metrics.csv' -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($csv) {
                $epochs = @(Import-Csv -LiteralPath $csv.FullName | Where-Object { $_.epoch -ne '' } | ForEach-Object { [int]$_.epoch })
                if ($epochs.Count -gt 0) { return "第 $(($epochs | Measure-Object -Maximum).Maximum + 1)/$max 轮，开始于 $start" }
            }
            return "准备中，开始于 $start"
        }
        default { return "开始于 $start，这个任务结束时才写日志" }
    }
}

function Show-Group([string]$label, [object[]]$runs, [string]$kind, [int]$max) {
    $done = @($runs | Where-Object { Test-Path -LiteralPath (Join-Path $_.Dir 'metrics.json') })
    $pending = @($runs | Where-Object { -not (Test-Path -LiteralPath (Join-Path $_.Dir 'metrics.json')) })
    $failed = @($pending | Where-Object { (Tail-Lines $_.Log 30) -match 'Traceback|Error:' })
    $withLog = @($pending | Where-Object { (Test-Path -LiteralPath $_.Log) -and ($failed -notcontains $_) })
    $active = @($withLog | Where-Object { -not $_.Job -or ($script:Running -contains $_.Job) })
    $dead = @($withLog | Where-Object { $_.Job -and ($script:Running -notcontains $_.Job) })
    "  {0}：已完成 {1}/{2}" -f $label, $done.Count, $runs.Count
    foreach ($a in $active) {
        $text = "    {0} 正在跑，{1}" -f $a.Name, (Progress-Text $kind $a $max)
        $age = ((Get-Date) - (Get-Item -LiteralPath $a.Log).LastWriteTime).TotalMinutes
        if ($kind -ne 'xgb' -and $age -gt 20) { $text += "；注意：日志已 {0:N0} 分钟没有更新" -f $age }
        $text
    }
    if ($active.Count -eq 0 -and $dead.Count -eq 0 -and $done.Count -lt $runs.Count -and $failed.Count -eq 0) { '    排队中，还没开始' }
    if ($failed.Count -gt 0) { "    出错 {0} 个：{1}" -f $failed.Count, (($failed | ForEach-Object { $_.Name }) -join '、') }
    if ($dead.Count -gt 0) { "    已中断，进程不在了 {0} 个：{1}" -f $dead.Count, (($dead | ForEach-Object { $_.Name }) -join '、') }
}

function Stream-Status([int]$n) {
    $line = (Tail-Lines (Join-Path $Abl "queue_stream_$n.log") 1) -join ''
    if (-not $line) { return "    第 $n 路：还没有启动" }
    if ($line -match 'waiting for start condition after_xgb') { return "    第 $n 路：等新药划分的 XGBoost 全部跑完、内存够用后开始" }
    if ($line -match 'waiting for start condition after_cardD') { return "    第 $n 路：等卡 D 的 GPU 队列全部跑完后开始" }
    if ($line -match 'start condition met') { return "    第 $n 路：启动条件已满足，正在领取任务" }
    if ($line -match '(B\d+)_seed_(\d+) waits for memory \(free commit (\d+) GB\)') { return "    第 $n 路：$($Matches[1]) 种子 $($Matches[2]) 在等内存，当前可提交余量 $($Matches[3]) GB" }
    if ($line -match '(B\d+)_seed_(\d+) start') { return "    第 $n 路：正在跑 $($Matches[1]) 种子 $($Matches[2])" }
    if ($line -match '(B\d+)_seed_(\d+) exit=0') { return "    第 $n 路：$($Matches[1]) 种子 $($Matches[2]) 刚跑完，正在领取下一个" }
    if ($line -match '(B\d+)_seed_(\d+) exit=(\d+)') { return "    第 $n 路：$($Matches[1]) 种子 $($Matches[2]) 出错退出，退出码 $($Matches[3])" }
    if ($line -match 'no unclaimed jobs left') { return "    第 $n 路：已经没有剩余任务" }
    return "    第 $n 路：$line"
}

function Marker([string]$name) { return Test-Path -LiteralPath (Join-Path $V2 $name) }

while ($true) {
    if (-not $Once) { Clear-Host }
    $script:Running = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_ablation\.py --arm (B\d+) --seed (\d+)' } |
        ForEach-Object { if ($_.CommandLine -match '--arm (B\d+) --seed (\d+)') { '{0}_seed_{1}' -f $Matches[1], $Matches[2] } } | Sort-Object -Unique)
    'BiSurfGNN 补充实验进度    ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
    $gpu = (& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null) -split ',\s*'
    $os = Get-CimInstance Win32_OperatingSystem
    if ($gpu.Count -ge 2) { "GPU 占用 {0}%，显存已用 {1} MiB；内存可提交余量 {2:N1} GB" -f $gpu[0], $gpu[1], ($os.FreeVirtualMemory / 1MB) }
    ''
    '已完成'
    if (Marker 'SYNERGY_TEST_EVALUATED.marker') { '  卡 A  Ours 协同分数复查：验证集与 test 都已完成' }
    if (Marker 'COMBOKR_TEST_EVALUATED.marker') { '  卡 C2 comboKR 2.0：test 已评分' }
    if (Marker 'DDPRISM_TEST_EVALUATED.marker') { '  卡 C1 DD-PRiSM：5 个种子已训练，test 已评分' }
    ''
    if (-not (Marker 'DDPRISM_TEST_EVALUATED.marker')) {
        '卡 C1  DD-PRiSM 对比模型'
        $dd = foreach ($s in 41..45) { New-Run "种子 $s" (Join-Path $V2 "ddprism_measured_mono\seed_$s") (Join-Path $V2 "ddprism_seed_${s}_console.log") }
        Show-Group '训练（5 个种子）' $dd 'dd' 0
        ''
    }
    '卡 D   新药划分（药物在训练中没见过）'
    $x3 = foreach ($s in 41..45) { New-Run "种子 $s" (Join-Path $Cold "xgb_seed_$s") (Join-Path $Cold "xgb_seed_${s}_console.log") }
    Show-Group 'XGBoost，3000 棵上限' $x3 'xgb' 0
    $x20 = foreach ($s in 41..45) { New-Run "种子 $s" (Join-Path $Cold "xgb_cap20000_seed_$s") (Join-Path $Cold "xgb_cap20000_seed_${s}_console.log") }
    Show-Group 'XGBoost，20000 棵上限（CPU）' $x20 'xgb' 0
    $trees = @($x20 | ForEach-Object { Join-Path $_.Dir 'metrics.json' } | Where-Object { Test-Path -LiteralPath $_ } |
        ForEach-Object { (Get-Content -LiteralPath $_ -Raw -Encoding utf8 | ConvertFrom-Json).best_trees })
    if ($trees.Count -gt 0) { '    已完成种子的最佳树数：' + ($trees -join '、') + '（低于 20000 说明没有顶到上限）' }
    $v2runs = foreach ($s in 41..45) { New-Run "种子 $s" (Join-Path $Cold "v2_both_seed_$s") (Join-Path $Cold "v2_both_seed_${s}_console.log") }
    Show-Group 'Ours（5 个种子，每个约 2.3 小时）' $v2runs 'v2' 60
    $rv = foreach ($s in 41..43) { New-Run "种子 $s" (Join-Path $Cold "v2_random_seed_$s") (Join-Path $Cold "v2_random_seed_${s}_console.log") }
    Show-Group 'Ours 随机特征对照（3 个种子）' $rv 'v2' 60
    $dsb = @(New-Run '种子 10' (Join-Path $Cold 'deepsynba_seed_10') (Join-Path $Cold 'deepsynba_console.log'))
    Show-Group 'DeepSynBa（1 个种子，约 6 小时）' $dsb 'dsb' 500
    if (Marker 'COLD_TEST_SCORED.marker') { '  新药 test 评分：已完成' } else { '  新药 test 评分：等上面全部跑完后做一次' }
    ''
    '卡 B   组件消融（只用验证集，两路并行）'
    $ab = foreach ($s in 41..45) { foreach ($arm in 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10') {
        New-Run "$arm 种子 $s" (Join-Path $Abl "${arm}_seed_$s") (Join-Path $Abl "${arm}_seed_${s}_console.log") "${arm}_seed_$s" } }
    Show-Group '10 个臂 × 5 个种子（种子 44、45 为 10-20 补跑）' $ab 'v2' 60
    $disp = (Tail-Lines (Join-Path $Abl 'queue_dispatch.log') 1) -join ''
    if ($disp -match '(B\d+)_seed_(\d+) start') { "    调度器：刚启动 $($Matches[1]) 种子 $($Matches[2])" }
    elseif ($disp -match '(B\d+)_seed_(\d+) exit=0') { "    调度器：$($Matches[1]) 种子 $($Matches[2]) 已完成，等空位启动下一个" }
    elseif ($disp -match '(B\d+)_seed_(\d+) exit=(\d+)') { "    调度器：$($Matches[1]) 种子 $($Matches[2]) 出错退出，退出码 $($Matches[3])" }
    elseif ($disp -match 'all jobs done') { '    调度器：所有任务已完成' }
    elseif ($disp -match 'start:') { '    调度器：已启动，等空位' }
    elseif (-not $disp) { Stream-Status 1; Stream-Status 2 }
    $retry = (Tail-Lines (Join-Path $Abl 'queue_retry.log') 1) -join ''
    if ($retry -match 'waiting for a free stream slot for: (.+)$') { "    补跑：$($Matches[1]) 等任一路领完任务后开始" }
    elseif ($retry -match '(B\d+)_seed_(\d+) start') { "    补跑：正在跑 $($Matches[1]) 种子 $($Matches[2])" }
    elseif ($retry -match 'waits for memory') { '    补跑：在等内存' }
    elseif ($retry -match 'exit=(\d+)') { "    补跑：上一个任务已结束，退出码 $($Matches[1])" }
    elseif ($retry -match 'retry: done') { '    补跑：已完成' }
    ''
    'GPU：卡 D 已全部完成；消融两路并行，队列与看门狗以隐藏窗口运行，不随会话结束。'
    if ($Once) { break }
    'Ctrl-C 只关闭这个查看页，训练照常进行。60 秒后刷新。'
    Start-Sleep -Seconds 60
}
