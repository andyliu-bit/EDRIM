#!/usr/bin/env bash
# Run setting D detached (WMI launch, survives the Claude session) once memory allows: DeepSynBa seed 13/14 training
# shares the machine and crashed before on Windows error 1455 (commit memory exhausted).
# Gate: 3 checks 2 min apart, each with free commit >= 6 GB and free physical RAM >= 2.5 GB.
cd /c/LHZ/research/bisurfgnn-model-experiments || exit 1
PS=/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
D=runs/eval_paper/interaction_level
LOG=$D/D_gate.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
ok=0; n=0
while [ $ok -lt 3 ]; do
  read -r commit phys < <($PS -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; "{0} {1}" -f [int]($o.FreeVirtualMemory/1MB*10), [int]($o.FreePhysicalMemory/1MB*10)' | tr -d '\r')
  if [ "${commit:-0}" -ge 60 ] && [ "${phys:-0}" -ge 25 ]; then ok=$((ok + 1)); else ok=0; fi
  [ $((n % 15)) -eq 0 ] && say "waiting: free commit ${commit}/10 GB, free RAM ${phys}/10 GB (ok $ok/3)"
  n=$((n + 1)); [ $ok -lt 3 ] && sleep 120
done
say "memory free (commit ${commit}/10 GB, RAM ${phys}/10 GB); setting_d.py start"
.venv/Scripts/python.exe -I experiments/eval_paper/interaction_level/setting_d.py > $D/D_console.log 2>&1
say "setting_d.py exit $?"
