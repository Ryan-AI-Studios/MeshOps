# MeshOps 0130 — discover Python 3.12 and invoke Face Landmarker sidecar.
# Emits skip JSON (exit 2) when no 3.12 interpreter is available.
# MESHOPS_FACE_LANDMARKER overrides discovery; run.py itself does not read it.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThru
)

$ErrorActionPreference = "Stop"
$Honesty = "face_landmarker_sidecar_not_mesh_or_print_success"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunPy = Join-Path $ScriptDir "run.py"

function Write-Skip([string]$Reason, [string]$Detail) {
    $outIdx = [array]::IndexOf($PassThru, "--out")
    $payload = @{
        ok      = $false
        skip    = $Reason
        honesty = $Honesty
        detail  = $Detail
    } | ConvertTo-Json -Compress
    if ($outIdx -ge 0 -and ($outIdx + 1) -lt $PassThru.Count) {
        $outPath = $PassThru[$outIdx + 1]
        $parent = Split-Path -Parent $outPath
        if ($parent) {
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
        }
        Set-Content -Path $outPath -Value $payload -Encoding utf8
    }
    else {
        Write-Output $payload
    }
    exit 2
}

$candidates = @()
if ($env:MESHOPS_FACE_LANDMARKER) {
    $candidates += $env:MESHOPS_FACE_LANDMARKER
}
$venvPy = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $candidates += $venvPy
}
$candidates += @("py -3.12", "python3.12", "python312")

$exe = $null
$prefixArgs = @()
foreach ($c in $candidates) {
    if ($c -eq "py -3.12") {
        $pyCmd = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $pyCmd) { continue }
        # Probe: py -3.12 -c "..."
        & py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $exe = "py"
            $prefixArgs = @("-3.12")
            break
        }
        continue
    }
    if (Test-Path -LiteralPath $c) {
        & $c -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $exe = $c
            break
        }
        continue
    }
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        & $cmd.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $exe = $cmd.Source
            break
        }
    }
}

if ($null -eq $exe) {
    Write-Skip "tool_missing" "no Python 3.12 interpreter (set MESHOPS_FACE_LANDMARKER or install 3.12)"
}

& $exe @prefixArgs $RunPy @PassThru
exit $LASTEXITCODE
