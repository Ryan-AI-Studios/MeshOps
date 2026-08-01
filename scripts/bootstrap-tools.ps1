#Requires -Version 5.1
<#
.SYNOPSIS
  Mirror-aware portable Blender 5.2.0 bootstrap for MeshOps (Difficulty section 4).

.DESCRIPTION
  Downloads blender-5.2.0-windows-x64.zip from non-CDN mirrors first
  (RWTH -> dotsrc -> cicku/iu13 -> download.blender.org last), verifies SHA-256
  against the pinned hash, extracts to a well-known portable layout, and sets
  MESHOPS_BLENDER (User scope + current session). Never uses setx.

  Requires a repo clone (this script is not shipped in the wheel).

.PARAMETER SkipDownload
  Print plan only; no network (alias for -WhatIf behavior).

.PARAMETER WhatIf
  Print plan only; no network.

.PARAMETER MirrorUrl
  Force a single zip URL (overrides mirror list; still SHA-verified).

.PARAMETER InstallDir
  Tools root (default: %LOCALAPPDATA%\MeshOps\tools or MESHOPS_BOOTSTRAP_DIR).

.PARAMETER Interactive
  Optional prompt menu (not required for track exit). Blender is always the
  install target; Orca is always hint-only (no separate -BlenderOnly switch).
#>
[CmdletBinding(SupportsShouldProcess = $false)]
param(
    [switch]$SkipDownload,
    [switch]$WhatIf,
    [string]$MirrorUrl = "",
    [string]$InstallDir = "",
    [switch]$Interactive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Pins (must match src/meshops/ops/mirrors.py) ---
$BlenderVersion = "5.2.0"
$ZipName = "blender-5.2.0-windows-x64.zip"
$Sha256FileName = "blender-5.2.0.sha256"
$PinnedSha256 = "2d184b626c001692c362291911293b6a297179d618d95e9e9192c3a80318adc4"
$PortableDirName = "blender-5.2.0"
$ZipNestedDirName = "blender-5.2.0-windows-x64"
$ExeName = "blender.exe"
$ReleaseRel = "release/Blender5.2/$ZipName"

$MirrorList = @(
    "https://ftp.halifax.rwth-aachen.de/blender/$ReleaseRel",
    "https://mirrors.dotsrc.org/blender/$ReleaseRel",
    "https://mirror.cicku.me/blender/$ReleaseRel",
    "https://mirrors.iu13.net/blender/$ReleaseRel",
    "https://download.blender.org/$ReleaseRel"
)

$ShortTimeoutSec = 15
$DefaultTimeoutSec = 600
$UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 MeshOps-bootstrap/1.0"

function Write-Info([string]$Message) {
    Write-Host "[bootstrap] $Message"
}

function Write-Warn([string]$Message) {
    Write-Warning "[bootstrap] $Message"
}

function Get-ToolsRoot {
    if ($InstallDir -and $InstallDir.Trim().Length -gt 0) {
        return [System.IO.Path]::GetFullPath($InstallDir.Trim())
    }
    if ($env:MESHOPS_BOOTSTRAP_DIR -and $env:MESHOPS_BOOTSTRAP_DIR.Trim().Length -gt 0) {
        return [System.IO.Path]::GetFullPath($env:MESHOPS_BOOTSTRAP_DIR.Trim())
    }
    $local = $env:LOCALAPPDATA
    if (-not $local) {
        $local = Join-Path $env:USERPROFILE "AppData\Local"
    }
    return Join-Path $local "MeshOps\tools"
}

function Get-MirrorCandidates {
    $list = New-Object System.Collections.Generic.List[string]
    if ($MirrorUrl -and $MirrorUrl.Trim().Length -gt 0) {
        $list.Add($MirrorUrl.Trim())
        return $list
    }
    if ($env:MESHOPS_BLENDER_MIRROR -and $env:MESHOPS_BLENDER_MIRROR.Trim().Length -gt 0) {
        $list.Add($env:MESHOPS_BLENDER_MIRROR.Trim())
    }
    foreach ($m in $MirrorList) {
        if (-not $list.Contains($m)) {
            $list.Add($m)
        }
    }
    return $list
}

function Get-ChecksumUrl([string]$ZipUrl) {
    if ($ZipUrl.EndsWith($ZipName)) {
        return $ZipUrl.Substring(0, $ZipUrl.Length - $ZipName.Length) + $Sha256FileName
    }
    $idx = $ZipUrl.LastIndexOf("/")
    if ($idx -lt 0) { return $Sha256FileName }
    return $ZipUrl.Substring(0, $idx + 1) + $Sha256FileName
}

function Test-IsOfficialMirror([string]$Url) {
    return $Url -match "download\.blender\.org"
}

function Get-ConnectTimeoutSec([string]$Url) {
    if ($Url -match "cicku\.me|iu13\.net") {
        return $ShortTimeoutSec
    }
    return $DefaultTimeoutSec
}

function Invoke-DownloadFile {
    param(
        [string]$Url,
        [string]$OutFile,
        [int]$TimeoutSec
    )
    $ProgressPreference = "SilentlyContinue"
    $oldPref = $ProgressPreference
    try {
        $ProgressPreference = "SilentlyContinue"
        # Prefer BITS-free Invoke-WebRequest with UA; fall back to WebClient
        $headers = @{ "User-Agent" = $UserAgent }
        Invoke-WebRequest -Uri $Url -OutFile $OutFile -Headers $headers -TimeoutSec $TimeoutSec -UseBasicParsing
    }
    finally {
        $ProgressPreference = $oldPref
    }
}

function Get-FileSha256([string]$Path) {
    $hash = Get-FileHash -Path $Path -Algorithm SHA256
    return $hash.Hash.ToLowerInvariant()
}

function Expand-BlenderZip {
    param(
        [string]$ZipPath,
        [string]$ToolsRoot,
        [string]$FinalDir
    )
    $staging = Join-Path $ToolsRoot "_extract_staging"
    if (Test-Path $staging) {
        Remove-Item -Recurse -Force $staging
    }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $staging -Force

    # Zip usually nests blender-5.2.0-windows-x64\
    $nested = Join-Path $staging $ZipNestedDirName
    $sourceDir = $null
    if (Test-Path (Join-Path $nested $ExeName)) {
        $sourceDir = $nested
    }
    else {
        $found = Get-ChildItem -Path $staging -Recurse -Filter $ExeName -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) {
            $sourceDir = $found.DirectoryName
        }
    }
    if (-not $sourceDir) {
        throw "Could not find $ExeName inside extracted zip"
    }

    if (Test-Path $FinalDir) {
        Remove-Item -Recurse -Force $FinalDir
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $FinalDir -Parent) | Out-Null
    Move-Item -Path $sourceDir -Destination $FinalDir
    Remove-Item -Recurse -Force $staging -ErrorAction SilentlyContinue

    $exe = Join-Path $FinalDir $ExeName
    if (-not (Test-Path $exe)) {
        throw "Extract layout missing $exe"
    }
    return $exe
}

function Set-MeshOpsBlenderEnv([string]$ExePath) {
    # R1/C6: User scope via .NET + in-session $env: - never setx
    [Environment]::SetEnvironmentVariable("MESHOPS_BLENDER", $ExePath, "User")
    $env:MESHOPS_BLENDER = $ExePath
    Write-Info "Set MESHOPS_BLENDER (User + current session) = $ExePath"
    Write-Info "Other shells: open a new terminal to pick up the User env var."
}

function Show-OrcaHints {
    Write-Host ""
    Write-Info "OrcaSlicer (not auto-installed): pin 2.4.2"
    Write-Info "  GitHub: https://github.com/OrcaSlicer/OrcaSlicer/releases/tag/v2.4.2"
    Write-Info "  Microsoft Store: search OrcaSlicer"
    Write-Info "  Then: `$env:MESHOPS_ORCA = 'C:\Program Files\OrcaSlicer\orca-slicer.exe'"
}

# --- Main ---
$dryRun = [bool]($WhatIf -or $SkipDownload)
$toolsRoot = Get-ToolsRoot
$finalDir = Join-Path $toolsRoot $PortableDirName
$finalExe = Join-Path $finalDir $ExeName
$cacheDir = Join-Path $toolsRoot "cache"
$zipPath = Join-Path $cacheDir $ZipName
$mirrors = @(Get-MirrorCandidates)
$primaryMirror = $mirrors[0]
$checksumUrl = Get-ChecksumUrl $primaryMirror

Write-Info "MeshOps bootstrap-tools.ps1 (Blender $BlenderVersion portable)"
Write-Info "Difficulty section 4: non-CDN mirrors first; official CDN last."
Write-Info "Tools root: $toolsRoot"
Write-Info "Portable layout pin: $finalExe"
Write-Info "Zip: $ZipName"
Write-Info "SHA-256 pin: $PinnedSha256"
Write-Info "Primary mirror candidate: $primaryMirror"
Write-Info "Checksum URL (sibling): $checksumUrl"
Write-Info "Env action: User + session MESHOPS_BLENDER -> $finalExe"
if ($dryRun) {
    Write-Info "DRY-RUN (-WhatIf/-SkipDownload): no network, no extract, no env write."
    Write-Info "Mirror order:"
    $i = 0
    foreach ($m in $mirrors) {
        $i++
        $tag = if (Test-IsOfficialMirror $m) { " [official LAST]" } else { "" }
        Write-Info ("  {0}. {1}{2}" -f $i, $m, $tag)
    }
    Show-OrcaHints
    Write-Info "Next (after real run): uv run meshops doctor --require blender"
    exit 0
}

if ($Interactive) {
    Write-Host "Interactive: install portable Blender $BlenderVersion to $finalDir ?"
    $ans = Read-Host "Continue? [Y/n]"
    if ($ans -and $ans -notmatch '^[Yy]') {
        Write-Info "Aborted by user."
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null

$downloaded = $false
$winner = $null
$lastError = $null

foreach ($url in $mirrors) {
    $timeout = Get-ConnectTimeoutSec $url
    $attempt = 0
    $maxAttempts = 2  # try once, then retry once per mirror (spec section 3.4)
    $mirrorOk = $false
    while ($attempt -lt $maxAttempts -and -not $mirrorOk) {
        $attempt++
        Write-Info "Trying mirror attempt $attempt/$maxAttempts (timeout ${timeout}s): $url"
        try {
            if (Test-Path $zipPath) {
                Remove-Item -Force $zipPath
            }
            Invoke-DownloadFile -Url $url -OutFile $zipPath -TimeoutSec $timeout
            if (-not (Test-Path $zipPath)) {
                throw "Download produced no file"
            }
            $size = (Get-Item $zipPath).Length
            $minBytes = 1MB
            if ($size -lt $minBytes) {
                throw "Downloaded file too small ($size bytes) - likely an error page"
            }
            $downloaded = $true
            $winner = $url
            $sizeMb = [math]::Round($size / 1MB, 1)
            Write-Info "Mirror OK: $url ($sizeMb MB)"
            $mirrorOk = $true
        }
        catch {
            $lastError = $_
            Write-Warn "Mirror failed (attempt $attempt): $url - $($_.Exception.Message)"
            if (Test-Path $zipPath) {
                Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
            }
        }
    }
    if ($mirrorOk) {
        break
    }
}

if (-not $downloaded) {
    Write-Error "All mirrors failed. Last error: $lastError. Try -MirrorUrl with a local/cached URL."
    exit 1
}

Write-Info "Verifying SHA-256 (fail closed on mismatch)..."
$actual = Get-FileSha256 $zipPath
if ($actual -ne $PinnedSha256.ToLowerInvariant()) {
    Write-Error "SHA-256 mismatch: expected $PinnedSha256 got $actual - aborting (R7 fail closed)."
    exit 1
}
Write-Info "SHA-256 OK."

Write-Info "Extracting to $finalDir ..."
try {
    $exe = Expand-BlenderZip -ZipPath $zipPath -ToolsRoot $toolsRoot -FinalDir $finalDir
}
catch {
    Write-Error "Extract failed: $_"
    exit 1
}

# Fail closed: never claim success without blender --version 5.2.x (spec section 2.5)
$verOk = $false
$verOut = ""
try {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = "--version"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    if (-not $proc.WaitForExit(30000)) {
        try { $proc.Kill() } catch { }
        throw "blender --version timed out after 30s"
    }
    $verOut = ($proc.StandardOutput.ReadToEnd() + $proc.StandardError.ReadToEnd())
    if ($proc.ExitCode -ne 0) {
        throw "blender --version exited $($proc.ExitCode)"
    }
    if ($verOut -match "Blender\s+5\.2") {
        $verOk = $true
        Write-Info "blender --version OK (5.2.x)"
    }
    else {
        $head = $verOut.Substring(0, [Math]::Min(160, $verOut.Length))
        throw "version probe did not report Blender 5.2.x (got: $head)"
    }
}
catch {
    Write-Error "Blender version gate failed after extract: $_. Refusing to set MESHOPS_BLENDER (fail closed)."
    exit 1
}

if (-not $verOk) {
    Write-Error "Blender 5.2 version gate failed. Aborting without env write."
    exit 1
}

Set-MeshOpsBlenderEnv -ExePath $exe
Write-Info "Winner mirror: $winner"
Show-OrcaHints
Write-Info "Next: uv run meshops doctor --require blender"
Write-Info "Done."
exit 0
