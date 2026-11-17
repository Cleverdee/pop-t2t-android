@echo off
rem Prepares an installed Prince of Persia: Warrior Within (pop2.exe) or The Two Thrones (pop3.exe) folder for the
rem Android app: GOG installs the game exe packed with UPX, and the app needs it unpacked. Put this file in the game
rem folder and double-click it. It downloads the official UPX 4.2.4 from github.com/upx/upx (checked by SHA-256),
rem keeps the packed exe as <name>.exe.packed, and unpacks the exe in place. Nothing else in the folder changes.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $ErrorActionPreference = 'Stop'; try { . ([scriptblock]::Create(((Get-Content -Raw -LiteralPath '%~f0') -split '#PS#')[-1])) } catch { Write-Host ('ERROR: ' + $_) -ForegroundColor Red; exit 1 } }"
echo.
pause
exit /b
#PS#
$exe = @('pop2.exe', 'pop3.exe') | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $exe) { throw 'No pop2.exe or pop3.exe here. Put PrepareGame.bat in the installed game folder and run it there.' }
function Test-Packed($path) {
    $b = [System.IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $path))
    $pe = [BitConverter]::ToInt32($b, 0x3c)
    $nsec = [BitConverter]::ToUInt16($b, $pe + 6); $opt = [BitConverter]::ToUInt16($b, $pe + 20)
    for ($i = 0; $i -lt $nsec; $i++) {
        $t = $pe + 24 + $opt + 40 * $i
        if ([Text.Encoding]::ASCII.GetString($b, $t, 3) -eq 'UPX') { return $true }
    }
    return $false
}
if (-not (Test-Packed $exe)) { Write-Host "$exe is already unpacked: the folder is ready for the app." -ForegroundColor Green; return }
Write-Host "$exe is packed. Downloading UPX 4.2.4 from github.com/upx/upx ..."
$tmp = Join-Path $env:TEMP ('popprep-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $zip = Join-Path $tmp 'upx.zip'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/upx/upx/releases/download/v4.2.4/upx-4.2.4-win64.zip' -OutFile $zip
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLower()
    if ($hash -ne '22e9ef20e4c72aad85e32c71cbc9c086436c179456382aa75c0c24868456a671') { throw "the UPX download doesn't match the official release (SHA-256 $hash)" }
    Expand-Archive -LiteralPath $zip -DestinationPath $tmp
    $upx = Join-Path $tmp 'upx-4.2.4-win64\upx.exe'
    if (-not (Test-Path -LiteralPath "$exe.packed")) { Copy-Item -LiteralPath $exe -Destination "$exe.packed" }
    & $upx -d $exe | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "UPX couldn't unpack $exe (exit code $LASTEXITCODE)" }
    if (Test-Packed $exe) { throw "$exe still looks packed after UPX" }
    Write-Host "Done: $exe is unpacked (the packed one is kept as $exe.packed). Copy this folder to your phone, or zip it, and import it in the app." -ForegroundColor Green
} finally {
    Remove-Item -Recurse -Force -LiteralPath $tmp -ErrorAction SilentlyContinue
}
