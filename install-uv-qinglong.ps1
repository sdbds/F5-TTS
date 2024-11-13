Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://pypi.mirrors.ustc.edu.cn/simple"
$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
$Env:UV_CACHE_DIR="${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_CACHE=0
$Env:UV_LINK_MODE="symlink"
function InstallFail {
    Write-Output "安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

try {
    ~/.cargo/bin/uv --version
    ~/.cargo/bin/uv self update
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Install uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        Check "install uv failed"
    }
    else {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        Check "install uv failed。"
    }
}

if ($env:OS -ilike "*windows*") {
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        . ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        . ./.venv/Scripts/activate
    }else{
        Write-Output "Create .venv"
        ~/.cargo/bin/uv.exe venv -p 3.10
        . ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    . ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    . ./.venv/bin/activate.ps1
}
else{
    Write-Output "Create .venv"
    uv venv -p 3.10
    . ./.venv/bin/activate.ps1
}

Write-Output "install requirements"

~/.cargo/bin/uv pip sync requirements-uv.txt --index-strategy unsafe-best-match
Check "install requirements failed"

Write-Output "install finished"
Read-Host | Out-Null ;
