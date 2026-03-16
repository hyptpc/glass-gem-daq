from __future__ import annotations

from pathlib import Path
import os


# 必須: オシロのIPアドレス
OSC_IP = os.environ.get("OSC_IP")
if not OSC_IP:
    raise RuntimeError("OSC_IP environment variable is not set.")

# 任意: チャンネルやレコード長などはコード側デフォルトをそのまま使用
DEFAULT_CHANNEL = os.environ.get("OSC_CHANNEL", "CH1")
START_INDEX = int(os.environ.get("OSC_START_INDEX", "1"))
STOP_INDEX = int(os.environ.get("OSC_STOP_INDEX", "10000"))
TIMEOUT_MS = int(os.environ.get("OSC_TIMEOUT_MS", "30000"))

# 出力ディレクトリ（これは秘密でないので固定でもOK）
OUTDIR = Path(os.environ.get("SCOPE_OUTDIR", "scope_output"))


