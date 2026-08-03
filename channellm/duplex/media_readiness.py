"""P5 媒体部署前置条件的无副作用检查。"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaReadiness:
    livekit_sdk: bool
    livekit_url: bool
    livekit_api_key: bool
    livekit_api_secret: bool
    local_pcm_devices: tuple[str, ...]

    @property
    def remote_ready(self) -> bool:
        return self.livekit_sdk and self.livekit_configured

    @property
    def livekit_configured(self) -> bool:
        return self.livekit_url and self.livekit_api_key and self.livekit_api_secret

    @property
    def local_device_ready(self) -> bool:
        return bool(self.local_pcm_devices)

    def missing_remote(self) -> tuple[str, ...]:
        checks = {
            "LiveKit Python SDK": self.livekit_sdk,
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
        }
        return tuple(label for label, present in checks.items() if not present)


def check_media_readiness(
    *,
    environment: dict[str, str] | None = None,
    snd_path: Path = Path("/dev/snd"),
    module_available: Callable[[str], bool] | None = None,
) -> MediaReadiness:
    """检查能力存在性，不连接服务器、不读取或打印任何 secret。"""
    env = os.environ if environment is None else environment
    available = module_available or _module_available
    devices = tuple(sorted(path.name for path in _pcm_devices(snd_path)))
    return MediaReadiness(
        livekit_sdk=available("livekit"),
        livekit_url=bool(env.get("LIVEKIT_URL")),
        livekit_api_key=bool(env.get("LIVEKIT_API_KEY")),
        livekit_api_secret=bool(env.get("LIVEKIT_API_SECRET")),
        local_pcm_devices=devices,
    )


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _pcm_devices(snd_path: Path) -> Iterable[Path]:
    if not snd_path.is_dir():
        return ()
    return (path for path in snd_path.iterdir() if path.name.startswith("pcm"))
