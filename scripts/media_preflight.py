#!/usr/bin/env python
"""检查 P5 LiveKit/本地媒体接入的部署前置条件，不输出任何凭据。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from channellm.duplex.media_readiness import check_media_readiness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-remote", action="store_true")
    parser.add_argument("--require-local-device", action="store_true")
    args = parser.parse_args()
    readiness = check_media_readiness()
    print(f"LiveKit SDK       [{'PASS' if readiness.livekit_sdk else 'MISSING'}]")
    print(f"LiveKit config    [{'PASS' if readiness.livekit_configured else 'MISSING'}]")
    print(
        "local PCM devices "
        f"[{'PASS' if readiness.local_device_ready else 'MISSING'}] "
        f"{', '.join(readiness.local_pcm_devices) if readiness.local_pcm_devices else ''}"
    )
    if readiness.missing_remote():
        print("remote missing: " + ", ".join(readiness.missing_remote()))
    required_ok = (
        (not args.require_remote or readiness.remote_ready)
        and (not args.require_local_device or readiness.local_device_ready)
    )
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
