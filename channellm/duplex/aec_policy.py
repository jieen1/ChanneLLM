"""客户端 AEC 健康状态的安全降级策略。

这不是 EOU 或说话权判定器；MiniCPM-o duplex 仍是唯一说话决策源。该模块仅在
客户端报告 AEC 失效时选择媒体交互模式，避免扬声器回声形成自我打断循环。
"""

from __future__ import annotations

import dataclasses
import enum


class AudioInteractionMode(str, enum.Enum):
    FULL_DUPLEX = "full_duplex"
    HEADSET_REQUIRED = "headset_required"
    PUSH_TO_TALK = "push_to_talk"


@dataclasses.dataclass(frozen=True)
class AecStatus:
    """由真实客户端/真机矩阵上报的能力事实，未知不得当作健康。"""

    healthy: bool | None
    headset_available: bool


def choose_audio_interaction(status: AecStatus) -> AudioInteractionMode:
    """AEC 健康才允许扬声器全双工；否则优先耳机，再退到 PTT。"""
    if status.healthy is True:
        return AudioInteractionMode.FULL_DUPLEX
    if status.headset_available:
        return AudioInteractionMode.HEADSET_REQUIRED
    return AudioInteractionMode.PUSH_TO_TALK
