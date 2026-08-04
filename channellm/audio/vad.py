"""流式 VAD + 噪声门 —— Silero ONNX(参考官方 MiniCPM-o-Demo/vad)。

能量阈值无法区分人声与环境噪声(键盘/风扇/呼吸),导致不说话时误触发。
本模块用 Silero v5(16kHz,512 样本窗,内部 RNN 状态)做真语音检测,两个职责:

1. **噪声门**:非人声区间把输入替换为静音再送入 duplex 模型,模型听到的是
   干净静音而非噪声,从根上消除"不说话却一直被当成说话"的误触发。
2. **语音端点**:以 Silero 的"持续静默"判定替代 RMS 阈值触发提前冲刷,
   鲁棒得多。

实现为带 ``pad`` 前瞻的延迟线:输出滞后输入 pad 个样本,使语音起点被
检测到后才放行,不切词头。参考实现在 ~/project/MiniCPM-o-Demo/vad/vad.py
(SileroVADModel/StreamingVAD, Apache-2.0),这里精简为门控形态。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLING_RATE = 16_000
_WINDOW = 512  # Silero v5 @16kHz 固定窗长


class VoiceGate:
    """Silero 流式 VAD 噪声门。``feed(frame)`` 返回等长门控音频。"""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        pad_ms: int = 64,
        min_silence_ms: int = 450,
        min_speech_ms: int = 120,
    ) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.threshold = threshold
        self.neg_threshold = neg_threshold if neg_threshold is not None else threshold - 0.15
        self.pad = int(SAMPLING_RATE * pad_ms / 1000)
        self.min_silence = int(SAMPLING_RATE * min_silence_ms / 1000)
        self.min_speech = int(SAMPLING_RATE * min_speech_ms / 1000)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._raw = np.array([], dtype=np.float32)
        self._flags = np.array([], dtype=bool)
        self._proc_end = 0
        self._voiced = False
        self._silence_run = 0
        self._speech_run = 0
        self.speech_ended = False  # VoiceSession 读取并清零,触发提前冲刷
        self.speaking = False

    def _run_window(self, window: np.ndarray) -> float:
        out, self._h, self._c = self.session.run(
            None,
            {
                "input": window[None, :],
                "h": self._h,
                "c": self._c,
                "sr": np.array(SAMPLING_RATE, dtype=np.int64),
            },
        )
        return float(out.squeeze())

    def _update_voiced(self, prob: float) -> None:
        if not self._voiced:
            if prob >= self.threshold:
                self._speech_run += _WINDOW
                if self._speech_run >= self.min_speech:
                    self._voiced = True
                    self.speaking = True
                    self._silence_run = 0
            else:
                self._speech_run = 0
        else:
            if prob < self.neg_threshold:
                self._silence_run += _WINDOW
                if self._silence_run >= self.min_silence:
                    self._voiced = False
                    self.speaking = False
                    self._silence_run = 0
                    self._speech_run = 0
                    self.speech_ended = True
            else:
                self._silence_run = 0

    def feed(self, frame: np.ndarray) -> np.ndarray:
        """喂入 float32 帧,返回门控后的音频(静音段为零)。"""
        if frame.size == 0:
            return frame
        frame = frame.astype(np.float32, copy=False)
        self._raw = np.concatenate([self._raw, frame]) if self._raw.size else frame
        # 对完整窗做 VAD,生成逐窗 voiced 标志
        while self._proc_end + _WINDOW <= len(self._raw):
            window = self._raw[self._proc_end : self._proc_end + _WINDOW]
            self._update_voiced(self._run_window(window))
            flag_block = np.full(_WINDOW, self._voiced, dtype=bool)
            self._flags = (
                np.concatenate([self._flags, flag_block]) if self._flags.size else flag_block
            )
            self._proc_end += _WINDOW
        # 输出已决样本,保留 pad 前瞻
        out_len = min(len(self._flags) - self.pad, len(self._raw))
        if out_len <= 0:
            return np.zeros(0, dtype=np.float32)
        out = np.where(self._flags[:out_len], self._raw[:out_len], 0.0).astype(np.float32)
        self._raw = self._raw[out_len:]
        self._flags = self._flags[out_len:]
        self._proc_end -= out_len
        return out
