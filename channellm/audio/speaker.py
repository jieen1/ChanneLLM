"""声纹嵌入与验证 —— campplus ONNX(复用 Code2Wav 的说话人模型)。

噪声与轻语音在能量上同量级,能量原理无法分离;声纹按说话人身份过滤:
注册阶段采集目标说话人若干秒语音求平均嵌入,在线阶段对候选语音段计算
嵌入并比对余弦相似度。实测同人不同段相似度 ~0.6,人声 vs 噪声 ~0.05,
阈值 0.35 有充分区分度。

特征口径与官方 stepaudio2 Token2wav._prepare_prompt 提取参考音频 spk_emb
完全一致(kaldi.fbank 80 维,dither=0,逐维均值归一),注册/验证/参考音频
三者同空间。

``SpeakerGate`` 只在"模型回复播放中"门控候选语音(防环境音 barge-in 打断
回复);空闲收听路径零额外延迟、零改动。验证单位是"语音 episode"而非
单个 Silero run:实测自然对话被 Silero 切碎成大量 0.2~0.6s 短 run,逐 run
验证会丢弃目标说话人的多数短语;故跨短 run 间隙累积语音,累计 confirm_s
首次判定,通过 → 一次性冲刷全部扣留音频并直通至 episode 结束;不通过 →
继续累积、每 recheck_s 复核至 max_verify_s;连续 episode_gap_s 无语音才
结束 episode 重新设防。扣留期间样本不下发(不替换静音,保住词头)。
短段分离度实测(阈值 0.35):0.3s 同人 0.510/异人 0.004,1.0s 0.720/0.057。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLING_RATE = 16_000
_MIN_SEGMENT_S = 0.3  # 短于此的语音段嵌入不可靠


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    return vec / (float(np.linalg.norm(vec)) + 1e-9)


class SpeakerEmbedder:
    """campplus 说话人嵌入(L2 归一化)。"""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )

    def embedding(self, segment: np.ndarray) -> np.ndarray | None:
        """float32 16kHz 语音段 → L2 归一化嵌入;过短返回 None。"""
        if segment.size < int(_MIN_SEGMENT_S * SAMPLING_RATE):
            return None
        import torch
        import torchaudio

        wave = torch.from_numpy(np.ascontiguousarray(segment, dtype=np.float32)).unsqueeze(0)
        feat = torchaudio.compliance.kaldi.fbank(
            wave, num_mel_bins=80, sample_frequency=SAMPLING_RATE, dither=0.0
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        out = self.session.run(None, {"input": feat.unsqueeze(0).numpy()})[0]
        return _normalize(out[0])

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def embed_average(self, segments: list[np.ndarray]) -> np.ndarray | None:
        """多段语音 → 逐段嵌入后取均值再归一;无有效段返回 None。"""
        embs = [e for e in map(self.embedding, segments) if e is not None]
        if not embs:
            return None
        return _normalize(np.mean(embs, axis=0))


class VoiceprintStore:
    """声纹持久化:单个 L2 归一化嵌入,存 ``.npy``。"""

    def __init__(self, path: str | Path, embedder: SpeakerEmbedder) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.embedding: np.ndarray | None = None
        if self.path.is_file():
            self.embedding = _normalize(np.load(self.path))

    def save(self, embedding: np.ndarray) -> None:
        self.embedding = _normalize(embedding)
        np.save(self.path, self.embedding)

    def clear(self) -> None:
        self.embedding = None
        if self.path.is_file():
            self.path.unlink()


class SpeakerGate:
    """声纹门状态机(episode 语义):``feed(frame, reply_active=...)``。

    输入是 Silero 门控后的 float32 帧(非语音区间已为零),voiced 判定用
    "样本 != 0"(Silero 按 512 样本窗整窗门控,不受语音过零点影响)。
    输出:未门控/已通过 → 原音频;挂起确认或被拒 → 空数组(样本扣留,
    不下发也不替换,保住真实说话人的词头);确认通过时一次性冲刷全部扣留
    音频。调用方必须容忍空输出与比输入更长的输出。

    episode 生命周期:首个 voiced 样本开启;连续 episode_gap_s 无 voiced
    结束;期间跨 Silero run 间隙持续累积验证。已通过(open)的 episode
    剩余部分直通,不再重复验证。

    ``event`` 供上层读取后清零:"open"(声纹通过)/"muted"(声纹拒绝)/
    "dropped"(episode 在凑满确认窗前结束,丢弃)。
    """

    def __init__(
        self,
        embedder: SpeakerEmbedder,
        print_embedding: np.ndarray | None,
        *,
        threshold: float = 0.35,
        confirm_s: float = 0.3,
        recheck_s: float = 0.5,
        max_verify_s: float = 3.0,
        episode_gap_s: float = 0.9,
    ) -> None:
        self.embedder = embedder
        self.print_emb = print_embedding
        self.threshold = threshold
        self._confirm_n = int(confirm_s * SAMPLING_RATE)
        self._recheck_n = int(recheck_s * SAMPLING_RATE)
        self._max_verify_n = int(max_verify_s * SAMPLING_RATE)
        self._episode_gap_n = int(episode_gap_s * SAMPLING_RATE)
        self.event: str | None = None
        self.state = "idle"
        self._run: list[np.ndarray] = []
        self._run_n = 0
        self._gap_n = 0  # 距上一个 voiced 样本的静默样本数
        self._next_check_n = 0
        self._exhausted = False

    def reset(self) -> None:
        self.state = "idle"
        self._run = []
        self._run_n = 0
        self._gap_n = 0
        self._exhausted = False

    def feed(self, frame: np.ndarray, *, reply_active: bool) -> np.ndarray:
        if frame.size == 0 or self.print_emb is None:
            return frame
        if not reply_active:
            if self.state != "idle":
                self._end_episode()
            return frame
        voiced = frame != 0.0
        if not voiced.any():
            self._advance_gap(frame.size)
            return frame[:0]  # 静默帧:本就无声,不下发
        change = np.flatnonzero(np.diff(voiced.astype(np.int8))) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [frame.size]])
        parts: list[np.ndarray] = []
        for s, e in zip(starts, ends):
            if voiced[s]:
                self._gap_n = 0
                parts.append(self._feed_voiced(frame[s:e]))
            else:
                self._advance_gap(e - s)
        if not parts:
            return frame[:0]
        out = np.concatenate(parts)
        return out if out.size else frame[:0]

    def _advance_gap(self, n: int) -> None:
        self._gap_n += n
        if self._gap_n >= self._episode_gap_n and self.state != "idle":
            self._end_episode()

    def _feed_voiced(self, seg: np.ndarray) -> np.ndarray:
        if self.state == "open":
            return seg
        if self.state == "idle":  # 新 episode:挂起确认
            self.state = "pending"
            self._run, self._run_n = [], 0
            self._exhausted = False
        self._run.append(seg)
        self._run_n += seg.size
        if self.state == "pending" and self._run_n >= self._confirm_n:
            flush = self._decide()
            if flush is not None:
                return flush
        elif (
            self.state == "muted"
            and not self._exhausted
            and self._run_n >= self._next_check_n
        ):
            flush = self._decide()
            if flush is not None:
                return flush
        return seg[:0]  # 扣留:不下发

    def _decide(self) -> np.ndarray | None:
        """对整段累积语音做声纹判定;通过返回全部扣留音频,否则返回 None。"""
        full = np.concatenate(self._run)
        self._run = []
        emb = self.embedder.embedding(full)
        matched = (
            emb is not None
            and float(np.dot(emb, self.print_emb)) >= self.threshold  # type: ignore[arg-type]
        )
        if matched:
            self.state = "open"
            self.event = "open"
            return full
        self.state = "muted"
        self.event = "muted"
        self._next_check_n = self._run_n + self._recheck_n
        if self._run_n >= self._max_verify_n:
            self._exhausted = True  # 复核到顶仍不匹配,episode 内不再计算
        return None

    def _end_episode(self) -> None:
        if self.state == "pending" and self._run_n > 0:
            self.event = "dropped"  # 凑不满确认窗的 episode 无法验证,丢弃
        self.state = "idle"
        self._run = []
        self._run_n = 0
        self._exhausted = False
