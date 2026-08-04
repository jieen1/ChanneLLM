"""声纹门/声纹注册单元测试(无 GPU):状态机语义、持久化、嵌入均值。

真实 campplus 的分离度证据测试(同人 > 阈值 > 异人)需要本地 checkpoint,
缺失时 skip——CI 无模型环境不阻塞,本机必须通过。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from channellm.audio.speaker import (
    SAMPLING_RATE,
    SpeakerEmbedder,
    SpeakerGate,
    VoiceprintStore,
)

_FRAME = SAMPLING_RATE // 10  # 0.1s


def _voiced(value: float = 0.2, n: int = _FRAME) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


def _silence(n: int = _FRAME) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


class FakeEmbedder:
    """verdict=True 返回与声纹同向的嵌入,否则返回反向。"""

    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.calls: list[int] = []

    def embedding(self, segment: np.ndarray) -> np.ndarray:
        self.calls.append(segment.size)
        vec = np.ones(4, dtype=np.float32) if self.verdict else -np.ones(4, dtype=np.float32)
        return vec / np.linalg.norm(vec)


def _gate(verdict: bool = True, **kw) -> tuple[SpeakerGate, FakeEmbedder]:
    emb = FakeEmbedder(verdict)
    print_emb = np.ones(4, dtype=np.float32) / 2.0
    return SpeakerGate(emb, print_emb, **kw), emb


def test_passthrough_without_voiceprint() -> None:
    emb = FakeEmbedder()
    gate = SpeakerGate(emb, None)
    frame = _voiced()
    out = gate.feed(frame, reply_active=True)
    assert out is frame
    assert emb.calls == []


def test_passthrough_when_reply_inactive() -> None:
    gate, emb = _gate()
    frame = _voiced()
    out = gate.feed(frame, reply_active=False)
    assert out is frame
    assert gate.state == "idle"
    for _ in range(10):
        gate.feed(_voiced(), reply_active=False)
    assert emb.calls == []  # 空闲收听路径零额外计算


def test_matching_speaker_holds_then_flushes_intact() -> None:
    gate, emb = _gate()  # confirm 默认 0.3s = 3 帧
    held = [_voiced(value=v) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    outs = [gate.feed(f, reply_active=True) for f in held]
    # 前 2 帧扣留(空输出),第 3 帧判定通过并冲刷全部扣留音频
    assert outs[0].size == 0 and outs[1].size == 0
    np.testing.assert_array_equal(outs[2], np.concatenate(held[:3]))
    assert gate.event == "open" and gate.state == "open"
    # 之后直通,话语后续样本零延迟
    np.testing.assert_array_equal(outs[3], held[3])
    np.testing.assert_array_equal(outs[4], held[4])
    assert emb.calls == [3 * _FRAME]  # 只算了一次


def test_mismatched_speaker_held_then_dropped_with_rechecks() -> None:
    gate, emb = _gate(verdict=False)
    total_frames = 40  # 4s 连续语音
    for _ in range(total_frames):
        out = gate.feed(_voiced(), reply_active=True)
        assert out.size == 0  # 全程扣留,模型听不到噪声说话人
    assert gate.event == "muted" and gate.state == "muted"
    # 0.3s 首判 + 每 0.5s 复核,3.0s 到顶后不再计算:3,8,13,18,23,28,33 帧 → 7 次
    assert len(emb.calls) == 7
    gate.feed(_silence(n=9 * _FRAME), reply_active=True)  # 长静默结束 episode
    assert gate.state == "idle"


def test_short_episode_dropped_without_embedding_call() -> None:
    gate, emb = _gate()
    gate.feed(_voiced(n=_FRAME), reply_active=True)  # 0.1s < confirm
    gate.feed(_voiced(n=_FRAME), reply_active=True)  # 0.2s < confirm
    assert emb.calls == []
    gate.feed(_silence(n=9 * _FRAME), reply_active=True)  # episode 结束才丢弃
    assert gate.event == "dropped" and gate.state == "idle"


def test_short_gap_keeps_episode_open_long_gap_rearms() -> None:
    gate, emb = _gate()
    for _ in range(3):
        gate.feed(_voiced(), reply_active=True)
    assert gate.state == "open"
    # 短间隙(< episode_gap 0.9s):episode 延续,保持直通
    gate.feed(_silence(n=3 * _FRAME), reply_active=True)
    assert gate.state == "open"
    frame = _voiced()
    np.testing.assert_array_equal(gate.feed(frame, reply_active=True), frame)
    assert len(emb.calls) == 1  # 整个 episode 只验证一次
    # 长间隙(>= episode_gap):episode 结束,新语音重新设防
    gate.feed(_silence(n=9 * _FRAME), reply_active=True)
    assert gate.state == "idle"
    out = gate.feed(_voiced(), reply_active=True)
    assert out.size == 0 and gate.state == "pending"


def test_fragmented_natural_speech_accumulates_across_runs() -> None:
    """真实语音被 Silero 切成短 run(实测 0.2~0.6s),必须跨 run 累积验证。"""
    gate, emb = _gate()
    voiced_lens = (0.2, 0.6, 0.48, 0.59, 0.32)  # 秒,提问 fixture 实测分布
    parts: list[np.ndarray] = []
    forwarded: list[np.ndarray] = []
    for i, secs in enumerate(voiced_lens):
        seg = _voiced(value=0.1 + i * 0.01, n=int(secs * SAMPLING_RATE))
        parts.append(seg)
        out = gate.feed(seg, reply_active=True)
        if out.size:
            forwarded.append(out)
        if i < len(voiced_lens) - 1:
            gap = gate.feed(_silence(n=int(0.2 * SAMPLING_RATE)), reply_active=True)
            assert gap.size == 0
    total_voiced = sum(p.size for p in parts)
    total_out = sum(f.size for f in forwarded)
    assert gate.state == "open"
    assert len(emb.calls) == 1  # 只验证一次,不逐 run 重复
    # 冲刷+直通:全部 voiced 样本最终下发,一个不少
    assert total_out == total_voiced
    np.testing.assert_array_equal(np.concatenate(forwarded), np.concatenate(parts))


def test_reply_deactivating_resets_gate_midrun() -> None:
    gate, emb = _gate(verdict=False)
    gate.feed(_voiced(), reply_active=True)
    assert gate.state == "pending"
    frame = _voiced()
    out = gate.feed(frame, reply_active=False)  # 回复结束:立即放行并复位
    assert out is frame and gate.state == "idle"
    assert emb.calls == []


def test_mixed_voiced_silence_frame_splits_runs() -> None:
    gate, emb = _gate(verdict=False)
    mixed = np.concatenate([_voiced(n=800), _silence(n=400), _voiced(n=800)])
    out = gate.feed(mixed, reply_active=True)
    assert out.size == 0
    # 两个不连续语音段:第二段重新 pending,首判只发生在 ≥confirm 的段
    assert gate.state == "pending" and emb.calls == []


def test_empty_frame_is_noop() -> None:
    gate, emb = _gate()
    out = gate.feed(np.zeros(0, dtype=np.float32), reply_active=True)
    assert out.size == 0 and gate.state == "idle"


def test_store_save_load_clear_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "voiceprint.npy"
    emb = FakeEmbedder()
    store = VoiceprintStore(path, emb)
    assert store.embedding is None
    vec = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)
    store.save(vec)
    np.testing.assert_allclose(store.embedding, vec / 5.0, atol=1e-6)
    reloaded = VoiceprintStore(path, emb)
    np.testing.assert_allclose(reloaded.embedding, store.embedding, atol=1e-6)
    reloaded.clear()
    assert reloaded.embedding is None and not path.is_file()


def _stub_embedder(results: list[np.ndarray | None]) -> SpeakerEmbedder:
    stub = SpeakerEmbedder.__new__(SpeakerEmbedder)
    queue = list(results)

    def embedding(segment: np.ndarray) -> np.ndarray | None:
        return queue.pop(0)

    stub.embedding = embedding  # type: ignore[method-assign]
    return stub


def test_embed_average_normalizes_mean_and_skips_none() -> None:
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    stub = _stub_embedder([None, v1, v2])
    out = stub.embed_average([_voiced()] * 3)
    np.testing.assert_allclose(out, [np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


def test_embed_average_all_none_returns_none() -> None:
    stub = _stub_embedder([None, None])
    assert stub.embed_average([_voiced()] * 2) is None


# ---------------------------------------------------------------- 真实模型证据

_SNAPSHOTS = sorted(
    (Path.home() / ".cache/huggingface/hub").glob(
        "models--openbmb--MiniCPM-o-4_5/snapshots/*/assets/token2wav/campplus.onnx"
    )
)
_ASSETS = sorted(
    (Path.home() / ".cache/huggingface/hub").glob(
        "models--openbmb--MiniCPM-o-4_5/snapshots/*/assets"
    )
)
_need_model = pytest.mark.skipif(
    not _SNAPSHOTS or not _ASSETS, reason="本机无 MiniCPM-o checkpoint"
)


def _load_wav_16k(path: Path, max_s: float = 3.0) -> np.ndarray:
    from channellm.models.minicpmo_compat import patch_torchaudio_load

    patch_torchaudio_load()  # 本机无 ffmpeg,torchaudio.load 走 soundfile 兜底
    import s3tokenizer

    wave = s3tokenizer.load_audio(str(path), sr=16000)
    return wave.numpy()[: int(max_s * 16000)].astype(np.float32)


@_need_model
def test_real_embedder_short_segment_returns_none() -> None:
    emb = SpeakerEmbedder(_SNAPSHOTS[0])
    assert emb.embedding(np.zeros(1600, dtype=np.float32)) is None


@_need_model
def test_real_embedder_separation_same_vs_other_speaker() -> None:
    """阈值 0.35 的分离度证据:同人分段显著高于阈值,异人显著低于阈值。"""
    assets = _ASSETS[0]
    emb = SpeakerEmbedder(_SNAPSHOTS[0])
    ref = _load_wav_16k(assets / "HT_ref_audio.wav", max_s=6.0)
    other = _load_wav_16k(assets / "nezha.wav", max_s=3.0)
    split = len(ref) // 2
    e_ref_a = emb.embedding(ref[:split])
    e_ref_b = emb.embedding(ref[split:])
    e_other = emb.embedding(other)
    assert e_ref_a is not None and e_ref_b is not None and e_other is not None
    assert float(np.linalg.norm(e_ref_a)) == pytest.approx(1.0, abs=1e-3)
    same = SpeakerEmbedder.similarity(e_ref_a, e_ref_b)
    cross = SpeakerEmbedder.similarity(e_ref_a, e_other)
    assert same > 0.35, f"同人分段相似度 {same:.3f} 必须高于阈值"
    assert cross < 0.35, f"异人相似度 {cross:.3f} 必须低于阈值"
    assert same > cross + 0.15, "同人/异人需有充分间隔"
