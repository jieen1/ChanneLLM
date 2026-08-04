#!/usr/bin/env python
"""ChanneLLM 流式语音服务入口 —— 装载模型并服务 WebSocket 语音会话。

用法:
    python scripts/app_server.py [--host 0.0.0.0] [--port 8765]

启动流程:装载 AudioFront/Thinker/Talker/Code2Wav(含各级 graph 捕获与
预热)→ 同端口提供 web 客户端静态文件与 /ws WebSocket 端点。单会话 MVP:
并发第二个连接收到 1013。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WEB_DIR = REPO_ROOT / "web"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--web-dir", type=Path, default=WEB_DIR, help="web 客户端静态文件目录"
    )
    parser.add_argument("--ssl-cert", type=Path, default=REPO_ROOT / "certs/server.crt")
    parser.add_argument("--ssl-key", type=Path, default=REPO_ROOT / "certs/server.key")
    parser.add_argument(
        "--no-tls", action="store_true",
        help="禁用 TLS(仅本机调试;麦克风/AudioWorklet 需要安全上下文)",
    )
    return parser


def load_models(device: str = "cuda"):
    """一次装载全部引擎并完成服务就绪预热(含各级 graph)。"""
    import torch

    from channellm.engine.audio_front import AudioFront
    from channellm.engine.code2wav import Code2Wav
    from channellm.engine.talker import load_talker_weights
    from channellm.engine.thinker import (
        SparkinferPagedKV,
        ThinkerConfig,
        load_thinker_weights,
    )
    from channellm.kernel.paged_kv import PagedKVPool
    from channellm.kernel.sparkinfer_attn import PagedAttnConfig, SparkinferPagedAttn
    from channellm.models.minicpmo_compat import (
        patch_torchaudio_load,
        patch_torchaudio_save,
    )
    from scripts.p1_voice_loop import REF_WAV_SUFFIX, find_snapshot

    patch_torchaudio_load()
    patch_torchaudio_save()

    dtype = torch.bfloat16
    model_dir = find_snapshot()
    print(f"[setup] snapshot: {model_dir}", flush=True)

    t0 = time.monotonic()
    audio_front = AudioFront(model_dir, device=device, dtype=dtype)
    audio_front.prewarm()
    torch.cuda.synchronize()
    print(f"[load] AudioFront(含预热) {time.monotonic() - t0:.1f}s", flush=True)

    t0 = time.monotonic()
    thinker = load_thinker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Thinker {time.monotonic() - t0:.1f}s", flush=True)

    t0 = time.monotonic()
    talker = load_talker_weights(model_dir, device=device, dtype=dtype)
    print(f"[load] Talker {time.monotonic() - t0:.1f}s", flush=True)

    t0 = time.monotonic()
    code2wav = Code2Wav(model_dir, model_dir / REF_WAV_SUFFIX)
    code2wav.prewarm_stream()
    code2wav.enable_stream_graphs()
    torch.cuda.synchronize()
    print(f"[load] Code2Wav prewarm + stream graphs {time.monotonic() - t0:.1f}s", flush=True)

    tconfig = ThinkerConfig.from_official(model_dir / "config.json")
    pool = PagedKVPool(
        num_layers=tconfig.num_hidden_layers,
        num_pages=512,
        page_size=64,
        num_kv_heads=tconfig.num_kv_heads,
        head_dim=tconfig.head_dim,
        dtype=dtype,
        device=device,
    )
    attn = SparkinferPagedAttn(
        PagedAttnConfig(
            num_q_heads=tconfig.num_q_heads,
            num_kv_heads=tconfig.num_kv_heads,
            head_dim=tconfig.head_dim,
            page_size=64,
            dtype=dtype,
        ),
        device,
    )

    class SharedModels:
        pass

    models = SharedModels()
    models.audio_front = audio_front
    models.thinker = thinker
    models.talker = talker
    models.code2wav = code2wav
    models.pool = pool
    models.attn = attn
    models.make_thinker_kv = lambda: SparkinferPagedKV(pool, attn)
    models.model_dir = model_dir
    return models


def start_ca_server(port: int, certs_dir: Path) -> None:
    """明纹 HTTP 端口只发布 CA 证书,供手机下载安装(证书本身可明文传输)。"""
    import functools
    import http.server
    import threading

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(certs_dir)
    )
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"[serve] CA 证书下载: http://<host>:{port}/ca.crt", flush=True)


_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


async def main() -> int:
    args = build_arg_parser().parse_args()

    import ssl as ssl_mod

    from websockets.asyncio.server import serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    from channellm.app.stream_server import VoiceSession

    models = load_models()

    # 声纹:复用 Code2Wav 的 campplus 说话人模型;加载失败降级为无声纹门。
    voiceprint = None
    try:
        from channellm.app.stream_server import VOICEPRINT_PATH
        from channellm.audio.speaker import SpeakerEmbedder, VoiceprintStore

        campplus = models.model_dir / "assets" / "token2wav" / "campplus.onnx"
        voiceprint = VoiceprintStore(VOICEPRINT_PATH, SpeakerEmbedder(campplus))
        print(f"[load] 声纹就绪 enrolled={voiceprint.embedding is not None}", flush=True)
    except Exception as exc:  # noqa: BLE001 - 声纹是增强项,不可用不能拖垮服务
        print(f"[load] 声纹不可用(降级无声纹门): {exc!r}", flush=True)

    ssl_ctx = None
    if not args.no_tls:
        if not (args.ssl_cert.is_file() and args.ssl_key.is_file()):
            print(
                f"[serve] 缺少证书 {args.ssl_cert} / {args.ssl_key};用 --no-tls 可退回 "
                "HTTP(手机浏览器将无法使用麦克风)", flush=True,
            )
        else:
            ssl_ctx = ssl_mod.SSLContext(ssl_mod.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(args.ssl_cert), str(args.ssl_key))
            start_ca_server(args.port + 1, REPO_ROOT / "certs")

    state = {"busy": False}

    async def sender(ws, session: VoiceSession) -> None:
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        session.sink.set_waker(lambda: loop.call_soon_threadsafe(wake.set))
        while True:
            items = session.sink.drain()
            for item in items:
                if item.kind == "audio":
                    await ws.send(item.epoch.to_bytes(2, "little") + item.payload)
                elif item.kind == "clear":
                    await ws.send(json.dumps({"type": "clear"}))
                elif item.kind == "turn":
                    await ws.send(json.dumps({"type": "turn", "epoch": item.epoch}))
                elif item.kind == "reply":
                    await ws.send(
                        json.dumps({"type": "reply", "epoch": item.epoch, **item.meta})
                    )
                elif item.kind == "control":
                    await ws.send(json.dumps(item.meta))
            if not items:
                wake.clear()
                await wake.wait()

    async def handler(ws) -> None:
        if state["busy"]:
            await ws.close(1013, "single-session server is busy")
            return
        state["busy"] = True
        session: VoiceSession | None = None
        try:
            session = VoiceSession(models, voiceprint=voiceprint)
            print("[session] opened", flush=True)
            send_task = asyncio.create_task(sender(ws, session))
            if voiceprint is not None:
                session.sink.post_control(
                    {"type": "voiceprint", "enrolled": voiceprint.embedding is not None}
                )
            try:
                async for message in ws:
                    if isinstance(message, bytes | bytearray):
                        session.feed_pcm16(bytes(message))
                    else:
                        try:
                            control = json.loads(message)
                        except json.JSONDecodeError:
                            continue
                        ctype = control.get("type")
                        if ctype == "eou":
                            session.mark_eou()
                        elif ctype == "enroll_start":
                            session.enroll_start()
                        elif ctype == "enroll_end":
                            session.enroll_end()
            finally:
                send_task.cancel()
        except Exception as exc:  # 会话级错误不能拖垮服务进程
            print(f"[session] error: {exc!r}", flush=True)
        finally:
            if session is not None:
                session.close()
            state["busy"] = False
            print("[session] closed", flush=True)

    async def process_request(connection, request):
        # WebSocket 升级请求放行;其余按静态文件服务(web 客户端)。
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        path = request.path.split("?", 1)[0]
        if path in ("/", ""):
            path = "/index.html"
        file_path = (args.web_dir / path.lstrip("/")).resolve()
        if (
            file_path.is_file()
            and file_path.is_relative_to(args.web_dir.resolve())
        ):
            body = file_path.read_bytes()
            content_type = _STATIC_TYPES.get(file_path.suffix, "application/octet-stream")
            return Response(
                200, "OK",
                Headers([
                    ("Content-Type", content_type),
                    ("Content-Length", str(len(body))),
                    ("Cache-Control", "no-store"),
                ]),
                body,
            )
        return Response(404, "Not Found", Headers([("Content-Length", "0")]), b"")

    scheme = "https" if ssl_ctx else "http"
    print(
        f"[serve] {scheme}://{args.host}:{args.port} (web 客户端) | "
        f"{'wss' if ssl_ctx else 'ws'}://{args.host}:{args.port}/ws",
        flush=True,
    )
    async with serve(
        handler, args.host, args.port, process_request=process_request, ssl=ssl_ctx
    ):
        await asyncio.get_running_loop().create_future()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
