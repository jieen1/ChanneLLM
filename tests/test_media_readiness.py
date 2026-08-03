from __future__ import annotations

from channellm.duplex.media_readiness import check_media_readiness


def test_media_readiness_keeps_remote_and_local_requirements_separate(tmp_path) -> None:
    snd = tmp_path / "snd"
    snd.mkdir()
    (snd / "timer").touch()
    (snd / "pcmC0D0c").touch()
    readiness = check_media_readiness(
        environment={
            "LIVEKIT_URL": "wss://example.invalid",
            "LIVEKIT_API_KEY": "key",
            "LIVEKIT_API_SECRET": "secret",
        },
        snd_path=snd,
        module_available=lambda name: name == "livekit",
    )

    assert readiness.remote_ready
    assert readiness.local_device_ready
    assert readiness.local_pcm_devices == ("pcmC0D0c",)
    assert readiness.missing_remote() == ()


def test_media_readiness_reports_missing_capabilities_without_reading_secrets(tmp_path) -> None:
    readiness = check_media_readiness(
        environment={}, snd_path=tmp_path / "absent", module_available=lambda _name: False
    )

    assert not readiness.remote_ready
    assert not readiness.local_device_ready
    assert readiness.missing_remote() == (
        "LiveKit Python SDK",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    )
