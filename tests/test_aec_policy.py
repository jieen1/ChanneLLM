from channellm.duplex.aec_policy import AecStatus, AudioInteractionMode, choose_audio_interaction


def test_healthy_aec_allows_full_duplex() -> None:
    assert choose_audio_interaction(AecStatus(healthy=True, headset_available=False)) is (
        AudioInteractionMode.FULL_DUPLEX
    )


def test_failed_aec_prefers_headset_before_ptt() -> None:
    assert choose_audio_interaction(AecStatus(healthy=False, headset_available=True)) is (
        AudioInteractionMode.HEADSET_REQUIRED
    )
    assert choose_audio_interaction(AecStatus(healthy=False, headset_available=False)) is (
        AudioInteractionMode.PUSH_TO_TALK
    )


def test_unknown_aec_never_assumes_speaker_full_duplex_is_safe() -> None:
    assert choose_audio_interaction(AecStatus(healthy=None, headset_available=True)) is (
        AudioInteractionMode.HEADSET_REQUIRED
    )
