from channellm.duplex.epoch import EpochGuard, EpochTag


def test_stale_before_advance():
    guard = EpochGuard()
    assert guard.is_stale(EpochTag(turn_epoch=1))


def test_advance_invalidates_old():
    guard = EpochGuard()
    first = guard.advance("speech-1")
    assert guard.accept(first)
    second = guard.advance("speech-2")
    assert guard.is_stale(first)  # 旧 epoch 无条件丢弃
    assert guard.accept(second)
    assert second.turn_epoch == first.turn_epoch + 1
