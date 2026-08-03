import asyncio

from channellm.pipeline.transport import ChunkChannel, OverrunPolicy


def test_put_get_roundtrip():
    async def run():
        channel = ChunkChannel[str]("thinker->talker", capacity=4)
        await channel.put("chunk-1")
        item = await channel.get()
        return item, channel.stats

    item, stats = asyncio.run(run())
    assert item == "chunk-1"
    assert stats.enqueued == 1
    assert stats.dequeued == 1


def test_drop_oldest_on_overrun():
    async def run():
        channel = ChunkChannel[int]("q", capacity=2, policy=OverrunPolicy.DROP_OLDEST)
        for i in range(4):
            channel.put_nowait(i)
        first = await channel.get()
        second = await channel.get()
        return channel, first, second

    channel, first, second = asyncio.run(run())
    assert channel.qsize() == 0
    assert channel.stats.dropped_oldest == 2
    assert (first, second) == (2, 3)  # oldest (0,1) dropped


def test_drop_newest_on_overrun():
    async def run():
        channel = ChunkChannel[int]("q", capacity=2, policy=OverrunPolicy.DROP_NEWEST)
        for i in range(4):
            channel.put_nowait(i)
        first = await channel.get()
        second = await channel.get()
        return channel, first, second

    channel, first, second = asyncio.run(run())
    assert channel.stats.dropped_newest == 2
    assert (first, second) == (0, 1)


def test_first_chunk_timeout_is_longer():
    async def run():
        channel = ChunkChannel[int]("q", capacity=2, first_timeout_s=0.05, next_timeout_s=0.01)
        item = await channel.get()  # first-chunk budget (0.05s)
        return item, channel.stats

    item, stats = asyncio.run(run())
    assert item is None
    assert stats.timeouts == 1


def test_subsequent_timeout_applies_after_first():
    async def run():
        channel = ChunkChannel[int]("q", capacity=2, first_timeout_s=1.0, next_timeout_s=0.01)
        channel.put_nowait(7)
        got = await channel.get()  # first seen
        item = await channel.get()  # now subsequent timeout applies
        return got, item, channel.stats

    got, item, stats = asyncio.run(run())
    assert got == 7
    assert item is None
    assert stats.timeouts == 1
