/* ChanneLLM web 客户端音频处理器:
 * - capture-processor: 麦克风(context 采样率) → 线性重采样 16kHz → Int16 帧上传,
 *   并周期性上报输入电平(供 UI 显示麦克风采样)
 * - playback-processor: 24kHz Int16 下行帧 → 环形缓冲(60ms 起播门限) →
 *   线性重采样到 context 采样率输出
 */

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.buf = [];
    this.pos = 0.0;
    this.levelCounter = 0;
    this.peak = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) {
      return true;
    }
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      const a = v < 0 ? -v : v;
      if (a > this.peak) this.peak = a;
      this.buf.push(v);
    }
    const out = [];
    while (this.pos + 1 < this.buf.length) {
      const i0 = Math.floor(this.pos);
      const frac = this.pos - i0;
      out.push(this.buf[i0] * (1 - frac) + this.buf[i0 + 1] * frac);
      this.pos += this.ratio;
    }
    const drop = Math.floor(this.pos);
    if (drop > 0) {
      this.buf.splice(0, drop);
      this.pos -= drop;
    }
    if (out.length > 0) {
      const i16 = new Int16Array(out.length);
      for (let i = 0; i < out.length; i++) {
        const s = Math.max(-1, Math.min(1, out[i]));
        i16[i] = s < 0 ? Math.round(s * 32768) : Math.round(s * 32767);
      }
      this.port.postMessage({ type: "pcm", buf: i16.buffer }, [i16.buffer]);
    }
    // ~每 100ms 上报一次电平
    this.levelCounter += ch.length;
    if (this.levelCounter >= sampleRate * 0.1) {
      this.port.postMessage({ type: "level", value: this.peak });
      this.levelCounter = 0;
      this.peak = 0;
    }
    return true;
  }
}

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = 24000 / sampleRate;
    this.cap = 24000 * 15;
    this.ring = new Float32Array(this.cap);
    this.w = 0;
    this.r = 0;
    this.count = 0;
    this.startThreshold = Math.floor(24000 * 0.06);
    this.playing = false;
    this.reportCounter = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d.type === "clear") {
        this.w = 0;
        this.r = 0;
        this.count = 0;
        this.playing = false;
      } else if (d.type === "push") {
        const pcm = new Int16Array(d.buf);
        for (let i = 0; i < pcm.length; i++) {
          if (this.count >= this.cap) {
            this.r = (this.r + 1) % this.cap;
            this.count--;
          }
          this.ring[this.w] = pcm[i] / 32768;
          this.w = (this.w + 1) % this.cap;
          this.count++;
        }
        if (!this.playing && this.count >= this.startThreshold) {
          this.playing = true;
        }
      }
    };
  }

  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!out) {
      return true;
    }
    let level = 0;
    if (!this.playing) {
      out.fill(0);
    } else {
      for (let i = 0; i < out.length; i++) {
        if (this.count <= 0) {
          out[i] = 0;
          continue;
        }
        const i0 = Math.floor(this.r);
        const frac = this.r - i0;
        const a = this.ring[i0 % this.cap];
        const b = this.ring[(i0 + 1) % this.cap];
        out[i] = a * (1 - frac) + b * frac;
        const av = out[i] < 0 ? -out[i] : out[i];
        if (av > level) level = av;
        this.r += this.ratio;
        if (this.r >= this.cap) {
          this.r -= this.cap;
        }
        this.count -= this.ratio;
      }
    }
    this.reportCounter += out.length;
    if (this.reportCounter >= sampleRate * 0.1) {
      this.port.postMessage({
        type: "state",
        playing: this.playing,
        bufferedMs: Math.round(this.count / 24),
        level,
      });
      this.reportCounter = 0;
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
registerProcessor("playback-processor", PlaybackProcessor);
