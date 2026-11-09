import numpy as np, wave, sys
SR = 44100; DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 80.0; OUT = sys.argv[2] if len(sys.argv) > 2 else "music2.wav"
BPM = 100; beat = 60 / BPM; N = int(SR * DUR)
L = np.zeros(N); Rr = np.zeros(N)
rng = np.random.default_rng(1)
f = lambda m: 440 * 2 ** ((m - 69) / 12)
def add(buf, start, sig):
    s = int(start * SR)
    if s >= N: return
    e = min(N, s + len(sig)); buf[s:e] += sig[:e - s]
def piano(m, d, v):
    t = np.arange(int(SR * d)) / SR; fr = f(m)
    s = sum(a * np.sin(2 * np.pi * fr * k * t) for k, a in [(1, 1), (2, .45), (3, .2), (4, .1)])
    env = np.exp(-t * 3.2) * np.minimum(1, t / 0.005)
    return v * s * env
def pad(ms, d, v):
    t = np.arange(int(SR * d)) / SR
    s = sum(np.sin(2 * np.pi * f(m) * t + 0.3 * np.sin(2 * np.pi * 0.2 * t)) + 0.5 * np.sin(2 * np.pi * f(m) * 1.003 * t) for m in ms)
    env = np.minimum(1, t / 0.8) * np.minimum(1, (d - t) / 0.8)
    return v * s * env / len(ms)
def kick(v):
    t = np.arange(int(SR * .35)) / SR
    return v * np.sin(2 * np.pi * (50 + 90 * np.exp(-t * 30)) * t) * np.exp(-t * 9)
def hat(v):
    t = np.arange(int(SR * .06)) / SR; n = rng.standard_normal(len(t))
    n = np.diff(np.concatenate([[0], n]))
    return v * n * np.exp(-t * 60)
def bass(m, d, v):
    t = np.arange(int(SR * d)) / SR
    return v * (np.sin(2 * np.pi * f(m) * t) + .3 * np.sin(4 * np.pi * f(m) * t)) * np.exp(-t * 1.2) * np.minimum(1, t / .01)
# C - G - Am - F  (one chord per bar of 4 beats)
prog = [(60, [60, 64, 67]), (55, [55, 59, 62]), (57, [57, 60, 64]), (53, [53, 57, 60])]
bar = 4 * beat; nb = int(DUR / bar) + 1
for b in range(nb):
    t0 = b * bar; root, ch = prog[b % 4]
    p = pad([m + 12 for m in ch], bar + .8, .10); add(L, t0, p); add(Rr, t0, p)
    bs = bass(root - 24, bar, .16); add(L, t0, bs); add(Rr, t0, bs)
    arp = [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[1] + 12, ch[0] + 24, ch[2] + 12, ch[1] + 12, ch[2] + 12]
    for i, m in enumerate(arp):
        pn = piano(m, 1.2, .09 if i % 2 == 0 else .07)
        add(L, t0 + i * beat / 2, pn * (1.1 if i % 2 else .9)); add(Rr, t0 + i * beat / 2, pn * (.9 if i % 2 else 1.1))
    if b >= 2:  # light beat after intro
        for i in range(4):
            k = kick(.22 if i % 2 == 0 else .12); add(L, t0 + i * beat, k); add(Rr, t0 + i * beat, k)
        for i in range(8):
            h = hat(.025); add(L, t0 + i * beat / 2 + beat / 4, h); add(Rr, t0 + i * beat / 2 + beat / 4, h * .8)
# simple echo
d = int(SR * beat * .75)
for buf in (L, Rr):
    e = np.zeros_like(buf); e[d:] = buf[:-d] * .25; buf += e
st = np.stack([L, Rr], 1)
fade = np.ones(N); fi = int(SR * 2); fo = int(SR * 3)
fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo)
st *= fade[:, None]; st = st / np.abs(st).max() * 0.8
with wave.open(OUT, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes())
print("ok", DUR)
