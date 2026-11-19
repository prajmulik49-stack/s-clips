"""Frame-exact continuous-span cutting from norm.mp4 in ONE ffmpeg filter graph.

- span snapped to the frame grid: frames [fa, fb) at the normalized fps
- audio samples [fa*sr/fps, fb*sr/fps): an integer for 30/60 fps at 48 kHz, so A and V have identical length
- input fast-seek a few seconds early + -copyts, then trim/atrim on ABSOLUTE timestamps -> exact frames
- never a concat demuxer, never separate A/V files
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    fa: int          # first frame (inclusive)
    fb: int          # last frame (exclusive)
    fps: int
    sr: int = 48000

    @property
    def start(self) -> float:
        return self.fa / self.fps

    @property
    def end(self) -> float:
        return self.fb / self.fps

    @property
    def dur(self) -> float:
        return (self.fb - self.fa) / self.fps

    @property
    def sa(self) -> int:
        return self.fa * self.sr // self.fps

    @property
    def sb(self) -> int:
        return self.fb * self.sr // self.fps

    @property
    def n_samples(self) -> int:
        return self.sb - self.sa


def snap(start: float, end: float, fps: int, sr: int = 48000) -> Span:
    if (sr % fps) != 0:
        raise ValueError(f"{sr} Hz is not an integer number of samples per frame at {fps} fps")
    return Span(int(round(start * fps)), int(round(end * fps)), fps, sr)


def seek_args(span: Span, preroll: float = 3.0) -> list[str]:
    return ["-ss", f"{max(0.0, span.start - preroll):.6f}", "-copyts"]


def trim_graph(span: Span, vin: str = "0:v", ain: str = "0:a", vout: str = "v0", aout: str = "a0") -> str:
    """trim/atrim on absolute timestamps (half-frame guards so exactly frames fa..fb-1 are kept).
    NOTE: atrim start_sample/end_sample count samples from the first sample *received* (i.e. from the seek point),
    not from the timestamp, so they are wrong after an input seek. Use pts-based start/end on the sample grid."""
    t0 = (span.fa - 0.5) / span.fps
    t1 = (span.fb - 0.5) / span.fps
    return (f"[{vin}]trim=start={t0:.6f}:end={t1:.6f},setpts=PTS-STARTPTS[{vout}];"
            f"[{ain}]atrim=start={span.sa / span.sr:.9f}:end={span.sb / span.sr:.9f},asetpts=PTS-STARTPTS[{aout}]")


def encode_args(preset: dict, fps: int, threads: int = 0) -> list[str]:
    a = ["-c:v", "libx264", "-preset", preset["preset"], "-crf", str(preset["crf"]), "-pix_fmt", "yuv420p",
         "-fps_mode", "cfr", "-r", str(fps), "-c:a", "aac", "-b:a", preset["audio_bitrate"], "-ar", "48000",
         "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
         "-flags:a", "+bitexact"]
    if threads:
        a += ["-threads", str(threads)]
    return a


def cut_plain(norm, span: Span, out, preset: dict, height: int | None = None, threads: int = 0, extra_vf: str = "") -> list[str]:
    """Build the ffmpeg command for a plain 16:9 continuous cut (step 3 / --export-candidates)."""
    from . import media
    g = trim_graph(span, vout="vt")
    chain = []
    if height:
        chain.append(f"scale=-2:{height}:flags=lanczos")
    if extra_vf:
        chain.append(extra_vf)
    chain.append("format=yuv420p")
    g += f";[vt]{','.join(chain)}[v0]"
    return [media.ff(), "-v", "error", "-y", *seek_args(span), "-i", str(norm), "-filter_complex", g,
            "-map", "[v0]", "-map", "[a0]", *encode_args(preset, span.fps, threads), str(out)]


def multi_trim_graph(spans: list[Span], vout: str = "vt", aout: str | None = None, fade_ms: float = 0.0,
                     want_video: bool = True) -> str:
    """Several frame-exact segments joined in ONE filter graph (split -> trim -> concat). Audio segments are cut on the
    same frame grid (integer samples per frame), so A/V stay locked at every join; optional tiny fades avoid clicks."""
    n = len(spans)
    parts = []
    if want_video:
        parts.append(f"[0:v]split={n}" + "".join(f"[vs{i}]" for i in range(n)) if n > 1 else "[0:v]null[vs0]")
        for i, s in enumerate(spans):
            parts.append(f"[vs{i}]trim=start={(s.fa - 0.5) / s.fps:.6f}:end={(s.fb - 0.5) / s.fps:.6f},setpts=PTS-STARTPTS[vc{i}]")
        parts.append("".join(f"[vc{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[{vout}]")
    if aout:
        parts.append(f"[0:a]asplit={n}" + "".join(f"[as{i}]" for i in range(n)) if n > 1 else "[0:a]anull[as0]")
        for i, s in enumerate(spans):
            f = ""
            if fade_ms and n > 1:
                d = fade_ms / 1000
                f = f",afade=t=in:d={d:.4f},afade=t=out:st={max(0.0, s.dur - d):.6f}:d={d:.4f}"
            parts.append(f"[as{i}]atrim=start={s.sa / s.sr:.9f}:end={s.sb / s.sr:.9f},asetpts=PTS-STARTPTS{f}[ac{i}]")
        parts.append("".join(f"[ac{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[{aout}]")
    return ";".join(parts)
