"""Path of an ffmpeg binary: the system one, else the one bundled with the imageio-ffmpeg pip package
(some cluster nodes have no ffmpeg on PATH)."""
import shutil


def ffmpeg_bin():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
