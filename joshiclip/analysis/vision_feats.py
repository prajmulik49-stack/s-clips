"""Vision analysis on proxy_360p.mp4 at a low fixed rate: faces (MediaPipe), motion (Farneback flow),
colour histograms (location changes), scene cuts (PySceneDetect)."""
from __future__ import annotations

import numpy as np

from .. import media

MAX_FACES = 4


def _frames(proxy, fps: float, w: int, h: int):
    import subprocess
    from .. import tools
    p = subprocess.Popen([media.ff(), "-v", "error", "-i", str(proxy), "-an", "-vf",
                          f"fps={fps}:round=near,scale={w}:{h}:flags=area,format=rgb24", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, env=tools.child_env())
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    p.wait()


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def analyze(proxy, model_path: str, fps: float, min_conf: float, detector: str = "yunet"):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    info = media.probe(proxy)["video"]
    W, H = info["width"], info["height"]
    opts = None if detector != "mediapipe" else vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model_path),
                                      running_mode=vision.RunningMode.IMAGE, min_detection_confidence=min_conf)
    faces, counts, motion, hists, bright = [], [], [], [], []
    prev = None
    if detector == "yunet":
        cv2.setNumThreads(1)
        yn = cv2.FaceDetectorYN.create(model_path, "", (W, H), min_conf, 0.3, 50)
    with vision.FaceDetector.create_from_options(opts) if detector == "mediapipe" else _Null() as det:
        for fr in _frames(proxy, fps, W, H):
            if detector == "yunet":
                _, f = yn.detect(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                dets = [] if f is None else [[x[0] / W, x[1] / H, x[2] / W, x[3] / H, x[-1]] for x in f]
            else:
                r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)))
                dets = [[d.bounding_box.origin_x / W, d.bounding_box.origin_y / H, d.bounding_box.width / W,
                         d.bounding_box.height / H, d.categories[0].score] for d in r.detections]
            boxes = sorted((list(map(float, b)) for b in dets), key=lambda b: (-round(b[2] * b[3], 6), b[0]))[:MAX_FACES]
            row = np.zeros((MAX_FACES, 5), np.float32)
            if boxes:
                row[:len(boxes)] = boxes
            faces.append(row); counts.append(len(boxes))
            small = cv2.cvtColor(cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            if prev is None:
                motion.append(0.0)
            else:
                fl = cv2.calcOpticalFlowFarneback(prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion.append(float(np.sqrt((fl ** 2).sum(-1)).mean()))
            prev = small
            hsv = cv2.cvtColor(cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2HSV)
            hst = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
            hists.append(hst / (hst.sum() + 1e-9)); bright.append(float(hsv[..., 2].mean()))
    return {"faces": np.array(faces, np.float32), "face_count": np.array(counts, np.int16),
            "motion": np.array(motion, np.float32), "hist": np.array(hists, np.float32),
            "brightness": np.array(bright, np.float32), "fps": np.float32(fps)}


def scene_cuts(proxy, threshold: float) -> list[float]:
    from scenedetect import ContentDetector, SceneManager, open_video
    v = open_video(str(proxy))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=8))
    sm.detect_scenes(v, show_progress=False)
    return [round(s[0].get_seconds(), 3) for s in sm.get_scene_list()[1:]]
