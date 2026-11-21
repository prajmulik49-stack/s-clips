import datetime as dt

import pytest

from joshiclip import facebook


class R:
    def __init__(self, j, code=200):
        self._j, self.status_code, self.text = j, code, str(j)

    def json(self):
        return self._j


def test_off_without_secrets(monkeypatch):
    monkeypatch.setattr(facebook, "_env", lambda k: "")
    assert not facebook.enabled({"facebook": {"enabled": True}})


def test_reel_is_uploaded_and_scheduled(tmp_path, monkeypatch):
    monkeypatch.setattr(facebook, "_env", lambda k: {"FB_PAGE_ID": "123", "FB_PAGE_TOKEN": "tok"}.get(k, ""))
    f = tmp_path / "a.mp4"; f.write_bytes(b"x" * 10)
    calls = []

    def post(url, data=None, headers=None, timeout=None):
        calls.append((url, dict(data) if isinstance(data, dict) else "<bytes>", headers))
        if isinstance(data, dict) and data.get("upload_phase") == "start":
            return R({"video_id": "v1", "upload_url": "https://rupload.facebook.com/video-upload/v23.0/v1"})
        return R({"success": True})
    import requests
    monkeypatch.setattr(requests, "post", post)
    when = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert facebook.post_reel({}, f, "T", when, log=lambda *a: None) == "v1"
    assert calls[1][2]["file_size"] == "10" and calls[1][2]["Authorization"] == "OAuth tok"
    fin = calls[2][1]
    assert fin["video_state"] == "SCHEDULED" and int(fin["scheduled_publish_time"]) > 0 and fin["video_id"] == "v1"


def test_facebook_error_is_raised_to_the_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(facebook, "_env", lambda k: "x")
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: R({"error": {"message": "bad token"}}, 400))
    (tmp_path / "a.mp4").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="bad token"):
        facebook.post_reel({}, tmp_path / "a.mp4", "T", None, log=lambda *a: None)


def test_caption_has_no_duplicate_hashtags():
    c = facebook.caption("Funny 😂 #shorts #souravjoshi", ["#shorts", "#souravjoshi", "#family"], "Credit: X")
    assert c.count("#shorts") == 1 and "#family" in c and "Credit: X" in c
