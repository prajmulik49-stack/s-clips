from joshiclip import record, tools


def test_mode(monkeypatch):
    monkeypatch.setattr(tools, "_dotenv", lambda: {})
    monkeypatch.delenv("JOSHICLIP_FETCH", raising=False)
    assert record.mode() == "ytdlp"
    monkeypatch.setenv("JOSHICLIP_FETCH", "Record")
    assert record.mode() == "record"
    monkeypatch.setenv("JOSHICLIP_FETCH", "nope")
    try:
        record.mode()
        assert False
    except RuntimeError:
        pass


def test_cookies_from_netscape(tmp_path, monkeypatch):
    p = tmp_path / "c.txt"
    p.write_text("# Netscape HTTP Cookie File\n"
                 ".youtube.com\tTRUE\t/\tTRUE\t1999999999\tSID\tabc\n"
                 "#HttpOnly_.google.com\tTRUE\t/\tTRUE\t0\tHSID\tdef\n"
                 ".example.com\tTRUE\t/\tFALSE\t0\tX\ty\n", encoding="utf-8")
    monkeypatch.setattr(tools, "cookies_path", lambda: str(p))
    ck = record._cookies()
    assert [c["name"] for c in ck] == ["SID", "HSID"]
    assert ck[0]["expires"] == 1999999999 and ck[0]["secure"] and not ck[0]["httpOnly"]
    assert ck[1]["httpOnly"] and "expires" not in ck[1]


def test_quality_ladder_order():
    assert record.QUALITIES.index("hd2160") < record.QUALITIES.index("hd1080") < record.QUALITIES.index("medium")
    assert all(q in record.HEIGHT for q in record.QUALITIES)
