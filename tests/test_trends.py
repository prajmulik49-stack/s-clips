"""Data-backed category / title behaviour (research 2026-11-20 on 800 Sourav Joshi Shorts)."""
from joshiclip import boundaries, moment_first as m
from joshiclip.text import lex


def setup_module():
    lex.set_lexicon({"names": ["Avantika", "Piyush", "Nana"], "stopwords": [], "dependent_first": [],
                     "intro_outro": [], "sponsor": []})


def test_everyday_hindi_gaya_is_not_danger():
    assert m._category("hum ghar gaye aur khel gaye", "energy") != "danger_loss_fear"


def test_unlabelled_moment_is_everyday_not_funny():
    assert m._category("bas aise hi baithe hain", "energy") == "everyday"


def test_saas_bahu_moment_is_family():
    assert m._category("saas ne bahu ko khana banane bola", "energy") == "family_home"


def test_number_and_what_regexes_match_real_text():
    assert m._number("isme 35000 phone bante hain") == "35000"
    assert m._what("mujhe bahut accha laga", "everyday") == "mujhe"


def test_family_prior_beats_funny_prior(tmp_path):
    class Cfg(dict):
        def path(self, _):
            return tmp_path
    cfg = Cfg(research={"report_file": "none.json"})
    assert m._consensus_prior(cfg, "family_home") > m._consensus_prior(cfg, "funny_accident")


def test_named_line_wins_over_unnamed_question(monkeypatch):
    monkeypatch.setattr(boundaries, "question_title", lambda A, c: "Ye Kya Hai? 😄😄")
    monkeypatch.setattr(boundaries, "quote_title", lambda A, c: "Avantika Ne Khana Banaya 🥺😥")
    assert boundaries.make_title({}, {"category": "family_home"})[1] == "quote"
    monkeypatch.setattr(boundaries, "quote_title", lambda A, c: "Khana Ban Gaya 🥺😥")
    assert boundaries.make_title({}, {"category": "family_home"})[1] == "question"
