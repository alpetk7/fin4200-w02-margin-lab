import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MARGIN_GAME_NO_ANIM"] = "1"

import pytest
from streamlit.testing.v1 import AppTest

APP = str(ROOT / "app.py")


def click(at, label_part):
    for b in at.button:
        if label_part in b.label:
            b.click()
            return at.run(timeout=60)
    raise AssertionError(f"no button containing {label_part!r}: {[b.label for b in at.button]}")


def play(at, margin):
    for r in range(10):
        assert at.session_state.phase == "decide", at.session_state.phase
        at.select_slider[0].set_value(margin)
        at = at.run(timeout=60)
        at = click(at, "REVEAL WHAT HAPPENS")
        assert not at.exception
        assert at.session_state.phase == "result"
        labels = [b.label for b in at.button]
        at = click(at, "SEE FINAL RESULTS" if any("SEE FINAL" in x for x in labels) else "Continue")
        if at.session_state.phase == "done":
            break
    return at


@pytest.mark.parametrize("margin", [1.0, 0.5])
def test_full_game_and_replays(margin):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at = click(at, "START THE GAME")
    seed = at.session_state.seed
    at = play(at, margin)
    assert at.session_state.phase == "done" and not at.exception
    prices1 = at.session_state.market.price_series()
    final1 = at.session_state.series[-1]

    at = click(at, "SAME market")
    assert at.session_state.seed == seed and at.session_state.phase == "decide"
    assert at.session_state.market.price_series() == prices1
    at = play(at, 0.5 if margin == 1.0 else 1.0)
    assert at.session_state.phase == "done" and not at.exception
    assert at.session_state.market.price_series() == prices1
    assert at.session_state.series[-1] != final1

    at = click(at, "NEW market")
    assert at.session_state.phase == "decide" and not at.exception
