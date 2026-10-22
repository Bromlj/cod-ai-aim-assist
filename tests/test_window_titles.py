"""Regression tests for games that insert invisible Unicode in window titles."""
from screensight.windows import normalized_window_title, window_title_matches


def test_call_of_duty_black_ops_seven_title_with_zero_width_spaces():
    # Captured pattern: invisible U+200B is inserted between every glyph.
    decorated = "\u200b".join("Call of Duty®: Black Ops 7") + "\u200b" * 20
    assert "Call of Duty" not in decorated
    assert normalized_window_title(decorated) == "call of duty®: black ops 7"
    assert window_title_matches(decorated, ("Call of Duty", "Black Ops 7"))
    assert window_title_matches(decorated, ("Call of Duty®: Black Ops 7",))


def test_unicode_word_joiner_and_nonbreaking_space_title():
    decorated = "Call\u2060 of\u2060 Duty®:\u00a0Black\u2060 Ops 7"
    assert window_title_matches(decorated, ("Black Ops 7",))


def test_window_filters_remain_restrictive():
    decorated = "\u200b".join("Call of Duty®: Black Ops 7")
    assert not window_title_matches(decorated, ("Notepad", "Discord"))
    assert not window_title_matches(decorated, ())
    assert not window_title_matches(decorated, ("", "   "))
