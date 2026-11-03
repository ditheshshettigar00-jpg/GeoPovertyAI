"""Tests for the Hidden Gap Detector (residual_service + /api/residual-atlas + /hidden-gaps)."""
import pytest

from backend.services import residual_service as rs


def _all():
    return rs._load()


def test_summary_counts_add_up():
    s = rs.get_residual_atlas("all", limit=200)["summary"]
    assert s["hidden_candidates"] + s["resilient_candidates"] + s["consistent"] == s["n_districts"]


def test_candidates_need_interval_and_rank_gap_in_the_same_direction():
    df = _all()
    hid = df[df["gap_class"] == rs.HIDDEN]
    res = df[df["gap_class"] == rs.RESILIENT]
    assert len(hid) and len(res)
    assert (hid["y_true"] > hid["high_80"]).all() and (hid["rank_gap"] >= rs.RANK_THRESHOLD).all()
    assert (res["y_true"] < res["low_80"]).all() and (res["rank_gap"] <= -rs.RANK_THRESHOLD).all()
    # every candidate is outside the interval, but not every district outside the interval is a candidate
    assert df[df["gap_class"] != rs.CONSISTENT]["outside_80"].all()
    assert df["outside_80"].sum() > (df["gap_class"] != rs.CONSISTENT).sum()


def test_direction_filters_are_signed_correctly():
    hidden = rs.get_residual_atlas("hidden", limit=200)["districts"]
    resilient = rs.get_residual_atlas("resilient", limit=200)["districts"]
    assert hidden and all(r["gap"] > 0 and r["class"] == rs.HIDDEN for r in hidden)
    assert resilient and all(r["gap"] < 0 and r["class"] == rs.RESILIENT for r in resilient)


def test_results_sorted_by_abs_rank_gap_and_limited():
    rows = rs.get_residual_atlas("all", limit=10)["districts"]
    assert len(rows) <= 10
    gaps = [abs(r["rank_gap"]) for r in rows]
    assert gaps == sorted(gaps, reverse=True)


def test_invalid_direction_and_state_filter():
    with pytest.raises(ValueError):
        rs.get_residual_atlas("sideways")
    rows = rs.get_residual_atlas("all", state="bihar", limit=200)["districts"]
    assert all(r["state_name"] == "Bihar" for r in rows)


def test_small_states_are_marked_unreliable_and_listed_last():
    d = rs.get_residual_atlas("all")
    flags = [r["reliable"] for r in d["state_bias"]]
    assert all(r["reliable"] == (r["n"] >= d["min_state_districts"]) for r in d["state_bias"])
    assert flags == sorted(flags, reverse=True)          # reliable states come first
    assert not all(flags)                                  # the data does contain thin states


def test_match_caution_only_for_fuzzy_matches():
    rows = rs.get_residual_atlas("all", limit=200)["districts"]
    assert all(r["match_caution"] == (r["match_method"] == "fuzzy") for r in rows)


def test_note_states_the_limits():
    note = rs.get_residual_atlas("all")["note"].lower()
    assert "not proof" in note and "20%" in note


def test_api_endpoint(client):
    r = client.get("/api/residual-atlas?direction=hidden&limit=5")
    assert r.status_code == 200
    assert len(r.get_json()["districts"]) <= 5
    assert client.get("/api/residual-atlas?direction=bad").status_code == 400
    assert client.get("/hidden-gaps").status_code == 200
