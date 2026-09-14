"""Tests for hallucination detection (pure-logic, no model needed)."""

import pytest

from crisperwhisper.loop_detection import find_token_loop


class TestFindTokenLoop:
    def test_no_loop(self):
        ids = [1, 2, 3, 4, 5, 6, 7, 8]
        assert find_token_loop(ids, reps=3) is None

    def test_unigram_loop(self):
        ids = [10, 20, 5, 5, 5, 5, 5, 5, 5, 5]
        result = find_token_loop(ids, reps=8)
        assert result is not None
        start, gram = result
        assert gram == (5,)
        assert start == 2

    def test_bigram_loop(self):
        ids = [1, 2] * 10
        result = find_token_loop(ids, reps=8)
        assert result is not None
        _, gram = result
        assert len(gram) <= 2

    def test_trigram_loop(self):
        ids = [99] + [3, 4, 5] * 9
        result = find_token_loop(ids, reps=8, max_ngram=5)
        assert result is not None
        _, gram = result
        assert gram == (3, 4, 5)

    def test_threshold_exactly_met(self):
        ids = [7] * 8
        assert find_token_loop(ids, reps=8) is not None

    def test_threshold_not_met(self):
        ids = [7] * 7
        assert find_token_loop(ids, reps=8) is None

    def test_empty_input(self):
        assert find_token_loop([], reps=3) is None

    def test_min_ngram_filter(self):
        # Unigram loop exists but we only search bigrams+
        ids = [5] * 20
        result = find_token_loop(ids, min_ngram=2, reps=8)
        assert result is not None  # 5,5 repeated as bigram

    def test_max_ngram_filter(self):
        ids = [1, 2, 3, 4, 5, 6] * 10
        result = find_token_loop(ids, max_ngram=3, reps=8)
        # 6-gram wouldn't be found with max_ngram=3
        # but sub-patterns might not exist
        # The 6-gram repeats but max_ngram=3 won't catch it
        assert result is None

    def test_dict_reps_ignores_ngram_range(self):
        # Dict keys define the scanned sizes: an 11-gram sentence loop
        # must be caught even when max_ngram is left at its default (5).
        sentence = list(range(100, 111))  # 11 distinct tokens
        ids = [1, 2, 3] + sentence * 3
        result = find_token_loop(ids, reps={11: 3})
        assert result == (3, tuple(sentence))

    def test_dict_reps_scans_only_listed_sizes(self):
        ids = [7] * 20  # unigram loop
        assert find_token_loop(ids, reps={2: 4}) is not None  # as bigram
        assert find_token_loop(ids, reps={11: 3}) is None

    def test_default_thresholds_catch_sentence_loop(self):
        from crisperwhisper.loop_detection import DEFAULT_REPAIR_THRESHOLDS
        sentence = list(range(100, 111))
        ids = sentence * 3
        result = find_token_loop(
            ids, min_ngram=1, max_ngram=5, reps=DEFAULT_REPAIR_THRESHOLDS,
        )
        assert result == (0, tuple(sentence))


class TestFinalTrim:
    def test_persisting_loop_is_truncated(self):
        from crisperwhisper.loop_detection import (
            DEFAULT_REPAIR_THRESHOLDS,
            _final_trim,
        )
        sentence = list(range(100, 121))  # 21-token unit
        ids = [1, 2] + sentence * 10
        out = _final_trim(ids, DEFAULT_REPAIR_THRESHOLDS, keep_reps=1)
        assert out == [1, 2] + sentence

    def test_clean_output_untouched(self):
        from crisperwhisper.loop_detection import (
            DEFAULT_REPAIR_THRESHOLDS,
            _final_trim,
        )
        ids = list(range(50))
        assert _final_trim(ids, DEFAULT_REPAIR_THRESHOLDS, keep_reps=1) is ids


class TestBlockThresholds:
    def test_one_below_detection(self):
        from crisperwhisper.hallucination import _compute_bans
        from crisperwhisper.loop_detection import _block_thresholds
        block = _block_thresholds({1: 8, 11: 3})
        assert block == {1: 7, 11: 2}
        # With 2 consecutive copies of an 11-gram present, the token that
        # would start copy 3 (= the detection threshold) must be banned.
        sentence = list(range(100, 111))
        assert sentence[0] in _compute_bans(sentence * 2, block)
        assert sentence[0] not in _compute_bans(sentence, block)

    def test_int_thresholds(self):
        from crisperwhisper.loop_detection import _block_thresholds
        assert _block_thresholds(8, max_ngram=3) == {1: 7, 2: 7, 3: 7}


class TestUniqueNgramRatio:
    def test_near_copy_tail_is_degenerate(self):
        from crisperwhisper.loop_detection import (
            DEGENERATE_TAIL_RATIO,
            _unique_ngram_ratio,
        )
        # 12-token sentence repeated with a one-token mutation each copy —
        # the failure mode exact-match blocking cannot prevent.
        tail = []
        for k in range(10):
            unit = list(range(100, 112))
            unit[3] = 200 + k
            tail += unit
        assert _unique_ngram_ratio(tail) < DEGENERATE_TAIL_RATIO

    def test_normal_speech_is_kept(self):
        from crisperwhisper.loop_detection import (
            DEGENERATE_TAIL_RATIO,
            _unique_ngram_ratio,
        )
        assert _unique_ngram_ratio(list(range(200))) == 1.0
        # Even a genuinely repeated sentence inside otherwise fresh
        # content stays above the bar.
        ids = list(range(100)) + list(range(50, 62)) * 2 + list(range(300, 350))
        assert _unique_ngram_ratio(ids) > DEGENERATE_TAIL_RATIO

    def test_short_tail_never_degenerate(self):
        from crisperwhisper.loop_detection import _unique_ngram_ratio
        assert _unique_ngram_ratio([1, 2, 3]) == 1.0


class TestCt2FreeImport:
    """Issue #57: the loop detector must be importable (and the re-export
    modules loadable) on a transformers-only install without ctranslate2."""

    def test_loop_detection_imports_without_ctranslate2(self, monkeypatch):
        import importlib
        import sys

        # A None entry makes ``import ctranslate2`` raise ImportError.
        monkeypatch.setitem(sys.modules, "ctranslate2", None)
        monkeypatch.delitem(
            sys.modules, "crisperwhisper.loop_detection", raising=False,
        )
        mod = importlib.import_module("crisperwhisper.loop_detection")
        assert mod.find_token_loop([1, 1, 1, 1], reps=3) == (0, (1,))
        assert 1 in mod.DEFAULT_REPAIR_THRESHOLDS
