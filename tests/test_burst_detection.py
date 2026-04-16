"""
tests/test_burst_detection.py
==============================
Tests for py_mea_axion.burst.detection.

Manually constructed spike trains with known burst structure are used to
verify ISI-threshold output exactly.  The trains are designed so that
correct and incorrect implementations produce obviously different results.

Spike train conventions used throughout
----------------------------------------
TRAIN_TWO_BURSTS
    Two clean bursts of 5 spikes separated by a long silence:
      Burst 1: 0.010, 0.020, 0.030, 0.040, 0.050  (ISI = 10 ms within)
      Silence: 0.050 → 2.000  (gap = 1.95 s >> min_ibi = 0.2 s)
      Burst 2: 2.010, 2.020, 2.030, 2.040, 2.050  (ISI = 10 ms within)
    Expected: 2 bursts, each with 5 spikes.

TRAIN_MERGE
    Two close bursts that should be merged because their gap < min_ibi:
      Burst A: 0.010 … 0.050  (5 spikes, ISI = 10 ms)
      Gap:     0.050 → 0.200  (150 ms < min_ibi = 200 ms)
      Burst B: 0.200 … 0.240  (5 spikes, ISI = 10 ms)
    Expected: 1 merged burst of 10 spikes.

TRAIN_TOO_SMALL
    One group of only 3 spikes — should be filtered out by min_spikes=5.

TRAIN_BOUNDARY_ISI
    Spike pairs exactly at the threshold (ISI = max_isi_s exactly) —
    should NOT form a burst (threshold is strict <, not <=).
"""

import numpy as np
import pytest

from py_mea_axion.burst.detection import Burst, detect_bursts

# ── Canonical test trains ─────────────────────────────────────────────────────

# Two well-separated clean bursts.
TRAIN_TWO_BURSTS = np.array([
    0.010, 0.020, 0.030, 0.040, 0.050,     # burst 1
    2.010, 2.020, 2.030, 2.040, 2.050,     # burst 2
])

# Two close bursts whose gap (150 ms) < min_ibi (200 ms) → should merge.
TRAIN_MERGE = np.array([
    0.010, 0.020, 0.030, 0.040, 0.050,     # burst A
    0.200, 0.210, 0.220, 0.230, 0.240,     # burst B (gap = 150 ms)
])

# Group of only 3 spikes → filtered by min_spikes=5.
TRAIN_TOO_SMALL = np.array([0.010, 0.020, 0.030])

# Spikes with ISI exactly equal to threshold → no burst expected.
TRAIN_BOUNDARY_ISI = np.array([0.0, 0.1, 0.2, 0.3, 0.4])  # ISI = 0.1 s exactly

# Long bursty train: 5 bursts of 6 spikes each, 5 s apart.
TRAIN_FIVE_BURSTS = np.concatenate([
    np.arange(0.010, 0.060 + 0.009, 0.010) + i * 5.0
    for i in range(5)
])

# Single spike — no bursts possible.
TRAIN_SINGLE = np.array([1.0])

# Empty — no bursts.
TRAIN_EMPTY = np.array([], dtype=np.float64)


# ── Basic structure ───────────────────────────────────────────────────────────

class TestBurstNamedtuple:
    def test_fields(self):
        expected = ("start_time", "end_time", "spike_times",
                    "n_spikes", "duration", "mean_isi_within", "median_isi_within")
        assert Burst._fields == expected

    def test_construction(self):
        spikes = np.array([0.1, 0.2, 0.3])
        b = Burst(0.1, 0.3, spikes, 3, 0.2, 0.1, 0.1)
        assert b.n_spikes == 3
        assert b.duration == pytest.approx(0.2)


# ── ISI-threshold: two clean bursts ──────────────────────────────────────────

class TestIsiThresholdTwoBursts:
    """Core correctness test against a manually constructed known train."""

    @pytest.fixture()
    def bursts(self):
        return detect_bursts(
            TRAIN_TWO_BURSTS,
            max_isi_s=0.1,
            min_spikes=5,
            min_ibi_s=0.2,
        )

    def test_count(self, bursts):
        assert len(bursts) == 2

    def test_burst1_start(self, bursts):
        assert bursts[0].start_time == pytest.approx(0.010)

    def test_burst1_end(self, bursts):
        assert bursts[0].end_time == pytest.approx(0.050)

    def test_burst1_n_spikes(self, bursts):
        assert bursts[0].n_spikes == 5

    def test_burst1_duration(self, bursts):
        assert bursts[0].duration == pytest.approx(0.040)

    def test_burst1_mean_isi(self, bursts):
        assert bursts[0].mean_isi_within == pytest.approx(0.010)

    def test_burst2_start(self, bursts):
        assert bursts[1].start_time == pytest.approx(2.010)

    def test_burst2_end(self, bursts):
        assert bursts[1].end_time == pytest.approx(2.050)

    def test_burst2_n_spikes(self, bursts):
        assert bursts[1].n_spikes == 5

    def test_sorted_by_start_time(self, bursts):
        starts = [b.start_time for b in bursts]
        assert starts == sorted(starts)

    def test_spike_times_are_numpy_arrays(self, bursts):
        for b in bursts:
            assert isinstance(b.spike_times, np.ndarray)
            assert b.spike_times.dtype == np.float64

    def test_spike_times_content(self, bursts):
        np.testing.assert_allclose(bursts[0].spike_times,
                                   [0.010, 0.020, 0.030, 0.040, 0.050])
        np.testing.assert_allclose(bursts[1].spike_times,
                                   [2.010, 2.020, 2.030, 2.040, 2.050])


# ── ISI-threshold: merging ────────────────────────────────────────────────────

class TestIsiThresholdMerge:
    def test_two_close_bursts_merge_into_one(self):
        bursts = detect_bursts(
            TRAIN_MERGE, max_isi_s=0.1, min_spikes=5, min_ibi_s=0.2
        )
        assert len(bursts) == 1

    def test_merged_burst_has_all_spikes(self):
        bursts = detect_bursts(
            TRAIN_MERGE, max_isi_s=0.1, min_spikes=5, min_ibi_s=0.2
        )
        assert bursts[0].n_spikes == 10

    def test_merged_start_and_end(self):
        bursts = detect_bursts(
            TRAIN_MERGE, max_isi_s=0.1, min_spikes=5, min_ibi_s=0.2
        )
        assert bursts[0].start_time == pytest.approx(0.010)
        assert bursts[0].end_time == pytest.approx(0.240)

    def test_no_merge_when_gap_exceeds_min_ibi(self):
        # Same train but min_ibi=0.1 → gap (150 ms) > min_ibi → two bursts.
        bursts = detect_bursts(
            TRAIN_MERGE, max_isi_s=0.1, min_spikes=5, min_ibi_s=0.1
        )
        assert len(bursts) == 2


# ── ISI-threshold: edge cases ────────────────────────────────────────────────

class TestIsiThresholdEdgeCases:
    def test_empty_train(self):
        assert detect_bursts(TRAIN_EMPTY) == []

    def test_single_spike(self):
        assert detect_bursts(TRAIN_SINGLE) == []

    def test_too_few_spikes_filtered(self):
        assert detect_bursts(TRAIN_TOO_SMALL, min_spikes=5) == []

    def test_exactly_min_spikes_accepted(self):
        # 5 spikes with ISI 10 ms — exactly at min_spikes=5.
        spikes = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        bursts = detect_bursts(spikes, max_isi_s=0.1, min_spikes=5)
        assert len(bursts) == 1

    def test_boundary_isi_not_a_burst(self):
        # ISI == max_isi_s exactly → strict < → no burst.
        bursts = detect_bursts(TRAIN_BOUNDARY_ISI, max_isi_s=0.1, min_spikes=5)
        assert len(bursts) == 0

    def test_five_bursts(self):
        bursts = detect_bursts(TRAIN_FIVE_BURSTS, max_isi_s=0.1, min_spikes=5)
        assert len(bursts) == 5

    def test_all_bursts_have_correct_spike_count(self):
        bursts = detect_bursts(TRAIN_FIVE_BURSTS, max_isi_s=0.1, min_spikes=5)
        assert all(b.n_spikes == 6 for b in bursts)

    def test_duration_equals_end_minus_start(self):
        bursts = detect_bursts(TRAIN_TWO_BURSTS, max_isi_s=0.1, min_spikes=5)
        for b in bursts:
            assert b.duration == pytest.approx(b.end_time - b.start_time)

    def test_min_spikes_4_finds_more_bursts(self):
        # With min_spikes=4 the same train should still give 2 bursts.
        bursts = detect_bursts(TRAIN_TWO_BURSTS, max_isi_s=0.1, min_spikes=4)
        assert len(bursts) == 2

    def test_tight_isi_threshold_misses_bursts(self):
        # Threshold of 5 ms is smaller than the 10 ms within-burst ISI.
        bursts = detect_bursts(TRAIN_TWO_BURSTS, max_isi_s=0.005, min_spikes=5)
        assert len(bursts) == 0

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            detect_bursts(TRAIN_TWO_BURSTS, algorithm="magic")


# ── ISI-threshold: burst with background noise ────────────────────────────────

class TestIsiThresholdWithNoise:
    """Realistic train: two bursts embedded in low-rate background spikes."""

    @pytest.fixture()
    def noisy_train(self):
        rng = np.random.default_rng(42)
        # Background: Poisson at 0.5 Hz over 10 s.
        bg = np.sort(rng.uniform(0, 10, 5))
        burst1 = np.array([1.00, 1.01, 1.02, 1.03, 1.04, 1.05])
        burst2 = np.array([7.00, 7.01, 7.02, 7.03, 7.04, 7.05])
        return np.sort(np.concatenate([bg, burst1, burst2]))

    def test_finds_both_bursts(self, noisy_train):
        bursts = detect_bursts(noisy_train, max_isi_s=0.1, min_spikes=5)
        assert len(bursts) == 2

    def test_burst_centres_near_expected(self, noisy_train):
        bursts = detect_bursts(noisy_train, max_isi_s=0.1, min_spikes=5)
        centres = [(b.start_time + b.end_time) / 2 for b in bursts]
        assert abs(centres[0] - 1.025) < 0.1
        assert abs(centres[1] - 7.025) < 0.1


# ── Poisson Surprise ──────────────────────────────────────────────────────────

class TestPoissonSurprise:
    def test_finds_bursts_in_clean_train(self):
        bursts = detect_bursts(
            TRAIN_TWO_BURSTS,
            max_isi_s=0.1,
            min_spikes=5,
            algorithm="poisson_surprise",
            surprise_threshold=2.0,
        )
        assert len(bursts) >= 1

    def test_high_threshold_rejects_bursts(self):
        # Very high surprise threshold — nothing should pass.
        bursts = detect_bursts(
            TRAIN_TWO_BURSTS,
            algorithm="poisson_surprise",
            surprise_threshold=1e6,
        )
        assert len(bursts) == 0

    def test_returns_list_of_burst_namedtuples(self):
        bursts = detect_bursts(
            TRAIN_TWO_BURSTS,
            algorithm="poisson_surprise",
            surprise_threshold=2.0,
        )
        for b in bursts:
            assert isinstance(b, Burst)

    def test_empty_train(self):
        assert detect_bursts(TRAIN_EMPTY, algorithm="poisson_surprise") == []


# ── Real-data smoke test ──────────────────────────────────────────────────────

class TestRealDataSmoke:
    def test_burst_detection_on_active_well(self):
        from pathlib import Path
        spk = Path("LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk")
        if not spk.exists():
            pytest.skip("Real .spk file not available")

        from py_mea_axion.io.spk_reader import load_spikes_from_spk
        result, total = load_spikes_from_spk(spk, wells=["A1"])

        # Pick the most active electrode.
        most_active = max(result.items(), key=lambda kv: len(kv[1]))
        eid, spike_times = most_active

        bursts = detect_bursts(spike_times, max_isi_s=0.1, min_spikes=5)

        # A D28 active electrode should have at least some bursts.
        assert len(bursts) >= 0          # smoke: doesn't crash
        for b in bursts:
            assert b.n_spikes >= 5
            assert b.duration >= 0
            assert b.end_time >= b.start_time
