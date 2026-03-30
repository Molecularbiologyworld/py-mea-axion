"""
tests/test_network_detection.py
================================
Tests for py_mea_axion.network.detection.

Spike trains are constructed so that the network burst structure is
known exactly, allowing the tests to verify the algorithm precisely.

Test network layouts
--------------------
SYNC_TWO
    Two electrodes with perfectly synchronised bursts at t≈0.03 s and
    t≈5.03 s.  Both electrodes active → 100 % participation.
    Expected: 2 network bursts.

SYNC_FOUR_PARTIAL
    Four electrodes; only two burst together at each event.
    participation_threshold=0.25 → threshold_count = ceil(0.25×4) = 1
    → even one electrode bursting qualifies.
    With threshold=0.5 → threshold_count=2 → only the events where
    both burst together should count.

STAGGERED
    Two electrodes whose burst windows do NOT overlap → no network burst
    at participation_threshold=1.0, but one at threshold=0.5.
"""

import numpy as np
import pytest

from py_mea_axion.burst.detection import Burst, detect_bursts
from py_mea_axion.network.detection import NetworkBurst, detect_network_bursts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _burst(start, end, n=5):
    """Construct a single Burst namedtuple."""
    spikes = np.linspace(start, end, n)
    isis = np.diff(spikes)
    return Burst(
        start_time=float(start),
        end_time=float(end),
        spike_times=spikes,
        n_spikes=n,
        duration=float(end - start),
        mean_isi_within=float(np.mean(isis)),
    )


# Two synchronised bursts at ~0 s and ~5 s, two electrodes.
SYNC_TWO = {
    "A1_11": [_burst(0.010, 0.050), _burst(5.010, 5.050)],
    "A1_12": [_burst(0.010, 0.050), _burst(5.010, 5.050)],
}

# Four electrodes; two pairs burst together at different times.
SYNC_FOUR_PARTIAL = {
    "A1_11": [_burst(0.010, 0.050)],
    "A1_12": [_burst(0.010, 0.050)],
    "A1_13": [_burst(3.010, 3.050)],
    "A1_14": [_burst(3.010, 3.050)],
}

# Two electrodes with non-overlapping burst windows.
STAGGERED = {
    "A1_11": [_burst(0.010, 0.050)],
    "A1_12": [_burst(0.200, 0.240)],
}

TOTAL_TIME = 10.0


# ── NetworkBurst namedtuple ───────────────────────────────────────────────────

class TestNetworkBurstNamedtuple:
    def test_fields(self):
        expected = (
            "start_time", "end_time", "duration",
            "participating_electrodes", "participation_fraction",
            "peak_participation",
        )
        assert NetworkBurst._fields == expected

    def test_construction(self):
        nb = NetworkBurst(0.0, 0.05, 0.05, ["A1_11"], 0.5, 0.5)
        assert nb.duration == pytest.approx(0.05)
        assert nb.participating_electrodes == ["A1_11"]


# ── Synchronised two-electrode case ──────────────────────────────────────────

class TestSyncTwoElectrodes:
    @pytest.fixture()
    def nb(self):
        return detect_network_bursts(
            SYNC_TWO, TOTAL_TIME,
            participation_threshold=0.5,
            bin_size_s=0.01,
            min_network_ibi_s=1.0,
        )

    def test_count(self, nb):
        assert len(nb) == 2

    def test_sorted_by_start_time(self, nb):
        starts = [b.start_time for b in nb]
        assert starts == sorted(starts)

    def test_first_burst_start_near_zero(self, nb):
        assert nb[0].start_time < 0.1

    def test_second_burst_start_near_five(self, nb):
        assert 4.9 < nb[1].start_time < 5.1

    def test_participation_fraction_is_one(self, nb):
        for b in nb:
            assert b.participation_fraction == pytest.approx(1.0)

    def test_peak_participation_is_one(self, nb):
        for b in nb:
            assert b.peak_participation == pytest.approx(1.0)

    def test_both_electrodes_participating(self, nb):
        for b in nb:
            assert set(b.participating_electrodes) == {"A1_11", "A1_12"}

    def test_duration_positive(self, nb):
        for b in nb:
            assert b.duration > 0

    def test_end_time_ge_start_time(self, nb):
        for b in nb:
            assert b.end_time >= b.start_time

    def test_returns_list_of_network_bursts(self, nb):
        for b in nb:
            assert isinstance(b, NetworkBurst)


# ── Partial participation ─────────────────────────────────────────────────────

class TestPartialParticipation:
    def test_low_threshold_finds_both_events(self):
        # threshold=0.25 → ceil(0.25×4)=1 → any bursting electrode qualifies
        nb = detect_network_bursts(
            SYNC_FOUR_PARTIAL, TOTAL_TIME,
            participation_threshold=0.25,
            bin_size_s=0.01,
        )
        assert len(nb) == 2

    def test_high_threshold_finds_both_events(self):
        # threshold=0.5 → ceil(0.5×4)=2 → both pairs meet threshold
        nb = detect_network_bursts(
            SYNC_FOUR_PARTIAL, TOTAL_TIME,
            participation_threshold=0.5,
            bin_size_s=0.01,
        )
        assert len(nb) == 2

    def test_threshold_above_one_pair_misses_partial(self):
        # threshold=0.75 → ceil(0.75×4)=3 → neither pair (size 2) qualifies
        nb = detect_network_bursts(
            SYNC_FOUR_PARTIAL, TOTAL_TIME,
            participation_threshold=0.75,
            bin_size_s=0.01,
        )
        assert len(nb) == 0

    def test_participation_fraction_half_for_partial(self):
        nb = detect_network_bursts(
            SYNC_FOUR_PARTIAL, TOTAL_TIME,
            participation_threshold=0.5,
            bin_size_s=0.01,
        )
        for b in nb:
            assert b.participation_fraction == pytest.approx(0.5)


# ── Merging ───────────────────────────────────────────────────────────────────

class TestMerging:
    def test_close_events_merged(self):
        # Two burst pairs separated by 0.5 s < min_network_ibi_s=1.0 s
        close = {
            "A1_11": [_burst(0.010, 0.050), _burst(0.600, 0.640)],
            "A1_12": [_burst(0.010, 0.050), _burst(0.600, 0.640)],
        }
        nb = detect_network_bursts(
            close, TOTAL_TIME,
            participation_threshold=0.5,
            min_network_ibi_s=1.0,
        )
        assert len(nb) == 1

    def test_far_events_not_merged(self):
        nb = detect_network_bursts(
            SYNC_TWO, TOTAL_TIME,
            participation_threshold=0.5,
            min_network_ibi_s=1.0,
        )
        assert len(nb) == 2


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_burst_dict(self):
        assert detect_network_bursts({}, TOTAL_TIME) == []

    def test_all_silent_electrodes(self):
        assert detect_network_bursts(
            {"A1_11": [], "A1_12": []}, TOTAL_TIME
        ) == []

    def test_single_active_electrode_below_min_electrodes(self):
        nb = detect_network_bursts(
            {"A1_11": [_burst(0.01, 0.05)]}, TOTAL_TIME,
            min_electrodes=2,
        )
        assert nb == []

    def test_single_electrode_with_min_electrodes_1(self):
        nb = detect_network_bursts(
            {"A1_11": [_burst(0.01, 0.05)]}, TOTAL_TIME,
            participation_threshold=1.0,
            min_electrodes=1,
        )
        assert len(nb) == 1

    def test_staggered_no_overlap_full_threshold(self):
        # participation_threshold=1.0 → both must burst simultaneously → no NB
        nb = detect_network_bursts(
            STAGGERED, TOTAL_TIME,
            participation_threshold=1.0,
            bin_size_s=0.01,
        )
        assert len(nb) == 0

    def test_staggered_half_threshold_finds_separate_events(self):
        # threshold=0.5 → either electrode alone qualifies → 2 network bursts
        nb = detect_network_bursts(
            STAGGERED, TOTAL_TIME,
            participation_threshold=0.5,
            bin_size_s=0.01,
            min_network_ibi_s=0.05,   # short IBI so they're not merged
        )
        assert len(nb) == 2

    def test_participating_electrodes_sorted(self):
        nb = detect_network_bursts(
            SYNC_TWO, TOTAL_TIME, participation_threshold=0.5
        )
        for b in nb:
            assert b.participating_electrodes == sorted(b.participating_electrodes)

    def test_participation_fraction_in_range(self):
        nb = detect_network_bursts(
            SYNC_TWO, TOTAL_TIME, participation_threshold=0.5
        )
        for b in nb:
            assert 0.0 <= b.participation_fraction <= 1.0
            assert 0.0 <= b.peak_participation <= 1.0


# ── Real-data smoke test ──────────────────────────────────────────────────────

class TestRealDataSmoke:
    def test_network_detection_on_real_well(self):
        from pathlib import Path
        spk = Path("LGI2 KD data/20251004_LGI2 KD_Plate 1_D28N(000).spk")
        if not spk.exists():
            pytest.skip("Real .spk file not available")

        from py_mea_axion.burst.detection import detect_bursts
        from py_mea_axion.io.spk_reader import load_spikes_from_spk

        result, total = load_spikes_from_spk(spk, wells=["A1"])
        well_bursts = {
            eid: detect_bursts(ts)
            for eid, ts in result.items()
            if eid.startswith("A1")
        }
        nb = detect_network_bursts(well_bursts, total)

        # Structural checks — don't assert specific counts.
        for b in nb:
            assert isinstance(b, NetworkBurst)
            assert b.duration >= 0
            assert b.end_time >= b.start_time
            assert 0.0 <= b.participation_fraction <= 1.0
            assert len(b.participating_electrodes) >= 1
