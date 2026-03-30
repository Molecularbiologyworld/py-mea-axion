"""
tests/test_spk_reader.py
========================
Tests for py_mea_axion.io.spk_reader.

The synthetic .spk builder at the bottom of this file constructs a
minimal but fully valid binary that exercises every code path in
load_spikes_from_spk():

  Primary header → ChannelArray → BlockVectorHeader → BlockVectorData

One channel is declared (well A1, electrode row 1 col 1 → "A1_11") with
two spike records at 1.0 s and 2.0 s.

Binary layout
─────────────
Offset  Size  Content
      0     8  Magic "AxionBio"
      8     2  PrimaryDataType = 0
     10     2  HeaderVersionMajor = 1
     12     2  HeaderVersionMinor = 0
     14     8  NotesStart = 0
     22     4  NotesLength = 600
     26     8  EntriesStart = 1022
     34   984  EntrySlots (123 × uint64):
               slot 0: type=0x02 (ChannelArray),       length=16
               slot 1: type=0x03 (BlockVectorHeader),  length=64
               slot 2: type=0x04 (BlockVectorData),    length=100
               slots 3–122: all zero (Terminate)
   1018     4  CRC32 = 0 (not validated by parser)
── entry content (starting at offset 1022 = EntriesStart) ──
   1022    16  ChannelArray content
   1038    64  BlockVectorHeader content
   1102   100  BlockVectorData content (2 × 50-byte spike records)

BlockVectorHeader (64 bytes, read by _read_bv_header):
  [0 :  8]  double fs        = 12 500.0
  [8 : 16]  double vs        = 1.0
  [16: 36]  20 zero bytes    (DateTime fields — skipped)
  [36: 44]  int64  FirstBlock = 0
  [44: 48]  uint32 n_ch      = 1
  [48: 52]  uint32 n_samp    = 10
  [52: 56]  uint32 hdr_sz    = 30
  [56: 64]  8 zero bytes     (padding to 64)

  RecordSize = 30 + 1×10×2 = 50 bytes

Spike record (50 bytes):
  [ 0:  8]  int64  StartingSample
  [  8]     uint8  HardwareChannelIndex (chi)
  [  9]     uint8  HardwareChannelAchk  (achk)
  [10: 14]  uint32 TriggerSampleOffset  = 0
  [14: 30]  16 zero bytes (StdDev + ThreshMult)
  [30: 50]  20 zero bytes (waveform, 10 × int16)

  SpikeTime = (StartingSample + 0) / 12500
  Record 0:  StartingSample = 12 500  → t = 1.0 s
  Record 1:  StartingSample = 25 000  → t = 2.0 s
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from py_mea_axion.io.spk_reader import (
    CHANNELS_PER_WELL,
    PLATE_COLS,
    PLATE_ROWS,
    build_well_list,
    electrode_id,
    load_spikes_from_spk,
    well_label,
)


# ── Synthetic binary builder ──────────────────────────────────────────────────

def _pack_slot(type_id: int, length: int) -> bytes:
    """Pack one uint64 entry slot: top byte = type_id, lower 7 bytes = length."""
    return struct.pack("<Q", (type_id << 56) | (length & 0x00FFFFFFFFFFFFFF))


def build_synthetic_spk(
    spike_samples: list[int],
    fs: float = 12_500.0,
    n_samp: int = 10,
    well_row: int = 1,
    well_col: int = 1,
    elec_row: int = 1,
    elec_col: int = 1,
    achk: int = 0,
    chi: int = 0,
) -> bytes:
    """Construct a minimal but fully valid .spk binary in memory.

    Parameters
    ----------
    spike_samples : list of int
        StartingSample values for each spike.  SpikeTime = sample / fs.
    fs : float
        Sampling frequency written into the BlockVectorHeader.
    well_row, well_col : int
        1-based well position for the single channel declared in ChannelArray.
    elec_row, elec_col : int
        1-based electrode position within the well.
    achk, chi : int
        ChannelAchk and ChannelIndex bytes (must match the values embedded
        in every spike record).

    Returns
    -------
    bytes
        A complete .spk file ready to be written to disk or parsed directly.
    """
    N_SAMP = n_samp
    HDR_SZ = 30
    REC_SZ = HDR_SZ + 1 * N_SAMP * 2  # default 50; 68 when n_samp=19

    # ── Entry content ─────────────────────────────────────────────────────────

    # ChannelArray (16 bytes):
    #   uint32 PlateType + uint32 NumChannels + 1 × ChannelMapping (8 bytes)
    chan_content = bytearray(16)
    struct.pack_into("<I", chan_content, 0, 0)   # PlateType
    struct.pack_into("<I", chan_content, 4, 1)   # NumChannels
    # ChannelMapping: WellCol WellRow ElecCol ElecRow Achk Chi AuxData(uint16)
    struct.pack_into(
        "<BBBBBBH", chan_content, 8,
        well_col, well_row, elec_col, elec_row, achk, chi, 0,
    )

    # BlockVectorHeader (64 bytes):
    bvh_content = bytearray(64)
    struct.pack_into("<d", bvh_content, 0, fs)          # SamplingFrequency
    struct.pack_into("<d", bvh_content, 8, 1.0)         # VoltageScale
    # bytes 16–35: DateTime fields (zeros — skipped by parser)
    struct.pack_into("<q", bvh_content, 36, 0)          # FirstBlock
    struct.pack_into("<I", bvh_content, 44, 1)          # NumChannelsPerBlock
    struct.pack_into("<I", bvh_content, 48, N_SAMP)     # NumSamplesPerBlock
    struct.pack_into("<I", bvh_content, 52, HDR_SZ)     # BlockHeaderSize

    # BlockVectorData (len(spike_samples) × REC_SZ bytes):
    bvd_content = bytearray()
    for samp in spike_samples:
        rec = bytearray(REC_SZ)
        struct.pack_into("<q", rec, 0, samp)    # StartingSample
        rec[8] = chi                            # HardwareChannelIndex
        rec[9] = achk                           # HardwareChannelAchk
        struct.pack_into("<I", rec, 10, 0)      # TriggerSampleOffset = 0
        # bytes 14–29: StdDev, ThreshMult (zeros)
        # bytes 30–49: waveform (zeros)
        bvd_content += rec

    chan_len = len(chan_content)   # 16
    bvh_len = len(bvh_content)    # 64
    bvd_len = len(bvd_content)    # len(spike_samples) * REC_SZ

    # ── Entry slots (123 × uint64) ────────────────────────────────────────────
    slots = bytearray(123 * 8)
    struct.pack_into("<Q", slots,  0, (0x02 << 56) | chan_len)  # ChannelArray
    struct.pack_into("<Q", slots,  8, (0x03 << 56) | bvh_len)  # BVHeader
    struct.pack_into("<Q", slots, 16, (0x04 << 56) | bvd_len)  # BVData
    # slots 3–122: all zeros → Terminate (type 0x00, length 0)

    ENTRIES_START = 1022  # 8+2+2+2+8+4+8 + 984 + 4

    # ── Primary header ────────────────────────────────────────────────────────
    header = bytearray()
    header += b"AxionBio"                        # magic       (8)
    header += struct.pack("<H", 0)               # DataType    (2)
    header += struct.pack("<H", 1)               # Major       (2)
    header += struct.pack("<H", 0)               # Minor       (2)
    header += struct.pack("<Q", 0)               # NotesStart  (8)
    header += struct.pack("<I", 600)             # NotesLength (4)
    header += struct.pack("<q", ENTRIES_START)   # EntriesStart(8)
    header += slots                              # 984 bytes
    header += struct.pack("<I", 0)               # CRC32       (4)

    assert len(header) == ENTRIES_START, (
        f"Header size mismatch: {len(header)} != {ENTRIES_START}"
    )

    return bytes(header) + bytes(chan_content) + bytes(bvh_content) + bytes(bvd_content)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_build_well_list_length(self):
        wells = build_well_list()
        assert len(wells) == 24

    def test_build_well_list_first_last(self):
        wells = build_well_list()
        assert wells[0] == "A1"
        assert wells[-1] == "D6"

    def test_well_label(self):
        assert well_label(1, 1) == "A1"
        assert well_label(4, 6) == "D6"
        assert well_label(2, 3) == "B3"

    def test_electrode_id_format(self):
        assert electrode_id(1, 1, 1, 1) == "A1_11"
        assert electrode_id(4, 6, 4, 4) == "D6_44"
        assert electrode_id(2, 3, 1, 4) == "B3_14"

    def test_plate_constants(self):
        assert len(PLATE_ROWS) == 4
        assert PLATE_COLS == 6
        assert CHANNELS_PER_WELL == 16


class TestSyntheticSpkRoundTrip:
    """Round-trip: build a synthetic .spk, parse it, check results."""

    SPIKE_SAMPLES = [12_500, 25_000]  # → 1.0 s, 2.0 s at fs=12500

    @pytest.fixture()
    def spk_file(self, tmp_path: Path) -> Path:
        data = build_synthetic_spk(
            spike_samples=self.SPIKE_SAMPLES,
            fs=12_500.0,
            well_row=1, well_col=1,
            elec_row=1, elec_col=1,
            achk=0, chi=0,
        )
        p = tmp_path / "test.spk"
        p.write_bytes(data)
        return p

    def test_returns_correct_electrode_id(self, spk_file: Path):
        result, _ = load_spikes_from_spk(spk_file)
        assert "A1_11" in result

    def test_spike_timestamps_correct(self, spk_file: Path):
        result, _ = load_spikes_from_spk(spk_file)
        ts = result["A1_11"]
        np.testing.assert_array_almost_equal(ts, [1.0, 2.0], decimal=6)

    def test_total_time_s(self, spk_file: Path):
        _, total = load_spikes_from_spk(spk_file)
        assert abs(total - 2.0) < 1e-9

    def test_timestamps_sorted(self, spk_file: Path):
        result, _ = load_spikes_from_spk(spk_file)
        ts = result["A1_11"]
        assert np.all(np.diff(ts) >= 0), "Timestamps are not sorted"

    def test_result_is_numpy_float64(self, spk_file: Path):
        result, _ = load_spikes_from_spk(spk_file)
        ts = result["A1_11"]
        assert isinstance(ts, np.ndarray)
        assert ts.dtype == np.float64

    def test_wells_filter_includes(self, spk_file: Path):
        result, _ = load_spikes_from_spk(spk_file, wells=["A1"])
        assert "A1_11" in result
        assert len(result["A1_11"]) == 2

    def test_wells_filter_excludes(self, spk_file: Path):
        result, _ = load_spikes_from_spk(spk_file, wells=["B2"])
        # A1_11 should not be present when we only requested B2
        assert "A1_11" not in result

    def test_none_wells_returns_all(self, spk_file: Path):
        result_all, _ = load_spikes_from_spk(spk_file, wells=None)
        result_a1, _ = load_spikes_from_spk(spk_file, wells=["A1"])
        assert "A1_11" in result_all
        np.testing.assert_array_equal(result_all["A1_11"], result_a1["A1_11"])


class TestSyntheticSpkTriggerOffset:
    """Verify TriggerSampleOffset shifts the timestamp correctly."""

    def test_trigger_offset_applied(self, tmp_path: Path):
        # Manually build one spike record with TriggerSampleOffset = 625
        # StartingSample=0, TriggerOffset=625, fs=12500 → t = 0.05 s
        N_SAMP = 10
        HDR_SZ = 30
        REC_SZ = HDR_SZ + 1 * N_SAMP * 2

        rec = bytearray(REC_SZ)
        struct.pack_into("<q", rec, 0, 0)      # StartingSample = 0
        rec[8] = 0                             # chi
        rec[9] = 0                             # achk
        struct.pack_into("<I", rec, 10, 625)   # TriggerSampleOffset

        # Build a file with one channel declaration matching achk=0, chi=0
        file_bytes = build_synthetic_spk(
            spike_samples=[0],  # will be overwritten below
            fs=12_500.0,
        )
        # Re-inject our custom record at the BVData offset (1022+16+64=1102)
        file_ba = bytearray(file_bytes)
        file_ba[1102 : 1102 + REC_SZ] = rec
        p = tmp_path / "trigger_offset.spk"
        p.write_bytes(bytes(file_ba))

        result, total = load_spikes_from_spk(p)
        np.testing.assert_almost_equal(result["A1_11"][0], 0.05, decimal=6)


class TestFsOverride:
    def test_fs_override_changes_timestamps(self, tmp_path: Path):
        # Build file with fs=12500 in BVHeader; override to 25000.
        # Same StartingSample=12500 → should give 0.5 s instead of 1.0 s.
        data = build_synthetic_spk(spike_samples=[12_500], fs=12_500.0)
        p = tmp_path / "override.spk"
        p.write_bytes(data)
        result_normal, _ = load_spikes_from_spk(p)
        result_override, _ = load_spikes_from_spk(p, fs_override=25_000.0)
        np.testing.assert_almost_equal(result_normal["A1_11"][0], 1.0, decimal=6)
        np.testing.assert_almost_equal(result_override["A1_11"][0], 0.5, decimal=6)

    def test_fs_override_used_in_fallback(self, tmp_path: Path):
        # Build a file then replace the BVHeader slot with a Skip entry so the
        # parser uses the fallback path.  We cannot zero it (type 0x00 =
        # Terminate, which stops the walker before reaching BVData).
        # Slot 1 is at bytes 34 + 8 = 42; replace with Skip (0xFF), length 64.
        #
        # n_samp=19 → RecordSize = 30 + 1*19*2 = 68, which is in the fallback
        # candidate list, so the probe succeeds without a BVHeader.
        import struct as _struct
        data = bytearray(
            build_synthetic_spk(spike_samples=[25_000], fs=12_500.0, n_samp=19)
        )
        _struct.pack_into("<Q", data, 42, (0xFF << 56) | 64)
        p = tmp_path / "no_bvhdr.spk"
        p.write_bytes(bytes(data))
        # StartingSample=25000, fs_override=25000 → 1.0 s
        result, _ = load_spikes_from_spk(p, fs_override=25_000.0)
        np.testing.assert_almost_equal(result["A1_11"][0], 1.0, decimal=6)


class TestBadMagic:
    def test_raises_on_wrong_magic(self, tmp_path: Path):
        bad = b"BadMagic" + bytes(1014)
        p = tmp_path / "bad.spk"
        p.write_bytes(bad)
        with pytest.raises(ValueError, match="AxionBio"):
            load_spikes_from_spk(p)


class TestUnsupportedVersion:
    def test_raises_on_v0(self, tmp_path: Path):
        data = bytearray(build_synthetic_spk([12_500]))
        # Overwrite HeaderVersionMajor at offset 10 with 0
        struct.pack_into("<H", data, 10, 0)
        p = tmp_path / "v0.spk"
        p.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="v0"):
            load_spikes_from_spk(p)

    def test_raises_on_v2(self, tmp_path: Path):
        data = bytearray(build_synthetic_spk([12_500]))
        struct.pack_into("<H", data, 10, 2)
        p = tmp_path / "v2.spk"
        p.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Unsupported"):
            load_spikes_from_spk(p)


class TestMultipleSpikes:
    def test_ten_spikes(self, tmp_path: Path):
        samples = list(range(12_500, 12_500 * 11, 12_500))  # 1s … 10s
        data = build_synthetic_spk(spike_samples=samples, fs=12_500.0)
        p = tmp_path / "ten.spk"
        p.write_bytes(data)
        result, total = load_spikes_from_spk(p)
        ts = result["A1_11"]
        assert len(ts) == 10
        np.testing.assert_array_almost_equal(
            ts, [float(i) for i in range(1, 11)], decimal=6
        )
        assert abs(total - 10.0) < 1e-9
