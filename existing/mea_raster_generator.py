#!/usr/bin/env python3
"""
mea_raster_generator.py
=======================
Python replication of the MATLAB MEA analysis pipeline (Axion_TVAloopv3 / v4).

Generates per-well figures with:
  - Top panel  : ASDR histogram  (spike-count histogram, 200 ms bins, summed
                 across all 16 electrodes)
  - Lower rows : Raster plot     (16 electrode rows, one vertical tick per spike)

Also saves:
  - Standalone ASDR bar-chart PNG
  - Statistics CSV

SUPPORTED INPUT FILES:
  .spk  — Axion spike file (recommended; no REC_SECONDS needed)
  .raw  — Axion Maestro continuous raw recording (spike detection on-the-fly)
  .npz  — Pre-parsed spike cache from spk_parse_to_npz_v2.py

DEPENDENCIES:
  pip install numpy matplotlib scipy
"""

import argparse
import csv
import math
import mmap
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks


# ── Internal defaults (do not normally need to change) ────────
_DEFAULT_REFRACTORY_MS = 1.0

# ═══════════════════════════════════════════════════════════
#  PLATE GEOMETRY  (Axion 24-well CytoView)
# ═══════════════════════════════════════════════════════════
PLATE_ROWS        = "ABCD"
PLATE_COLS        = 6
CHANNELS_PER_WELL = 16
TOTAL_CHANNELS    = len(PLATE_ROWS) * PLATE_COLS * CHANNELS_PER_WELL  # 384


# ───────────────────────────────────────────
#  Well / channel helpers
# ───────────────────────────────────────────

def build_well_list() -> List[str]:
    return [f"{r}{c}" for r in PLATE_ROWS for c in range(1, PLATE_COLS + 1)]

def well_channels_1based(well: str) -> List[int]:
    """Global 1-based channel numbers for a given well (raw-file convention)."""
    all_wells = build_well_list()
    idx = all_wells.index(well.upper())
    start = idx * CHANNELS_PER_WELL + 1
    return list(range(start, start + CHANNELS_PER_WELL))

def well_label(row_1based: int, col_1based: int) -> str:
    return f"{PLATE_ROWS[row_1based - 1]}{col_1based}"


# ═══════════════════════════════════════════════════════════
#  SPK FILE PARSER  — proper implementation based on AxisFile.m
# ═══════════════════════════════════════════════════════════
# Binary format documented from the Axion AxIS MATLAB source:
#
# Primary file header (version 1, little-endian):
#   "AxionBio"  8 bytes
#   uint16      PrimaryDataType
#   uint16      HeaderVersionMajor
#   uint16      HeaderVersionMinor
#   uint64      NotesStart
#   uint32      NotesLength  (must equal 600)
#   int64       EntriesStart  ← file offset where entry CONTENT begins
#   uint64×123  EntrySlots    ← each: top byte=ID, lower 7 bytes=length
#   uint32      CRC32
#
# Entry type IDs (EntryRecordID.m):
#   0x00 Terminate  0x01 NotesArray  0x02 ChannelArray  0x03 BlockVectorHeader
#   0x04 BlockVectorData  0x05 BlockVectorHeaderExtension  0x06 Tag  0xFF Skip
#
# Entry content (sequential, starting at EntriesStart):
#
#   ChannelArray (0x02):
#     uint32 PlateType, uint32 NumChannels
#     NumChannels × ChannelMapping (8 bytes each):
#       uint8 WellColumn, WellRow, ElectrodeColumn, ElectrodeRow,
#             ChannelAchk, ChannelIndex
#       uint16 AuxData
#
#   BlockVectorHeader (0x03) [also attempted for unknown IDs — some file versions
#   use non-standard IDs for this entry]:
#     double SamplingFrequency (8)
#     double VoltageScale (8)
#     DateTime FileStartTime (7×uint16 = 14)
#     DateTime ExperimentStartTime (7×uint16 = 14)
#     int64  FirstBlock (8)
#     uint32 NumChannelsPerBlock (4)
#     uint32 NumSamplesPerBlock  (4)
#     uint32 BlockHeaderSize     (4)
#     Total: 64 bytes (BlockVectorHeader.SIZE = 64)
#
#   BlockVectorData (0x04):
#     Spike records packed back-to-back. Per record:
#       int64  StartingSample      (bytes  0–7)
#       uint8  HardwareChannelIndex (byte   8)
#       uint8  HardwareChannelAchk  (byte   9)
#       uint32 TriggerSampleOffset  (bytes 10–13)
#       double StandardDeviation    (bytes 14–21)  — from Spike_v1.LOADED_HEADER_SIZE=30
#       double ThresholdMultiplier  (bytes 22–29)
#       int16  waveform×NumSamplesPerBlock (bytes 30+)
#     Record size = BlockHeaderSize + NumChannelsPerBlock×NumSamplesPerBlock×2
#     Spike time (s) = (StartingSample + TriggerSampleOffset) / SamplingFrequency

_MAGIC       = b"AxionBio"
_PRIMARY_MAX = 123       # PRIMARY_HEADER_MAXENTRIES
_SUB_MAX     = 126       # SUBHEADER_MAXENTRIES
_CRC_SIZE    = 1018      # PRIMARY_HEADER_CRCSIZE (bytes covered by CRC)
_SUB_CRC     = 1016      # SUBHEADER_CRCSIZE

_ID_TERMINATE = 0x00
_ID_NOTES     = 0x01
_ID_CHANARRAY = 0x02
_ID_BVHEADER  = 0x03
_ID_BVDATA    = 0x04
_ID_BVHDREXT  = 0x05
_ID_TAG       = 0x06
_ID_SKIP      = 0xFF
_KNOWN_IDS    = {_ID_TERMINATE, _ID_NOTES, _ID_CHANARRAY, _ID_BVHEADER,
                 _ID_BVDATA, _ID_BVHDREXT, _ID_TAG, _ID_SKIP}


def _parse_entry_uint64(raw: int) -> Tuple[int, int]:
    """Decode a uint64 entry slot: returns (type_id, length)."""
    type_id = (raw >> 56) & 0xFF
    # length occupies bits 0-55 (7 bytes)
    length  = raw & 0x00FFFFFFFFFFFFFF
    if length == 0x00FFFFFFFFFFFFFF:
        length = None   # "read to end of file" sentinel
    return type_id, length


def _read_channel_array(data: bytes, offset: int, length: int) -> dict:
    """
    Parse ChannelArray entry content.
    Returns dict: (achk, ch_idx) -> (well_row, well_col, elec_num_1to16)
    """
    plate_type, n_ch = struct.unpack_from("<II", data, offset)
    lookup = {}         # (achk, ch_idx) → (well_row, well_col, elec_num)
    # also track order per well so we can assign electrode numbers 1-16
    well_order: Dict[Tuple[int,int], int] = {}

    pos = offset + 8
    for _ in range(n_ch):
        wc, wr, ec, er, achk, chi, aux = struct.unpack_from("<BBBBBBH", data, pos)
        pos += 8
        well_key = (wr, wc)
        elec_num = well_order.get(well_key, 0) + 1
        well_order[well_key] = elec_num
        lookup[(achk, chi)] = (wr, wc, elec_num)

    return lookup


def _read_bv_header(data: bytes, offset: int) -> Optional[dict]:
    """
    Parse BlockVectorHeader entry content (64 bytes).
    Returns None if the content doesn't look like a valid header.
    """
    if offset + 64 > len(data):
        return None
    fs, vs = struct.unpack_from("<dd", data, offset)
    if not (1000.0 < fs < 200_000.0):   # plausibility check for sampling frequency
        return None
    # skip 2 × DateTime (2 × 14 = 28 bytes)
    first_block, n_ch, n_samp, hdr_sz = struct.unpack_from("<qIII", data, offset + 8 + 28)
    return {
        "fs":           fs,
        "voltage_scale": vs,
        "first_block":  first_block,
        "n_ch":         n_ch,
        "n_samp":       n_samp,
        "hdr_sz":       hdr_sz,
    }


def load_spikes_from_spk(
    spk_path: Path,
    wells: List[str],
) -> Tuple[Dict[str, Dict[int, np.ndarray]], float]:
    """
    Parse an Axion .spk file and return per-well spike times.

    Uses the exact binary format specified in:
      AxisFile.m, BlockVectorData.m, ChannelArray.m, ChannelMapping.m, DateTime.m

    Returns
    -------
    spike_data  : dict[well_label] -> dict[electrode_1to16] -> np.ndarray of spike times (s)
    total_time_s: recording duration inferred from last spike time
    """
    raw = spk_path.read_bytes()
    size = len(raw)

    # ── 1. Validate magic ────────────────────────────────────────────────
    if raw[:8] != _MAGIC:
        raise ValueError(f"Not an Axion file (missing 'AxionBio' header): {spk_path.name}")

    # ── 2. Parse primary file header ─────────────────────────────────────
    ptype, major, minor = struct.unpack_from("<HHH", raw, 8)
    notes_start,       = struct.unpack_from("<Q",  raw, 14)
    notes_len,         = struct.unpack_from("<I",  raw, 22)
    if notes_len != 600:
        raise ValueError(f"Unexpected NotesLength {notes_len} (expected 600) — possibly unsupported format")
    entries_start,     = struct.unpack_from("<q",  raw, 26)

    if major == 0:
        raise ValueError("Legacy AxIS v0 file format is not supported by this parser.")
    if major != 1:
        raise ValueError(f"Unsupported file format version {major}.{minor}")

    # Read 123 uint64 entry slots
    slot_offset = 34   # 8+2+2+2+8+4+8 = 34
    entry_records = []
    for i in range(_PRIMARY_MAX):
        raw_u64, = struct.unpack_from("<Q", raw, slot_offset + i * 8)
        type_id, length = _parse_entry_uint64(raw_u64)
        entry_records.append((type_id, length))

    # ── 3. Walk entry content sequentially from entries_start ─────────────
    channel_lookup: Dict[Tuple[int,int], Tuple[int,int,int]] = {}   # (achk,chi)->(wr,wc,en)
    bv_header: Optional[dict] = None
    bv_data_start: Optional[int] = None
    bv_data_length: Optional[int] = None

    # We may need to walk through sub-headers too (for multi-header files).
    # For simplicity, we collect ALL entry records from primary + sub-headers first.
    all_records = list(entry_records)   # primary header records

    # Sub-header detection: if no Terminate in primary, there will be sub-headers in the file.
    # (Sub-headers appear at entries_start + consumed_bytes, and also at further offsets.)
    # We handle this by finding them during content walking.

    pos = int(entries_start)
    rec_idx = 0
    header_sets = [all_records]   # We'll extend if we find sub-headers

    hset_idx = 0
    while hset_idx < len(header_sets):
        cur_records = header_sets[hset_idx]
        hset_idx += 1

        for type_id, length in cur_records:

            if type_id == _ID_TERMINATE:
                # Check for sub-header at current position
                if pos + len(_MAGIC) <= size and raw[pos:pos + 8] == _MAGIC:
                    # Parse sub-header
                    sub_records = []
                    sub_pos = pos + 8  # skip magic
                    for i in range(_SUB_MAX):
                        if sub_pos + 8 > size:
                            break
                        raw_u64, = struct.unpack_from("<Q", raw, sub_pos)
                        sub_pos += 8
                        tid, slen = _parse_entry_uint64(raw_u64)
                        sub_records.append((tid, slen))
                    # Skip CRC (4 bytes) + 4 reserved
                    pos = sub_pos + 8
                    header_sets.append(sub_records)
                break   # done with this header set's records

            if type_id == _ID_NOTES or type_id == _ID_SKIP or type_id == _ID_TAG:
                if length is not None:
                    pos += length
                continue

            if type_id == _ID_CHANARRAY:
                if length is not None and pos + length <= size:
                    channel_lookup = _read_channel_array(raw, pos, length)
                    pos += length
                continue

            if type_id == _ID_BVHEADER:
                hdr = _read_bv_header(raw, pos)
                if hdr is not None:
                    bv_header = hdr
                if length is not None:
                    pos += length
                elif pos + 64 <= size:
                    pos += 64
                continue

            if type_id == _ID_BVHDREXT:
                if length is not None:
                    pos += length
                continue

            if type_id == _ID_BVDATA:
                bv_data_start  = pos
                bv_data_length = length
                if length is not None:
                    pos += length
                continue

            # Unknown entry type — try to interpret as BlockVectorHeader
            # (some newer file versions use IDs not in the v3.1.1.6 enum)
            if bv_header is None and length is not None and pos + 64 <= size:
                hdr = _read_bv_header(raw, pos)
                if hdr is not None:
                    bv_header = hdr
            if length is not None:
                pos += length

    if bv_data_start is None:
        raise ValueError("Could not find BlockVectorData in .spk file.")

    # ── 4. Determine record size ──────────────────────────────────────────
    if bv_header is not None:
        fs      = bv_header["fs"]
        hdr_sz  = bv_header["hdr_sz"]
        n_samp  = bv_header["n_samp"]
        n_ch    = bv_header["n_ch"]
        rec_sz  = int(hdr_sz) + int(n_ch) * int(n_samp) * 2
        print(f"  [spk]  fs={fs:.1f} Hz  BlockHeaderSize={hdr_sz}  "
              f"SamplesPerBlock={n_samp}  RecordSize={rec_sz}")
    else:
        # Fallback: find record size that divides data length evenly
        # and gives sensible spike count (assume ~12500 Hz)
        avail = (bv_data_length if bv_data_length is not None else size - bv_data_start)
        fs = 12500.0
        rec_sz = None
        for sz in [68, 82, 96, 106, 110, 114, 126, 130, 134, 158, 162]:
            if avail % sz == 0:
                rec_sz = sz
                break
        if rec_sz is None:
            raise ValueError("Could not determine spike record size — BlockVectorHeader missing.")
        hdr_sz = 30   # LOADED_HEADER_SIZE
        n_samp = (rec_sz - hdr_sz) // 2
        print(f"  [spk]  WARNING: BlockVectorHeader not found. "
              f"Assumed fs={fs} Hz, RecordSize={rec_sz}")

    # ── 5. Parse spike records ────────────────────────────────────────────
    avail = (bv_data_length if bv_data_length is not None else size - bv_data_start)
    n_spikes = avail // rec_sz
    print(f"  [spk]  {n_spikes:,} spike records at file offset {bv_data_start}")

    if n_spikes == 0:
        raise ValueError("No spike records found in BlockVectorData.")

    # Read all spike records
    spike_ts  = np.empty(n_spikes, dtype=np.float64)
    spike_ach = np.empty(n_spikes, dtype=np.uint8)
    spike_chi = np.empty(n_spikes, dtype=np.uint8)

    view = memoryview(raw)
    for i in range(n_spikes):
        off = bv_data_start + i * rec_sz
        start_samp, = struct.unpack_from("<q",  view, off)      # int64
        chi_val     = raw[off + 8]                              # uint8 channel index
        ach_val     = raw[off + 9]                              # uint8 achk
        trig_off,   = struct.unpack_from("<I",  view, off + 10) # uint32 trigger offset
        spike_ts[i]  = (start_samp + trig_off) / fs
        spike_chi[i] = chi_val
        spike_ach[i] = ach_val

    total_time_s = float(spike_ts.max()) if n_spikes > 0 else 0.0

    # ── 6. Map channels to wells ──────────────────────────────────────────
    target_wells = set(w.upper() for w in wells)
    result: Dict[str, Dict[int, np.ndarray]] = {}

    if channel_lookup:
        # Use the proper ChannelArray lookup
        well_buckets: Dict[Tuple[int,int], Dict[int, List[float]]] = {}
        for i in range(n_spikes):
            key = (int(spike_ach[i]), int(spike_chi[i]))
            info = channel_lookup.get(key)
            if info is None:
                continue
            wr, wc, en = info
            label = well_label(wr, wc)
            if label not in target_wells:
                continue
            if (wr, wc) not in well_buckets:
                well_buckets[(wr, wc)] = {}
            wb = well_buckets[(wr, wc)]
            wb.setdefault(en, []).append(spike_ts[i])

        for (wr, wc), elec_dict in well_buckets.items():
            label = well_label(wr, wc)
            result[label] = {en: np.sort(np.array(ts)) for en, ts in elec_dict.items()}

    else:
        # No ChannelArray found — fall back to 1-based channel index mapping
        # Map raw channel-index bytes to well/electrode positions using the standard
        # 24-well plate ordering (same as raw file convention: ch 1-16 → A1, etc.)
        print("  [spk]  WARNING: ChannelArray not found; using sequential channel ordering.")
        for w in target_wells:
            all_w = build_well_list()
            wi = all_w.index(w)
            result[w] = {}
            for en in range(1, CHANNELS_PER_WELL + 1):
                # global channel (1-based) in sequential order
                global_ch = wi * CHANNELS_PER_WELL + en
                # ChannelIndex byte tends to be 0-based within achk
                # Without the lookup table we can only use a simple heuristic
                mask = (spike_chi.astype(np.int32) == (global_ch - 1) % 256)
                result[w][en] = np.sort(spike_ts[mask])

    # Fill in empty electrodes
    for w in target_wells:
        if w not in result:
            result[w] = {}
        for en in range(1, CHANNELS_PER_WELL + 1):
            result[w].setdefault(en, np.array([], dtype=np.float64))

    return result, total_time_s


# ═══════════════════════════════════════════════════════════
#  RAW FILE SPIKE DETECTION
# ═══════════════════════════════════════════════════════════

def _detect_header_and_fs(raw_path: Path, rec_seconds: float) -> Tuple[int, float]:
    sz = raw_path.stat().st_size
    frame_size = TOTAL_CHANNELS * 2
    header = sz % frame_size
    frames = (sz - header) // frame_size
    if frames == 0:
        raise SystemExit("[ERROR] File appears empty or wrong format.")
    return header, frames / rec_seconds


def _mad_threshold(sig: np.ndarray, k: float) -> float:
    return -k * (np.median(np.abs(sig)) + 1e-12) / 0.6745


def _detect_spikes_1d(sig: np.ndarray, threshold: float, refractory: int) -> np.ndarray:
    peaks, _ = find_peaks(-sig.astype(np.float32), height=-threshold,
                          distance=max(1, refractory))
    return peaks.astype(np.int64)


def load_spikes_from_raw(
    raw_path: Path,
    wells: List[str],
    rec_seconds: float,
    thresh_k: float = 5.0,
    refractory_ms: float = 1.0,
    chunk_seconds: float = 10.0,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Detect spikes per electrode per well from a .raw continuous file."""
    header_bytes, fs = _detect_header_and_fs(raw_path, rec_seconds)
    print(f"  [raw]  header={header_bytes} B   estimated fs≈{fs:.1f} Hz")

    refr_samp  = max(1, int(round(refractory_ms * fs / 1000)))
    frame_size = TOTAL_CHANNELS * 2
    chunk_fr   = int(round(chunk_seconds * fs))

    w2chs     = {w: well_channels_1based(w) for w in wells}
    all_ch1   = sorted({c for chs in w2chs.values() for c in chs})
    tgt_idx   = np.array([c - 1 for c in all_ch1])

    spk_idx: Dict[int, List[int]] = {c: [] for c in all_ch1}
    total_fr = int(rec_seconds * fs)
    pay_end  = header_bytes + total_fr * frame_size
    abs_fr   = 0

    with raw_path.open("rb") as f:
        mm  = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pos = header_bytes
        while pos < pay_end:
            stop    = min(pay_end, pos + chunk_fr * frame_size)
            n_fr    = (stop - pos) // frame_size
            if n_fr <= 0:
                break
            buf = mm[pos: pos + n_fr * frame_size]
            arr = np.frombuffer(buf, dtype="<i2", count=n_fr * TOTAL_CHANNELS)
            if arr.size != n_fr * TOTAL_CHANNELS:
                break
            sub = arr.reshape(n_fr, TOTAL_CHANNELS)[:, tgt_idx].astype(np.float32)
            for j, ch1 in enumerate(all_ch1):
                sig = sub[:, j]
                thr = _mad_threshold(sig, thresh_k)
                lm  = _detect_spikes_1d(sig, thr, refr_samp)
                if lm.size:
                    spk_idx[ch1].extend((abs_fr + lm).tolist())
            abs_fr += n_fr
            pos = stop
        mm.close()

    result: Dict[str, Dict[int, np.ndarray]] = {}
    for w, chs in w2chs.items():
        result[w] = {}
        for en, ch1 in enumerate(chs, start=1):
            idx = np.array(spk_idx.get(ch1, []), dtype=np.int64)
            result[w][en] = idx / fs
    return result


# ═══════════════════════════════════════════════════════════
#  NPZ LOADER
# ═══════════════════════════════════════════════════════════

def load_spikes_from_npz(
    npz_path: Path, wells: List[str], rec_seconds: float
) -> Tuple[Dict[str, Dict[int, np.ndarray]], float]:
    d   = np.load(npz_path, allow_pickle=True)
    ts  = d["timestamps"].astype(np.float64)
    ch  = d["channels"].astype(np.uint16)
    tot = rec_seconds if rec_seconds > 0 else float(ts.max())

    result: Dict[str, Dict[int, np.ndarray]] = {}
    for w in wells:
        chs = well_channels_1based(w)
        result[w] = {}
        for en, ch1 in enumerate(chs, start=1):
            mask = ch == ch1
            result[w][en] = np.sort(ts[mask])
    return result, tot


# ═══════════════════════════════════════════════════════════
#  ASDR  (Activity Synchrony Discharge Rate histogram)
# ═══════════════════════════════════════════════════════════

def compute_asdr(
    spike_times: Dict[int, np.ndarray],
    time_start_s: float,
    time_end_s: float,
    bin_size_ms: float = 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin spikes across all 16 electrodes into bins of bin_size_ms.
    Only spikes within [time_start_s, time_end_s] are counted.
    Matches MATLAB: binsize=200, sums across all electrodes per bin.
    Returns (bin_left_edges_ms, counts).
    """
    start_ms = time_start_s * 1000.0
    end_ms   = time_end_s   * 1000.0
    edges_ms = np.arange(start_ms, end_ms + bin_size_ms, bin_size_ms)
    counts   = np.zeros(max(len(edges_ms) - 1, 1), dtype=np.int64)
    for times_s in spike_times.values():
        if times_s.size:
            ts_ms = times_s * 1000.0
            mask  = (ts_ms >= start_ms) & (ts_ms <= end_ms)
            if mask.any():
                c, _ = np.histogram(ts_ms[mask], bins=edges_ms)
                counts += c
    return edges_ms[:-1], counts


# ═══════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════

def _asdr_thresh(thresh: float) -> float:
    return max(0.0, float(thresh))


def plot_combined(
    well:         str,
    spike_times:  Dict[int, np.ndarray],
    time_start_s: float,
    time_end_s:   float,
    out_path:     Path,
    bin_size_ms:  float          = 200.0,
    asdr_thresh:  float          = 10,
    y_max:        Optional[float] = None,
    dpi:          int            = 150,
) -> None:
    """
    Combined figure: ASDR histogram on top, 16-electrode raster below.
    Matches the visual output of the MATLAB script.
    """
    n_elec = CHANNELS_PER_WELL

    bin_left_ms, asdr_counts = compute_asdr(spike_times, time_start_s, time_end_s, bin_size_ms)
    thresh_line = _asdr_thresh(asdr_thresh)
    auto_max    = max(int(asdr_counts.max()), 1)
    y_top       = float(y_max) if y_max is not None else auto_max + max(1, int(auto_max * 0.05))

    start_ms = time_start_s * 1000.0
    end_ms   = time_end_s   * 1000.0
    span_s   = time_end_s - time_start_s

    fig = plt.figure(figsize=(12, 14))
    gs  = gridspec.GridSpec(
        n_elec + 1, 1,
        height_ratios=[3] + [1] * n_elec,
        hspace=0.04,
        left=0.08, right=0.97, top=0.95, bottom=0.05,
    )

    # ── ASDR histogram (top) ─────────────────────────────────────────────
    ax_asdr = fig.add_subplot(gs[0])
    ax_asdr.bar(bin_left_ms, asdr_counts,
                width=bin_size_ms * 0.9, align="edge", color="#2166ac", linewidth=0)
    ax_asdr.axhline(thresh_line, color="red", linewidth=1.2, linestyle="--",
                    label=f"ASDR threshold = {thresh_line:.0f}")
    ax_asdr.set_xlim(start_ms, end_ms)
    ax_asdr.set_ylim(0, y_top)
    ax_asdr.set_ylabel("Spikes / bin\n(all electrodes)", fontsize=8)
    ax_asdr.set_title(
        f"Well {well}  —  ASDR histogram + Raster  "
        f"[{time_start_s:.0f}–{time_end_s:.0f} s | bin = {bin_size_ms:.0f} ms | "
        f"thresh = {thresh_line:.0f}]",
        fontsize=10,
    )
    ax_asdr.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax_asdr.tick_params(labelbottom=False, bottom=False)
    ax_asdr.spines["top"].set_visible(False)
    ax_asdr.spines["right"].set_visible(False)

    # ── Raster rows ───────────────────────────────────────────────────────
    for elec in range(1, n_elec + 1):
        ax = fig.add_subplot(gs[elec], sharex=ax_asdr)
        ts = spike_times.get(elec, np.array([]))
        ts_ms = ts[(ts >= time_start_s) & (ts <= time_end_s)] * 1000.0
        if ts_ms.size:
            ax.vlines(ts_ms, -1.3, 1.3, colors="black", linewidths=0.5)
        ax.set_ylim(-2, 2)
        ax.set_yticks([])
        ax.set_ylabel(str(elec), rotation=0, fontsize=6, labelpad=14, va="center")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if elec < n_elec:
            ax.tick_params(labelbottom=False, bottom=False)
        else:
            n_ticks = 7
            step    = max(1, int(span_s / (n_ticks - 1)))
            ticks_s = np.arange(time_start_s, time_end_s + step, step)
            ax.set_xticks(ticks_s * 1000)
            ax.set_xticklabels([f"{int(t)}" for t in ticks_s], fontsize=7)
            ax.set_xlabel("Time (s)", fontsize=8)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


def plot_asdr_standalone(
    well:         str,
    spike_times:  Dict[int, np.ndarray],
    time_start_s: float,
    time_end_s:   float,
    out_path:     Path,
    bin_size_ms:  float          = 200.0,
    asdr_thresh:  float          = 10,
    y_max:        Optional[float] = None,
    dpi:          int            = 150,
) -> None:
    """Standalone ASDR histogram — matches MATLAB bursts_XX_YY.tif."""
    bin_left_ms, counts = compute_asdr(spike_times, time_start_s, time_end_s, bin_size_ms)
    thresh_line = _asdr_thresh(asdr_thresh)
    auto_max    = max(int(counts.max()), 1)
    y_top       = float(y_max) if y_max is not None else auto_max + 1

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(bin_left_ms, counts,
           width=bin_size_ms * 0.9, align="edge", color="#2166ac", linewidth=0)
    ax.axhline(thresh_line, color="red", linewidth=1.2, linestyle="--",
               label=f"ASDR threshold = {thresh_line:.0f}")
    ax.set_xlim(time_start_s * 1000.0, time_end_s * 1000.0)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Spike counts / bin", fontsize=9)
    ax.set_title(
        f"Well {well}  —  ASDR Histogram  [{time_start_s:.0f}–{time_end_s:.0f} s]",
        fontsize=10,
    )
    span_s  = time_end_s - time_start_s
    step    = max(1, int(span_s / 6))
    ticks_s = np.arange(time_start_s, time_end_s + step, step)
    ax.set_xticks(ticks_s * 1000)
    ax.set_xticklabels([f"{int(t)}" for t in ticks_s])
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"  [saved] {out_path.name}")


# ═══════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════

def compute_stats(
    well:         str,
    spike_times:  Dict[int, np.ndarray],
    time_start_s: float,
    time_end_s:   float,
    bin_size_ms:  float = 200.0,
    asdr_thresh:  float = 10,
) -> dict:
    duration_s = time_end_s - time_start_s
    # Filter spike times to the time window
    st_win = {e: t[(t >= time_start_s) & (t <= time_end_s)] for e, t in spike_times.items()}

    n_spikes   = {e: int(t.size) for e, t in st_win.items()}
    T_Spikes   = sum(n_spikes.values())
    spk_rates  = {e: n / duration_s for e, n in n_spikes.items() if duration_s > 0}
    MaxERate   = max(spk_rates.values(), default=0.0)

    n_thresh   = max(1, int(max(n_spikes.values(), default=0) * 0.05))
    active_n   = [n for n in n_spikes.values() if n >= n_thresh]
    AvgSpks_E  = float(np.median(active_n)) if active_n else 0.0
    active_r   = [spk_rates[e] for e, n in n_spikes.items() if n >= n_thresh]
    AvgRate_E  = float(np.mean(active_r)) if active_r else 0.0

    all_isi = []
    for ts in st_win.values():
        ts = np.sort(ts)
        if ts.size > 1:
            all_isi.extend((np.diff(ts) * 1000).tolist())
    Avg_ISI = float(np.median(all_isi)) if all_isi else float("nan")

    total_bursts = 0
    for ts in st_win.values():
        ts = np.sort(ts)
        if ts.size < 3:
            continue
        cnt = 0
        for dt in np.diff(ts) * 1000.0:
            if dt < 300:
                cnt += 1
            else:
                if cnt >= 3:
                    total_bursts += 1
                cnt = 0
        if cnt >= 3:
            total_bursts += 1

    bin_left, counts = compute_asdr(st_win, time_start_s, time_end_s, bin_size_ms)
    max_asdr  = int(counts.max()) if counts.size else 0
    thresh    = _asdr_thresh(asdr_thresh)
    peaks, _  = find_peaks(counts.astype(float), height=thresh)

    if len(peaks) > 1:
        peak_ms    = bin_left[peaks] + bin_size_ms / 2.0
        intervals  = np.diff(peak_ms)
        large      = intervals[intervals > 5000]
        Avg_MAP    = float(np.median(large[1:])) if large.size > 1 else float("nan")
        Mean_MAP   = float(np.mean(large[1:]))   if large.size > 1 else float("nan")
    else:
        Avg_MAP = Mean_MAP = float("nan")

    def _fmt(v):
        return round(v, 3) if math.isfinite(v) else "NaN"

    return {
        "Well":                          well,
        "Time window (s)":               f"{time_start_s:.0f}-{time_end_s:.0f}",
        "Total Spikes":                  T_Spikes,
        "Average Spikes/E":              round(AvgSpks_E, 1),
        "Average Spike Rate (Hz)":       _fmt(AvgRate_E),
        "Max Electrode Spike Rate (Hz)": _fmt(MaxERate),
        "Average ISI (ms)":              _fmt(Avg_ISI),
        "Total Detected Bursts":         total_bursts,
        "Average MAP Interval (ms)":     _fmt(Avg_MAP),
        "Maximum ASDR":                  max_asdr,
        "Mean SB Interval (ms)":         _fmt(Mean_MAP),
    }


# ═══════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════

def run(
    input_file,
    output_dir=None,
    wells=None,
    time_start=0.0,
    time_end=0.0,
    asdr_thresh=50,
    combined=True,
    asdr_y_max=None,
    dpi=300,
    rec_seconds=0.0,
    bin_ms=200.0,
    thresh_k=5.0,
):
    """
    Run the MEA raster/ASDR analysis.

    Parameters
    ----------
    input_file : str or Path
        Path to .spk, .raw, or .npz recording file.
    output_dir : str or Path, optional
        Output directory. Defaults to a folder next to the input file.
    wells : list of str, optional
        Wells to analyse, e.g. ["A1", "B2"]. None or ["ALL"] = every well.
    time_start : float
        Start of time window in seconds (default: 0).
    time_end : float
        End of time window in seconds. 0 = full recording.
    asdr_thresh : int
        ASDR threshold spike count drawn as a red dashed line (default: 50).
    combined : bool
        Save combined raster+histogram figure (default: True).
    asdr_y_max : float or None
        Y-axis maximum for ASDR panels. None = autoscale.
    dpi : int
        Figure resolution in DPI (default: 300).
    rec_seconds : float
        Recording duration in seconds. Required for .raw and .npz files.
    bin_ms : float
        ASDR bin width in ms (default: 200).
    thresh_k : float
        Spike threshold multiplier K for MAD thresholding in .raw files (default: 5.0).
    """
    # ── Resolve input file ─────────────────────────────────────────────────
    infile = Path(input_file)
    if not infile.exists():
        raise FileNotFoundError(f"File not found: {infile}")

    suffix = infile.suffix.lower()
    if suffix not in (".spk", ".raw", ".npz"):
        raise ValueError(f"Unsupported file type '{suffix}'. Use .spk, .raw, or .npz")

    # ── Resolve output directory ───────────────────────────────────────────
    out_dir = Path(output_dir) if output_dir else infile.parent / "mea_raster_generator_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] {out_dir}\n")

    # ── Resolve wells ──────────────────────────────────────────────────────
    all_wells = build_well_list()
    if wells is None or wells == ["ALL"]:
        resolved_wells = all_wells
    else:
        resolved_wells = [w.upper() for w in wells]
        bad = [w for w in resolved_wells if w not in all_wells]
        if bad:
            raise ValueError(f"Unknown wells: {bad}. Valid wells: {all_wells}")

    print(f"[INFO] Input    : {infile.name}")
    print(f"[INFO] Wells    : {', '.join(resolved_wells)}")
    print(f"[INFO] ASDR bin : {bin_ms} ms")

    # ── Load spike data ────────────────────────────────────────────────────
    print(f"\n[LOADING] {infile.name} ...")
    if suffix == ".raw":
        if rec_seconds <= 0:
            raise ValueError("rec_seconds is required for .raw files")
        print(f"[INFO] Duration: {rec_seconds} s  |  threshold: -{thresh_k} x sigma_MAD")
        spike_data   = load_spikes_from_raw(infile, resolved_wells, rec_seconds, thresh_k)
        total_time_s = rec_seconds
    elif suffix == ".spk":
        spike_data, total_time_s = load_spikes_from_spk(infile, resolved_wells)
        print(f"[INFO] Inferred recording duration: {total_time_s:.1f} s")
    else:  # .npz
        if rec_seconds <= 0:
            raise ValueError("rec_seconds is required for .npz files")
        spike_data, total_time_s = load_spikes_from_npz(infile, resolved_wells, rec_seconds)

    print(f"[DONE loading]\n")

    # ── Resolve time window ────────────────────────────────────────────────
    t_start = float(time_start)
    t_end   = float(time_end) if time_end > 0 else total_time_s
    t_end   = min(t_end, total_time_s)

    if t_start >= t_end:
        raise ValueError(f"time_start ({t_start} s) must be less than time_end ({t_end} s)")

    print(f"[INFO] Time window: {t_start:.1f} s  to  {t_end:.1f} s\n")

    # ── Per-well figures + stats ───────────────────────────────────────────
    stem      = infile.stem
    all_stats = []

    for well in resolved_wells:
        if well not in spike_data:
            print(f"[SKIP] {well} — no data")
            continue

        st = spike_data[well]
        total_sp = sum(int(((t >= t_start) & (t <= t_end)).sum()) for t in st.values())
        active_e = sum(1 for t in st.values() if ((t >= t_start) & (t <= t_end)).any())

        if total_sp == 0:
            print(f"[SKIP] {well} — 0 spikes in time window")
            continue

        print(f"[WELL {well}]  {total_sp} spikes in window  |  {active_e}/16 active electrodes")

        if combined:
            plot_combined(
                well=well, spike_times=st,
                time_start_s=t_start, time_end_s=t_end,
                out_path=out_dir / f"{stem}_{well}_raster_histogram.png",
                bin_size_ms=bin_ms, asdr_thresh=asdr_thresh,
                y_max=asdr_y_max, dpi=dpi,
            )

        plot_asdr_standalone(
            well=well, spike_times=st,
            time_start_s=t_start, time_end_s=t_end,
            out_path=out_dir / f"{stem}_{well}_asdr_histogram.png",
            bin_size_ms=bin_ms, asdr_thresh=asdr_thresh,
            y_max=asdr_y_max, dpi=dpi,
        )

        all_stats.append(compute_stats(well, st, t_start, t_end, bin_ms, asdr_thresh))

    if all_stats:
        stats_path = out_dir / f"{stem}_stats.csv"
        with stats_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\n[STATS] {stats_path.name}")

    print(f"\n[ALL DONE]  Output: {out_dir}")


# ═══════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="mea-raster-generator",
        description="Raster plot and ASDR analysis for Axion Maestro MEA recordings.",
    )
    parser.add_argument(
        "input_file", metavar="INPUT_FILE",
        help="Path to .spk, .raw, or .npz recording file",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: auto-create folder next to input file)",
    )
    parser.add_argument(
        "--wells", nargs="+", default=["ALL"], metavar="WELL",
        help='Wells to analyse, e.g. --wells A1 B2 C3, or ALL (default: ALL)',
    )
    parser.add_argument(
        "--time-start", type=float, default=0.0,
        help="Start of time window in seconds (default: 0)",
    )
    parser.add_argument(
        "--time-end", type=float, default=0.0,
        help="End of time window in seconds; 0 = full recording (default: 0)",
    )
    parser.add_argument(
        "--asdr-thresh", type=int, default=50,
        help="ASDR threshold spike count drawn as red dashed line (default: 50)",
    )
    parser.add_argument(
        "--combined", action="store_true", default=True,
        help="Save combined raster+histogram figure (default: on)",
    )
    parser.add_argument(
        "--no-combined", dest="combined", action="store_false",
        help="Save ASDR histogram only, skip combined raster figure",
    )
    parser.add_argument(
        "--asdr-y-max", type=float, default=0.0,
        help="Y-axis maximum for ASDR panels; 0 = autoscale (default: 0)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure resolution in DPI (default: 300)",
    )
    parser.add_argument(
        "--rec-seconds", type=float, default=0.0,
        help="Recording duration in seconds; required for .raw and .npz files",
    )
    parser.add_argument(
        "--bin-ms", type=float, default=200.0,
        help="ASDR bin width in ms (default: 200)",
    )
    parser.add_argument(
        "--thresh-k", type=float, default=5.0,
        help="Spike threshold multiplier K for MAD thresholding in .raw files (default: 5.0)",
    )

    args = parser.parse_args()

    try:
        run(
            input_file  = args.input_file,
            output_dir  = args.output_dir,
            wells       = args.wells,
            time_start  = args.time_start,
            time_end    = args.time_end,
            asdr_thresh = args.asdr_thresh,
            combined    = args.combined,
            asdr_y_max  = args.asdr_y_max if args.asdr_y_max > 0 else None,
            dpi         = args.dpi,
            rec_seconds = args.rec_seconds,
            bin_ms      = args.bin_ms,
            thresh_k    = args.thresh_k,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
