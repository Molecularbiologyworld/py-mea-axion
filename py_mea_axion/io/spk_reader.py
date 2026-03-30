"""
spk_reader.py
=============
Parser for Axion BioSystems .spk binary files.

Binary format reverse-engineered from the Axion AxIS MATLAB source
(AxisFile.m, BlockVectorData.m, ChannelArray.m, ChannelMapping.m,
DateTime.m).  All parsing logic is preserved exactly from the
original ``mea_raster_generator.py`` implementation.

Public API
----------
load_spikes_from_spk(spk_path, wells=None)
    Parse a .spk file and return per-electrode spike timestamps.

Plate geometry helpers
----------------------
PLATE_ROWS, PLATE_COLS, CHANNELS_PER_WELL, TOTAL_CHANNELS
build_well_list(), well_label(), electrode_id()
"""

import logging
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Plate geometry (Axion 24-well CytoView MEA) ──────────────────────────────

PLATE_ROWS: str = "ABCD"          # row labels, top to bottom
PLATE_COLS: int = 6               # column count
CHANNELS_PER_WELL: int = 16       # 4×4 electrode grid
TOTAL_CHANNELS: int = (           # 384 total
    len(PLATE_ROWS) * PLATE_COLS * CHANNELS_PER_WELL
)

# Axion 4×4 electrode grid: rows 1-4, columns 1-4, 16 electrodes per well.
ELECTRODES_PER_ROW: int = 4
ELECTRODES_PER_COL: int = 4


# ── Well / electrode label helpers ───────────────────────────────────────────

def build_well_list() -> List[str]:
    """Return all 24 well labels in plate order (A1…D6).

    Returns
    -------
    list of str
        E.g. ``['A1', 'A2', ..., 'D6']``.
    """
    return [f"{r}{c}" for r in PLATE_ROWS for c in range(1, PLATE_COLS + 1)]


def well_label(row_1based: int, col_1based: int) -> str:
    """Convert 1-based well row/column indices to a well label string.

    Parameters
    ----------
    row_1based : int
        Well row index (1 = A, 2 = B, 3 = C, 4 = D).
    col_1based : int
        Well column index (1–6).

    Returns
    -------
    str
        E.g. ``'A1'``, ``'D6'``.
    """
    return f"{PLATE_ROWS[row_1based - 1]}{col_1based}"


def electrode_id(
    well_row: int,
    well_col: int,
    elec_row: int,
    elec_col: int,
) -> str:
    """Build the canonical electrode identifier string.

    Parameters
    ----------
    well_row : int
        1-based well row (1 = A … 4 = D).
    well_col : int
        1-based well column (1–6).
    elec_row : int
        1-based electrode row within the well (1–4).
    elec_col : int
        1-based electrode column within the well (1–4).

    Returns
    -------
    str
        Electrode ID following Axion convention, e.g. ``'A1_11'``,
        ``'A1_44'``, ``'D6_23'``.
    """
    return f"{PLATE_ROWS[well_row - 1]}{well_col}_{elec_row}{elec_col}"


def well_channels_1based(well: str) -> List[int]:
    """Return global 1-based channel numbers for a well (raw-file convention).

    Parameters
    ----------
    well : str
        Well label, e.g. ``'A1'``.

    Returns
    -------
    list of int
        16 consecutive global channel numbers.
    """
    all_wells = build_well_list()
    idx = all_wells.index(well.upper())
    start = idx * CHANNELS_PER_WELL + 1
    return list(range(start, start + CHANNELS_PER_WELL))


# ── .spk binary format constants ─────────────────────────────────────────────
#
# Primary file header (version 1, little-endian):
#   "AxionBio"  8 bytes  — magic
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
# BlockVectorHeader (0x03) — 64 bytes:
#   double SamplingFrequency (8)
#   double VoltageScale (8)
#   DateTime FileStartTime (7×uint16 = 14)
#   DateTime ExperimentStartTime (7×uint16 = 14)
#   int64  FirstBlock (8)
#   uint32 NumChannelsPerBlock (4)
#   uint32 NumSamplesPerBlock  (4)
#   uint32 BlockHeaderSize     (4)
#
# BlockVectorData (0x04) — spike records packed back-to-back:
#   int64  StartingSample      (bytes  0–7)
#   uint8  HardwareChannelIndex (byte   8)
#   uint8  HardwareChannelAchk  (byte   9)
#   uint32 TriggerSampleOffset  (bytes 10–13)
#   double StandardDeviation    (bytes 14–21)
#   double ThresholdMultiplier  (bytes 22–29)
#   int16  waveform×NumSamplesPerBlock (bytes 30+)
#   RecordSize = BlockHeaderSize + NumChannelsPerBlock×NumSamplesPerBlock×2
#   SpikeTime (s) = (StartingSample + TriggerSampleOffset) / SamplingFrequency

_MAGIC: bytes = b"AxionBio"
_PRIMARY_MAX: int = 123   # PRIMARY_HEADER_MAXENTRIES
_SUB_MAX: int = 126       # SUBHEADER_MAXENTRIES

_ID_TERMINATE: int = 0x00
_ID_NOTES: int = 0x01
_ID_CHANARRAY: int = 0x02
_ID_BVHEADER: int = 0x03
_ID_BVDATA: int = 0x04
_ID_BVHDREXT: int = 0x05
_ID_TAG: int = 0x06
_ID_SKIP: int = 0xFF


# ── Private parsing helpers ───────────────────────────────────────────────────

def _parse_entry_uint64(raw: int) -> Tuple[int, Optional[int]]:
    """Decode a uint64 entry slot into ``(type_id, length)``.

    Parameters
    ----------
    raw : int
        The raw uint64 value read from the entry slot array.

    Returns
    -------
    type_id : int
        Entry type identifier (top byte of the uint64).
    length : int or None
        Byte length of the entry content.  ``None`` when the sentinel
        ``0x00FFFFFFFFFFFFFF`` is stored, meaning "read to end of file".
    """
    type_id = (raw >> 56) & 0xFF
    length = raw & 0x00FFFFFFFFFFFFFF
    if length == 0x00FFFFFFFFFFFFFF:
        length = None  # sentinel: read to end of file
    return type_id, length


def _read_channel_array(
    data: bytes,
    offset: int,
    length: int,
) -> Dict[Tuple[int, int], Tuple[int, int, int, int]]:
    """Parse a ChannelArray entry.

    Parameters
    ----------
    data : bytes
        Full file contents.
    offset : int
        Byte offset of the ChannelArray content within *data*.
    length : int
        Byte length of the ChannelArray content.

    Returns
    -------
    dict
        Mapping ``(achk, channel_index)`` →
        ``(well_row, well_col, elec_row, elec_col)``, all 1-based.
    """
    _plate_type, n_ch = struct.unpack_from("<II", data, offset)
    lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}

    pos = offset + 8
    for _ in range(n_ch):
        # Layout: WellColumn, WellRow, ElectrodeColumn, ElectrodeRow,
        #         ChannelAchk, ChannelIndex, AuxData(uint16)
        wc, wr, ec, er, achk, chi, _aux = struct.unpack_from("<BBBBBBH", data, pos)
        pos += 8
        lookup[(achk, chi)] = (wr, wc, er, ec)

    return lookup


def _read_bv_header(data: bytes, offset: int) -> Optional[Dict]:
    """Parse a BlockVectorHeader entry (64 bytes).

    Parameters
    ----------
    data : bytes
        Full file contents.
    offset : int
        Byte offset of the BlockVectorHeader content within *data*.

    Returns
    -------
    dict or None
        Parsed header fields, or ``None`` if the content does not look
        like a valid header (plausibility check on sampling frequency).

        Keys: ``fs``, ``voltage_scale``, ``first_block``, ``n_ch``,
        ``n_samp``, ``hdr_sz``.
    """
    if offset + 64 > len(data):
        return None
    fs, vs = struct.unpack_from("<dd", data, offset)
    if not (1000.0 < fs < 200_000.0):  # plausibility check
        return None
    # Skip VoltageScale (8) + two DateTime structs (28 bytes combined as
    # stored in this file version) to reach FirstBlock.
    first_block, n_ch, n_samp, hdr_sz = struct.unpack_from(
        "<qIII", data, offset + 8 + 28
    )
    return {
        "fs": fs,
        "voltage_scale": vs,
        "first_block": first_block,
        "n_ch": n_ch,
        "n_samp": n_samp,
        "hdr_sz": hdr_sz,
    }


# ── Public loader ─────────────────────────────────────────────────────────────

def load_spikes_from_spk(
    spk_path: Path,
    wells: Optional[List[str]] = None,
    fs_override: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Parse an Axion .spk file and return per-electrode spike timestamps.

    Uses the exact binary format specified in AxisFile.m,
    BlockVectorData.m, ChannelArray.m, ChannelMapping.m, and DateTime.m.
    Handles multi-header files (sub-headers) and falls back gracefully
    when the BlockVectorHeader entry is absent.

    Parameters
    ----------
    spk_path : Path
        Path to the .spk binary file.
    wells : list of str, optional
        Well labels to include, e.g. ``['A1', 'B3']``.  ``None`` (default)
        returns all wells present in the file.
    fs_override : float, optional
        Sampling frequency in Hz.  When provided this value is used
        unconditionally, overriding both the value parsed from the
        BlockVectorHeader (if present) and the 12 500 Hz fallback
        assumption.  Useful when the file lacks a BlockVectorHeader
        and the default fallback is incorrect.

    Returns
    -------
    spike_data : dict[str, np.ndarray]
        Mapping from electrode ID (e.g. ``'A1_11'``) to a sorted
        1-D array of spike timestamps in **seconds**.
    total_time_s : float
        Recording duration inferred from the last spike timestamp.

    Raises
    ------
    ValueError
        If the file is not a valid Axion file, uses an unsupported format
        version, or if no spike data can be found.
    """
    spk_path = Path(spk_path)
    raw: bytes = spk_path.read_bytes()
    size: int = len(raw)

    # ── 1. Validate magic ─────────────────────────────────────────────────────
    if raw[:8] != _MAGIC:
        raise ValueError(
            f"Not an Axion file (missing 'AxionBio' header): {spk_path.name}"
        )

    # ── 2. Parse primary file header ──────────────────────────────────────────
    _ptype, major, minor = struct.unpack_from("<HHH", raw, 8)
    notes_start, = struct.unpack_from("<Q", raw, 14)   # noqa: F841
    notes_len, = struct.unpack_from("<I", raw, 22)
    if notes_len != 600:
        raise ValueError(
            f"Unexpected NotesLength {notes_len} (expected 600) — "
            "possibly unsupported format"
        )
    entries_start, = struct.unpack_from("<q", raw, 26)

    if major == 0:
        raise ValueError(
            "Legacy AxIS v0 file format is not supported by this parser."
        )
    if major != 1:
        raise ValueError(f"Unsupported file format version {major}.{minor}")

    # Read 123 uint64 entry slots starting at byte 34.
    slot_offset = 34  # 8+2+2+2+8+4+8 = 34
    primary_records: List[Tuple[int, Optional[int]]] = []
    for i in range(_PRIMARY_MAX):
        raw_u64, = struct.unpack_from("<Q", raw, slot_offset + i * 8)
        primary_records.append(_parse_entry_uint64(raw_u64))

    # ── 3. Walk entry content sequentially from entries_start ─────────────────
    channel_lookup: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
    bv_header: Optional[Dict] = None
    bv_data_start: Optional[int] = None
    bv_data_length: Optional[int] = None

    # Collect ALL entry records across primary + sub-headers before walking.
    header_sets: List[List[Tuple[int, Optional[int]]]] = [primary_records]

    pos: int = int(entries_start)
    hset_idx: int = 0
    while hset_idx < len(header_sets):
        cur_records = header_sets[hset_idx]
        hset_idx += 1

        for type_id, length in cur_records:

            if type_id == _ID_TERMINATE:
                # Check for a sub-header beginning at the current position.
                if pos + len(_MAGIC) <= size and raw[pos : pos + 8] == _MAGIC:
                    sub_records: List[Tuple[int, Optional[int]]] = []
                    sub_pos = pos + 8  # skip magic
                    for i in range(_SUB_MAX):
                        if sub_pos + 8 > size:
                            break
                        raw_u64, = struct.unpack_from("<Q", raw, sub_pos)
                        sub_pos += 8
                        sub_records.append(_parse_entry_uint64(raw_u64))
                    # Skip CRC (4 bytes) + 4 reserved bytes.
                    pos = sub_pos + 8
                    header_sets.append(sub_records)
                break  # done with this header set

            if type_id in (_ID_NOTES, _ID_SKIP, _ID_TAG):
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
                bv_data_start = pos
                bv_data_length = length
                if length is not None:
                    pos += length
                continue

            # Unknown entry type — some newer file versions use IDs not in
            # the v3.1.1.6 enum.  Try to interpret content as BVHeader.
            if bv_header is None and length is not None and pos + 64 <= size:
                hdr = _read_bv_header(raw, pos)
                if hdr is not None:
                    bv_header = hdr
            if length is not None:
                pos += length

    if bv_data_start is None:
        raise ValueError("Could not find BlockVectorData in .spk file.")

    # ── 4. Determine spike record size ────────────────────────────────────────
    if bv_header is not None:
        fs: float = bv_header["fs"]
        hdr_sz: int = int(bv_header["hdr_sz"])
        n_samp: int = int(bv_header["n_samp"])
        n_ch_blk: int = int(bv_header["n_ch"])
        rec_sz: int = hdr_sz + n_ch_blk * n_samp * 2
        if fs_override is not None:
            log.info(
                "[spk] fs overridden: %.1f Hz (file reported %.1f Hz)  "
                "BlockHeaderSize=%d  SamplesPerBlock=%d  RecordSize=%d",
                fs_override, fs, hdr_sz, n_samp, rec_sz,
            )
            fs = fs_override
        else:
            log.info(
                "[spk] fs=%.1f Hz  BlockHeaderSize=%d  "
                "SamplesPerBlock=%d  RecordSize=%d",
                fs, hdr_sz, n_samp, rec_sz,
            )
    else:
        # Fallback: find a record size that divides the data length evenly.
        avail_fb = (
            bv_data_length if bv_data_length is not None
            else size - bv_data_start
        )
        assumed_fs = fs_override if fs_override is not None else 12500.0
        rec_sz = 0
        for sz in [68, 82, 96, 106, 110, 114, 126, 130, 134, 158, 162]:
            if avail_fb % sz == 0:
                rec_sz = sz
                break
        if rec_sz == 0:
            raise ValueError(
                "Could not determine spike record size — "
                "BlockVectorHeader missing and no candidate size divides "
                "the data length evenly."
            )
        hdr_sz = 30  # Spike_v1.LOADED_HEADER_SIZE
        n_samp = (rec_sz - hdr_sz) // 2
        fs = assumed_fs
        log.warning(
            "[spk] WARNING: BlockVectorHeader not found in '%s'. "
            "Spike timestamps may be inaccurate. "
            "Assumed fs=%.1f Hz (RecordSize=%d bytes). "
            "If this is wrong, re-run with fs_override=<correct_hz>.",
            spk_path.name, fs, rec_sz,
        )

    # ── 5. Parse spike records ────────────────────────────────────────────────
    avail: int = (
        bv_data_length if bv_data_length is not None
        else size - bv_data_start
    )
    n_spikes: int = avail // rec_sz
    log.info("[spk] %d spike records at file offset %d", n_spikes, bv_data_start)

    if n_spikes == 0:
        raise ValueError("No spike records found in BlockVectorData.")

    spike_ts = np.empty(n_spikes, dtype=np.float64)
    spike_ach = np.empty(n_spikes, dtype=np.uint8)
    spike_chi = np.empty(n_spikes, dtype=np.uint8)

    view = memoryview(raw)
    for i in range(n_spikes):
        off = bv_data_start + i * rec_sz
        start_samp, = struct.unpack_from("<q", view, off)       # int64
        chi_val = raw[off + 8]                                  # uint8 ChannelIndex
        ach_val = raw[off + 9]                                  # uint8 ChannelAchk
        trig_off, = struct.unpack_from("<I", view, off + 10)    # uint32 TriggerOffset
        spike_ts[i] = (start_samp + trig_off) / fs
        spike_chi[i] = chi_val
        spike_ach[i] = ach_val

    total_time_s: float = float(spike_ts.max()) if n_spikes > 0 else 0.0

    # ── 6. Map channels → electrode IDs ──────────────────────────────────────
    if wells is not None:
        target_wells: Optional[set] = {w.upper() for w in wells}
    else:
        target_wells = None  # accept all

    # Accumulate spike lists keyed by electrode_id string.
    buckets: Dict[str, List[float]] = {}

    if channel_lookup:
        # Use the ChannelArray lookup from the file.
        for i in range(n_spikes):
            key = (int(spike_ach[i]), int(spike_chi[i]))
            info = channel_lookup.get(key)
            if info is None:
                continue
            wr, wc, er, ec = info
            wlabel = well_label(wr, wc)
            if target_wells is not None and wlabel not in target_wells:
                continue
            eid = electrode_id(wr, wc, er, ec)
            if eid not in buckets:
                buckets[eid] = []
            buckets[eid].append(spike_ts[i])

    else:
        # No ChannelArray — fall back to sequential global-channel ordering
        # (same convention as the raw-file channel layout: ch 1-16 → well A1).
        log.warning(
            "[spk] ChannelArray not found; using sequential channel ordering."
        )
        all_wells = build_well_list()
        wells_to_process = (
            [w.upper() for w in wells] if wells is not None
            else all_wells
        )
        for w in wells_to_process:
            wi = all_wells.index(w)
            for en in range(1, CHANNELS_PER_WELL + 1):
                global_ch = wi * CHANNELS_PER_WELL + en
                er_fb = (en - 1) // ELECTRODES_PER_ROW + 1
                ec_fb = (en - 1) % ELECTRODES_PER_COL + 1
                wr_fb = all_wells.index(w) // PLATE_COLS + 1
                wc_fb = all_wells.index(w) % PLATE_COLS + 1
                eid = electrode_id(wr_fb, wc_fb, er_fb, ec_fb)
                mask = spike_chi.astype(np.int32) == (global_ch - 1) % 256
                buckets[eid] = list(spike_ts[mask])

    # Convert lists → sorted NumPy arrays.
    result: Dict[str, np.ndarray] = {
        eid: np.sort(np.array(ts, dtype=np.float64))
        for eid, ts in buckets.items()
    }

    # Ensure every electrode for requested wells is present (even if silent).
    if channel_lookup:
        # Derive the complete set of electrode IDs from the lookup table.
        wells_seen: set = set()
        for wr, wc, er, ec in channel_lookup.values():
            wlabel = well_label(wr, wc)
            if target_wells is None or wlabel in target_wells:
                wells_seen.add(wlabel)
        for wr, wc, er, ec in channel_lookup.values():
            wlabel = well_label(wr, wc)
            if wlabel not in wells_seen:
                continue
            eid = electrode_id(wr, wc, er, ec)
            result.setdefault(eid, np.array([], dtype=np.float64))
    else:
        all_wells = build_well_list()
        wells_to_fill = (
            [w.upper() for w in wells] if wells is not None
            else all_wells
        )
        for w in wells_to_fill:
            wi = all_wells.index(w)
            wr_fb = wi // PLATE_COLS + 1
            wc_fb = wi % PLATE_COLS + 1
            for en in range(1, CHANNELS_PER_WELL + 1):
                er_fb = (en - 1) // ELECTRODES_PER_ROW + 1
                ec_fb = (en - 1) % ELECTRODES_PER_COL + 1
                eid = electrode_id(wr_fb, wc_fb, er_fb, ec_fb)
                result.setdefault(eid, np.array([], dtype=np.float64))

    return result, total_time_s
