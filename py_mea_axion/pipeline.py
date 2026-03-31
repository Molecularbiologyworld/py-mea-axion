"""
pipeline.py
===========
High-level facade that orchestrates the full MEA analysis pipeline.

``MEAExperiment`` chains every processing step — spike loading, metric
computation, burst detection, network-burst detection, STTC calculation
— into a single object.  Results are exposed as tidy DataFrames and can
be fed directly into the stats and viz modules.

Typical usage
-------------
>>> from py_mea_axion.pipeline import MEAExperiment
>>> exp = MEAExperiment("recording.spk", metadata="plate_map.csv")
>>> exp.run()
>>> exp.spike_metrics.head()
>>> fig = exp.plot_trajectory("mean_mfr_active_hz")

Public API
----------
MEAExperiment(spk_path, metadata, wells, ...)
    Main entry point for file-based loading.

MEAExperiment.from_spikes(spikes_flat, total_time_s, metadata, ...)
    Alternative constructor for pre-loaded spike data (used in tests and
    notebooks where the .spk file is not available).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from py_mea_axion.burst.detection import Burst, detect_bursts
from py_mea_axion.burst.metrics import aggregate_well_bursts
from py_mea_axion.network.detection import (
    NetworkBurst,
    detect_network_bursts,
    detect_network_bursts_combined_isi,
)
from py_mea_axion.network.synchrony import mean_sttc, sttc_matrix
from py_mea_axion.spike.metrics import summarise_well
from py_mea_axion.stats.compare import CompareResult, compare_conditions, longitudinal_model
from py_mea_axion.viz.burst_charts import plot_burst_raster, plot_isi_histogram
from py_mea_axion.viz.heatmap import plot_electrode_heatmap
from py_mea_axion.viz.network_plots import plot_network_burst_timeline, plot_sttc_matrix
from py_mea_axion.viz.trajectory import plot_metric_trajectory

log = logging.getLogger(__name__)


# ── MEAExperiment ─────────────────────────────────────────────────────────────

class MEAExperiment:
    """End-to-end MEA analysis pipeline for a single Axion .spk recording.

    Parameters
    ----------
    spk_path : str or Path
        Path to the Axion ``*.spk`` binary file.
    metadata : str, Path, or pd.DataFrame, optional
        Plate-map data.  Passed directly to
        :func:`~py_mea_axion.io.metadata.load_metadata`.  When ``None``
        the condition/DIV columns are absent from summary tables and
        statistical methods are unavailable.
    wells : list of str, optional
        Subset of well IDs to load (e.g. ``['A1', 'B2']``).  Defaults to
        all wells present in the file.
    fs_override : float, optional
        Force a specific sampling frequency (Hz).  Passed to
        :func:`~py_mea_axion.io.spk_reader.load_spikes_from_spk`.
    active_threshold_hz : float, optional
        MFR threshold for classifying an electrode as active.
        Default 0.1 Hz.
    min_active_electrodes : int, optional
        Minimum number of active electrodes a well must have to be
        included in the analysis.  Wells that fall below this threshold
        are silently excluded and listed in :attr:`excluded_wells`.
        Set to ``0`` to disable filtering.  Default 1.
    burst_kwargs : dict, optional
        Keyword arguments forwarded to
        :func:`~py_mea_axion.burst.detection.detect_bursts` for every
        electrode.  E.g. ``{'max_isi_s': 0.1, 'min_spikes': 5}``.
    network_kwargs : dict, optional
        Keyword arguments forwarded to
        :func:`~py_mea_axion.network.detection.detect_network_bursts`
        for every well.
    sttc_dt_s : float, optional
        Coincidence window half-width (seconds) for STTC computation.
        Default 0.05 s (50 ms).

    Attributes
    ----------
    spike_metrics : pd.DataFrame
        Per-electrode metrics (MFR, ISI stats, active flag).  Populated
        after :meth:`run`.
    burst_table : pd.DataFrame
        All detected bursts, one row per burst.  Populated after
        :meth:`run`.
    well_summary : pd.DataFrame
        Per-well aggregate metrics (active electrode count, mean MFR,
        burst rate, STTC, …).  Populated after :meth:`run`.
    excluded_wells : list of str
        Wells that were removed because they had fewer active electrodes
        than *min_active_electrodes*.  Populated after :meth:`run`.

    Examples
    --------
    >>> exp = MEAExperiment.from_spikes(
    ...     {"A1_11": np.arange(0, 10, 0.1),
    ...      "A1_12": np.arange(0, 10, 0.2)},
    ...     total_time_s=10.0,
    ... )
    >>> exp.run()
    MEAExperiment(wells=['A1'], ran=True)
    >>> exp.spike_metrics[["electrode_id", "mfr_hz"]].values.tolist()  # doctest: +SKIP
    [['A1_11', 10.0], ['A1_12', 5.0]]
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        spk_path: Union[str, Path],
        metadata: Optional[Union[str, Path, pd.DataFrame]] = None,
        wells: Optional[List[str]] = None,
        fs_override: Optional[float] = None,
        active_threshold_hz: float = 0.1,
        min_active_electrodes: int = 1,
        burst_kwargs: Optional[Dict[str, Any]] = None,
        network_kwargs: Optional[Dict[str, Any]] = None,
        sttc_dt_s: float = 0.05,
    ) -> None:
        self.spk_path = Path(spk_path)
        self._metadata_source = metadata
        self._wells_filter = wells
        self._fs_override = fs_override
        self.active_threshold_hz = active_threshold_hz
        self._min_active_electrodes = min_active_electrodes
        self._burst_kwargs: Dict[str, Any] = burst_kwargs or {}
        self._network_kwargs: Dict[str, Any] = network_kwargs or {}
        self.sttc_dt_s = sttc_dt_s

        self._preloaded_spikes_flat: Optional[Dict[str, np.ndarray]] = None
        self._init_results()

    @classmethod
    def from_spikes(
        cls,
        spikes_flat: Dict[str, np.ndarray],
        total_time_s: float,
        metadata: Optional[Union[str, Path, pd.DataFrame]] = None,
        active_threshold_hz: float = 0.1,
        min_active_electrodes: int = 1,
        burst_kwargs: Optional[Dict[str, Any]] = None,
        network_kwargs: Optional[Dict[str, Any]] = None,
        sttc_dt_s: float = 0.05,
    ) -> "MEAExperiment":
        """Create an experiment from a pre-loaded flat spike dictionary.

        Bypasses file I/O so that the analysis pipeline can be tested or
        driven from notebook-computed spike times without a .spk file.

        Parameters
        ----------
        spikes_flat : dict[str, np.ndarray]
            Flat electrode-ID → spike-times mapping (e.g. as returned by
            :func:`~py_mea_axion.io.spk_reader.load_spikes_from_spk`).
        total_time_s : float
            Recording duration in seconds.
        metadata : optional
            Same as the *metadata* parameter of :class:`MEAExperiment`.

        Returns
        -------
        MEAExperiment
            Un-run experiment object.  Call :meth:`run` to execute the
            pipeline.

        Examples
        --------
        >>> exp = MEAExperiment.from_spikes({"A1_11": np.array([1.0, 2.0])},
        ...                                  total_time_s=10.0)
        >>> exp.run()
        MEAExperiment(wells=['A1'], ran=True)
        """
        obj = cls.__new__(cls)
        obj.spk_path = None  # type: ignore[assignment]
        obj._metadata_source = metadata
        obj._wells_filter = None
        obj._fs_override = None
        obj.active_threshold_hz = active_threshold_hz
        obj._min_active_electrodes = min_active_electrodes
        obj._burst_kwargs = burst_kwargs or {}
        obj._network_kwargs = network_kwargs or {}
        obj.sttc_dt_s = sttc_dt_s
        obj._preloaded_spikes_flat = spikes_flat
        obj._init_results()
        obj._total_time_s = total_time_s   # set after _init_results to avoid reset
        return obj

    def _init_results(self) -> None:
        """Reset all result containers."""
        self._total_time_s: float = 0.0
        self._spikes: Dict[str, Dict[str, np.ndarray]] = {}
        self._spike_metrics: Optional[pd.DataFrame] = None
        self._burst_table: Optional[pd.DataFrame] = None
        self._well_bursts: Dict[str, Dict[str, List[Burst]]] = {}
        self._network_bursts_dict: Dict[str, List[NetworkBurst]] = {}
        self._sttc_matrices: Dict[str, pd.DataFrame] = {}
        self._well_summary: Optional[pd.DataFrame] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._excluded_wells: List[str] = []
        self._ran = False

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def run(self) -> "MEAExperiment":
        """Execute the full analysis pipeline.

        Steps (in order):

        1. Load spike times from .spk file (skipped when constructed via
           :meth:`from_spikes`).
        2. Group electrodes into wells.
        3. Load and validate plate-map metadata.
        4. Compute per-electrode spike metrics.
        5. Detect single-electrode bursts.
        6. Detect network bursts per well.
        7. Compute STTC matrices per well.
        8. Assemble the per-well summary table.

        Returns
        -------
        MEAExperiment
            *self*, so calls can be chained:
            ``exp = MEAExperiment(...).run()``.
        """
        self._step_load_spikes()
        self._step_load_metadata()
        self._step_spike_metrics()
        self._step_filter_viable_wells()
        self._step_burst_detection()
        self._step_network_bursts()
        self._step_sttc()
        self._step_well_summary()
        self._ran = True
        log.info(
            "MEAExperiment: pipeline complete — %d wells, %.1f s",
            len(self._spikes), self._total_time_s,
        )
        return self

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def _step_load_spikes(self) -> None:
        if self._preloaded_spikes_flat is not None:
            flat = self._preloaded_spikes_flat
        else:
            from py_mea_axion.io.spk_reader import load_spikes_from_spk
            flat, self._total_time_s = load_spikes_from_spk(
                self.spk_path,
                wells=self._wells_filter,
                fs_override=self._fs_override,
            )

        self._spikes = _group_by_well(flat)
        if self._wells_filter:
            self._spikes = {
                w: v for w, v in self._spikes.items()
                if w in self._wells_filter
            }
        log.info("Loaded %d wells.", len(self._spikes))

    def _step_load_metadata(self) -> None:
        if self._metadata_source is None:
            return
        if isinstance(self._metadata_source, pd.DataFrame):
            self._metadata = self._metadata_source.copy()
        else:
            from py_mea_axion.io.metadata import load_metadata
            self._metadata = load_metadata(self._metadata_source)
        log.info("Metadata loaded: %d rows.", len(self._metadata))

    def _step_spike_metrics(self) -> None:
        frames = []
        for well_id, well_spk in self._spikes.items():
            df = summarise_well(
                well_spk, self._total_time_s, well_id,
                threshold_hz=self.active_threshold_hz,
            )
            frames.append(df)

        if frames:
            self._spike_metrics = pd.concat(frames, ignore_index=True)
        else:
            self._spike_metrics = pd.DataFrame(columns=[
                "well_id", "electrode_id", "n_spikes", "mfr_hz",
                "mean_isi", "median_isi", "cv_isi", "is_active",
            ])

    def _step_filter_viable_wells(self) -> None:
        """Drop wells with fewer active electrodes than *min_active_electrodes*.

        This runs after spike metrics so that the same plate-map (e.g. all
        24 wells) can be supplied to both Plate 1 (full plate) and Plate 2
        (12 populated wells) recordings.  Empty wells are automatically
        excluded; their IDs are stored in :attr:`excluded_wells`.
        """
        if self._min_active_electrodes <= 0:
            return

        sm = self._spike_metrics
        if sm is None or sm.empty:
            return

        active_counts = (
            sm[sm["is_active"]].groupby("well_id").size()
        )
        to_drop = [
            w for w in list(self._spikes.keys())
            if active_counts.get(w, 0) < self._min_active_electrodes
        ]
        if not to_drop:
            return

        for w in to_drop:
            del self._spikes[w]
        self._spike_metrics = sm[
            ~sm["well_id"].isin(to_drop)
        ].reset_index(drop=True)
        self._excluded_wells = sorted(to_drop)
        log.info(
            "Excluded %d well(s) with < %d active electrode(s): %s",
            len(to_drop), self._min_active_electrodes, self._excluded_wells,
        )

    def _step_burst_detection(self) -> None:
        all_burst_dfs = []
        for well_id, well_spk in self._spikes.items():
            well_bd: Dict[str, List[Burst]] = {}
            for eid, ts in well_spk.items():
                well_bd[eid] = detect_bursts(ts, **self._burst_kwargs)
            self._well_bursts[well_id] = well_bd

            df = aggregate_well_bursts(well_bd, well_id)
            if not df.empty:
                all_burst_dfs.append(df)

        if all_burst_dfs:
            self._burst_table = pd.concat(all_burst_dfs, ignore_index=True)
        else:
            from py_mea_axion.burst.metrics import BURST_COLUMNS
            self._burst_table = pd.DataFrame(columns=BURST_COLUMNS)

    def _step_network_bursts(self) -> None:
        kwargs = dict(self._network_kwargs)
        algorithm = kwargs.pop("algorithm", "combined_isi")

        for well_id, well_bd in self._well_bursts.items():
            if algorithm == "combined_isi":
                self._network_bursts_dict[well_id] = detect_network_bursts_combined_isi(
                    self._spikes[well_id],
                    total_time_s=self._total_time_s,
                    **kwargs,
                )
            else:
                self._network_bursts_dict[well_id] = detect_network_bursts(
                    well_bd,
                    total_time_s=self._total_time_s,
                    **kwargs,
                )

    def _step_sttc(self) -> None:
        for well_id, well_spk in self._spikes.items():
            self._sttc_matrices[well_id] = sttc_matrix(
                well_spk,
                dt_s=self.sttc_dt_s,
                total_time_s=self._total_time_s,
            )

    def _step_well_summary(self) -> None:
        """Build a per-well aggregate metrics DataFrame."""
        rows = []
        for well_id in sorted(self._spikes.keys()):
            well_spk = self._spikes[well_id]
            well_sm = self._spike_metrics[self._spike_metrics["well_id"] == well_id]

            active_mask = well_sm["is_active"]
            n_active = int(active_mask.sum())

            mean_mfr = float(
                well_sm.loc[active_mask, "mfr_hz"].mean()
            ) if n_active else float("nan")

            mean_cv = float(
                well_sm.loc[active_mask, "cv_isi"].mean()
            ) if n_active else float("nan")

            # Burst rate: bursts per electrode per second (active only).
            bt = self._burst_table
            well_bt = bt[bt["well_id"] == well_id] if not bt.empty else pd.DataFrame()
            n_bursts = len(well_bt)
            burst_rate = (
                n_bursts / (n_active * self._total_time_s)
                if n_active and self._total_time_s > 0
                else float("nan")
            )
            mean_burst_dur = (
                float(well_bt["duration"].mean()) if n_bursts else float("nan")
            )

            n_network = len(self._network_bursts_dict.get(well_id, []))

            msttc = mean_sttc(
                well_spk,
                dt_s=self.sttc_dt_s,
                total_time_s=self._total_time_s,
                active_only=True,
                active_threshold_hz=self.active_threshold_hz,
            )

            rows.append({
                "well_id": well_id,
                "n_electrodes": len(well_spk),
                "n_active": n_active,
                "mean_mfr_active_hz": mean_mfr,
                "mean_cv_isi": mean_cv,
                "burst_rate_hz": burst_rate,
                "mean_burst_duration_s": mean_burst_dur,
                "n_network_bursts": n_network,
                "mean_sttc": msttc,
            })

        self._well_summary = pd.DataFrame(rows)

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def wells(self) -> List[str]:
        """Sorted list of well IDs present in the recording."""
        return sorted(self._spikes.keys())

    @property
    def total_time_s(self) -> float:
        """Recording duration in seconds."""
        return self._total_time_s

    @property
    def spike_metrics(self) -> pd.DataFrame:
        """Per-electrode spike metrics.  Requires :meth:`run`."""
        self._require_ran()
        return self._spike_metrics  # type: ignore[return-value]

    @property
    def burst_table(self) -> pd.DataFrame:
        """All detected single-electrode bursts.  Requires :meth:`run`."""
        self._require_ran()
        return self._burst_table  # type: ignore[return-value]

    @property
    def well_summary(self) -> pd.DataFrame:
        """Per-well aggregate metrics.  Requires :meth:`run`.

        Columns
        -------
        well_id, n_electrodes, n_active, mean_mfr_active_hz,
        mean_cv_isi, burst_rate_hz, mean_burst_duration_s,
        n_network_bursts, mean_sttc.
        """
        self._require_ran()
        return self._well_summary  # type: ignore[return-value]

    @property
    def network_bursts(self) -> Dict[str, List[NetworkBurst]]:
        """Dict mapping well_id → list of NetworkBurst.  Requires :meth:`run`."""
        self._require_ran()
        return self._network_bursts_dict

    @property
    def sttc_matrices(self) -> Dict[str, pd.DataFrame]:
        """Dict mapping well_id → STTC matrix DataFrame.  Requires :meth:`run`."""
        self._require_ran()
        return self._sttc_matrices

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """Plate-map metadata, or ``None`` if not provided."""
        return self._metadata

    @property
    def excluded_wells(self) -> List[str]:
        """Wells removed because they had fewer active electrodes than
        *min_active_electrodes*.  Empty list when no wells were excluded."""
        return list(self._excluded_wells)

    # ── Data access helpers ───────────────────────────────────────────────────

    def well_spikes(self, well_id: str) -> Dict[str, np.ndarray]:
        """Return the spike-time dictionary for one well.

        Parameters
        ----------
        well_id : str
            Well label, e.g. ``'A1'``.

        Returns
        -------
        dict[str, np.ndarray]
            Electrode ID → sorted spike times (s).

        Raises
        ------
        KeyError
            If *well_id* is not present in the recording.
        RuntimeError
            If called before :meth:`run`.
        """
        self._require_ran()
        if well_id not in self._spikes:
            raise KeyError(
                f"Well '{well_id}' not found. Available: {self.wells}"
            )
        return self._spikes[well_id]

    def well_burst_dict(self, well_id: str) -> Dict[str, List[Burst]]:
        """Return the burst dictionary for one well.

        Parameters
        ----------
        well_id : str
            Well label.

        Returns
        -------
        dict[str, list[Burst]]
            Electrode ID → list of Burst namedtuples.
        """
        self._require_ran()
        return self._well_bursts.get(well_id, {})

    def joined_summary(self) -> pd.DataFrame:
        """Return :attr:`well_summary` joined with :attr:`metadata`.

        The join is on ``well_id``.  If no metadata was loaded the raw
        :attr:`well_summary` is returned unchanged.

        Returns
        -------
        pd.DataFrame
            Combined table.  Extra metadata columns (``condition``,
            ``DIV``, ``replicate_id``, …) are appended when available.
        """
        self._require_ran()
        ws = self._well_summary.copy()  # type: ignore[union-attr]
        if self._metadata is not None:
            ws = ws.merge(self._metadata, on="well_id", how="left")
        return ws

    def to_csv(
        self,
        path: Union[str, Path],
        float_format: str = "%.6f",
    ) -> Path:
        """Save :meth:`joined_summary` to a CSV file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        float_format : str, optional
            printf-style format string for floating-point columns.
            Default ``'%.6f'``.

        Returns
        -------
        Path
            The resolved output path.
        """
        self._require_ran()
        p = Path(path)
        self.joined_summary().to_csv(p, index=False, float_format=float_format)
        log.info("Results saved to %s", p)
        return p

    # ── Statistics ────────────────────────────────────────────────────────────

    def compare(
        self,
        metric: str,
        group_col: str = "condition",
    ) -> CompareResult:
        """Compare a well-level metric across experimental conditions.

        Wraps :func:`~py_mea_axion.stats.compare.compare_conditions`
        using :meth:`joined_summary` as the data source.

        Parameters
        ----------
        metric : str
            Column in :meth:`joined_summary` to compare.
        group_col : str, optional
            Column identifying conditions.  Default ``'condition'``.

        Returns
        -------
        CompareResult
            Named tuple with ``test``, ``statistic``, ``p_value``,
            ``effect_size``, and ``posthoc`` fields.

        Raises
        ------
        ValueError
            If metadata was not provided (no group column) or if the
            metric column is absent.
        RuntimeError
            If called before :meth:`run`.

        Examples
        --------
        >>> res = exp.compare("mean_mfr_active_hz")
        >>> res.p_value < 0.05
        True
        """
        self._require_ran()
        df = self.joined_summary()
        if group_col not in df.columns:
            raise ValueError(
                f"Group column '{group_col}' not found. "
                "Did you provide metadata with condition labels?"
            )
        return compare_conditions(df, metric=metric, group_col=group_col)

    def longitudinal(
        self,
        metric: str,
        time_col: str = "DIV",
        group_col: str = "condition",
        subject_col: str = "replicate_id",
    ) -> pd.DataFrame:
        """Fit a longitudinal mixed-effects model.

        Wraps :func:`~py_mea_axion.stats.compare.longitudinal_model`
        using :meth:`joined_summary` as the data source.

        Parameters
        ----------
        metric : str
            Dependent variable column.
        time_col : str, optional
            Time covariate.  Default ``'DIV'``.
        group_col : str, optional
            Group/condition column.  Default ``'condition'``.
        subject_col : str, optional
            Random-intercept grouping column.  Default ``'replicate_id'``.

        Returns
        -------
        pd.DataFrame
            Coefficient table from the mixed-effects model.

        Raises
        ------
        ValueError
            If required columns are missing or the model fails.
        RuntimeError
            If called before :meth:`run`.
        """
        self._require_ran()
        df = self.joined_summary()
        return longitudinal_model(
            df, metric=metric, time_col=time_col,
            group_col=group_col, subject_col=subject_col,
        )

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_heatmap(
        self,
        well_id: str,
        metric: str = "mfr_hz",
        **kwargs: Any,
    ) -> Figure:
        """Plot the electrode heatmap for *well_id* coloured by *metric*.

        Parameters
        ----------
        well_id : str
            Well to visualise.
        metric : str, optional
            Column in :attr:`spike_metrics` to colour by.
            Default ``'mfr_hz'``.
        **kwargs
            Forwarded to
            :func:`~py_mea_axion.viz.heatmap.plot_electrode_heatmap`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_ran()
        sm = self._spike_metrics  # type: ignore[union-attr]
        well_rows = sm[sm["well_id"] == well_id]
        values = dict(zip(well_rows["electrode_id"], well_rows[metric]))
        return plot_electrode_heatmap(
            values, well_id=well_id, metric_name=metric, **kwargs
        )

    def plot_raster(
        self,
        well_id: str,
        **kwargs: Any,
    ) -> Figure:
        """Plot the burst raster for *well_id*.

        Parameters
        ----------
        well_id : str
            Well to visualise.
        **kwargs
            Forwarded to
            :func:`~py_mea_axion.viz.burst_charts.plot_burst_raster`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_ran()
        return plot_burst_raster(
            self.well_spikes(well_id),
            self.well_burst_dict(well_id),
            **kwargs,
        )

    def plot_isi(
        self,
        electrode_id: str,
        **kwargs: Any,
    ) -> Figure:
        """Plot the ISI histogram for one electrode.

        Parameters
        ----------
        electrode_id : str
            Electrode label (e.g. ``'A1_11'``).
        **kwargs
            Forwarded to
            :func:`~py_mea_axion.viz.burst_charts.plot_isi_histogram`.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        KeyError
            If the electrode is not present in the recording.
        """
        self._require_ran()
        well_id = electrode_id.split("_")[0]
        ws = self.well_spikes(well_id)
        if electrode_id not in ws:
            raise KeyError(
                f"Electrode '{electrode_id}' not found in well '{well_id}'."
            )
        return plot_isi_histogram(
            ws[electrode_id], electrode_id=electrode_id, **kwargs
        )

    def plot_trajectory(
        self,
        metric: str,
        **kwargs: Any,
    ) -> Figure:
        """Plot the longitudinal trajectory of a well-level metric.

        Requires metadata with ``condition``, ``DIV``, and
        ``replicate_id`` columns.

        Parameters
        ----------
        metric : str
            Column in :meth:`joined_summary` to plot.
        **kwargs
            Forwarded to
            :func:`~py_mea_axion.viz.trajectory.plot_metric_trajectory`.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If metadata was not loaded (no time/group columns available).
        """
        self._require_ran()
        df = self.joined_summary()
        return plot_metric_trajectory(df, metric=metric, **kwargs)

    def plot_sttc(
        self,
        well_id: str,
        **kwargs: Any,
    ) -> Figure:
        """Plot the STTC matrix for *well_id*.

        Parameters
        ----------
        well_id : str
            Well to visualise.
        **kwargs
            Forwarded to
            :func:`~py_mea_axion.viz.network_plots.plot_sttc_matrix`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_ran()
        kwargs.setdefault("title", f"STTC — {well_id}")
        return plot_sttc_matrix(
            self._sttc_matrices.get(well_id, pd.DataFrame()),
            **kwargs,
        )

    def plot_network_timeline(
        self,
        well_id: str,
        **kwargs: Any,
    ) -> Figure:
        """Plot the network-burst timeline for *well_id*.

        Parameters
        ----------
        well_id : str
            Well to visualise.
        **kwargs
            Forwarded to
            :func:`~py_mea_axion.viz.network_plots.plot_network_burst_timeline`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._require_ran()
        nbs = self._network_bursts_dict.get(well_id, [])
        kwargs.setdefault("title", f"Network bursts — {well_id}")
        return plot_network_burst_timeline(
            nbs, self._total_time_s,
            **kwargs,
        )

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MEAExperiment(wells={self.wells!r}, ran={self._ran})"
        )

    def _require_ran(self) -> None:
        if not self._ran:
            raise RuntimeError(
                "Analysis has not been run yet. Call .run() first."
            )


# ── Private helpers ────────────────────────────────────────────────────────────

def _group_by_well(
    flat: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Group a flat electrode dict by well ID.

    Well ID is taken as everything before the first ``'_'`` in the
    electrode label (e.g. ``'A1_23'`` → well ``'A1'``).

    Parameters
    ----------
    flat : dict[str, np.ndarray]
        Electrode ID → spike times mapping.

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Well ID → {electrode ID → spike times}.
    """
    grouped: Dict[str, Dict[str, np.ndarray]] = {}
    for eid, ts in flat.items():
        well_id = eid.split("_")[0]
        grouped.setdefault(well_id, {})[eid] = ts
    return grouped
