"""
export.py
---------
Export and import helpers for reciprocating friction test data.

Exports
~~~~~~~
* CSV  : human-readable test file with metadata header + time-series
         data (time, displacement, tangential force).
* shelve : binary key-value store for full session recovery.

Author : Iñigo Llavori Osa — Mondragon Unibertsitatea
"""

from __future__ import annotations

import csv
import shelve
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class Exporter:
    """
    Export simulation results to CSV and/or shelve files.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for all output files.  Created if it does not
        exist.
    filename : str
        Base name used for the sub-folder, CSV file and shelve file.
    """

    def __init__(self, output_dir: str | Path, filename: str) -> None:
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.test_dir = self.output_dir / filename
        self.test_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_csv(
        self,
        contact_force: float,
        stroke: float,
        frequency: float,
        sampling_rate: float,
        time: np.ndarray,
        displacement: np.ndarray,
        tangential_force: np.ndarray,
        final_wear_depth: float,
        configuration: str = "ball-on-flat",
        lubricant: str = "none",
        environment: str = "air",
        temperature: str = "RT",
        sensor_info: dict | None = None,
    ) -> Path:
        """
        Write a semicolon-delimited CSV with metadata + time-series data.

        Parameters
        ----------
        sensor_info : dict, optional
            If provided, append force-sensor metadata to the CSV header so
            the file is self-describing. Expected keys (all optional, written
            only if present):

                'Qfs_N'           : force sensor full-scale F_fs [N]
                'sigma_rel'       : instrumental noise fraction of F_fs [-]
                'preset_id'       : sensor library id (e.g. 'bruker-dfh-50g')
                'vendor'          : sensor vendor (e.g. 'Bruker')
                'official_name'   : full datasheet name
                'agnostic_label'  : vendor-agnostic label for paper / UI

        Returns
        -------
        Path
            Path to the written CSV file.
        """
        csv_path = self.test_dir / f"test{self.filename}.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=";")

            # ── Metadata ────────────────────────────────────────────
            writer.writerow(["sep=;"])
            writer.writerow(["filename", self.filename])
            writer.writerow(["test date", datetime.now().strftime("%d/%m/%Y %H:%M:%S")])
            writer.writerow(["configuration", configuration])
            writer.writerow(["movement", "linear reciprocating"])
            writer.writerow(["sampling frequency (Hz)", f"{sampling_rate:.2f}"])
            writer.writerow(["sample 1 geometry", "ball"])
            writer.writerow(["sample 2 geometry", "flat"])
            writer.writerow(["lubricant", lubricant])
            writer.writerow(["nominal load deadweight (N)", f"{contact_force:.3f}"])
            writer.writerow(["stroke length (mm)", f"{stroke:.3f}"])
            writer.writerow(["test frequency (Hz)", f"{frequency:.3f}"])
            writer.writerow(["nominal temperature (°C)", temperature])
            writer.writerow(["environment", environment])
            writer.writerow([
                "wear scar depth on sample 2 after test (µm)",
                f"{final_wear_depth * 1e3:.3f}",
            ])

            # ── Sensor metadata ──────────────────────────────────────
            # Written just before the empty separator line so the data
            # block byte-format remains identical. All fields are
            # optional; only those present in *sensor_info* are emitted.
            if sensor_info:
                if sensor_info.get("Qfs_N") is not None:
                    writer.writerow([
                        "force sensor full scale (N)",
                        f"{float(sensor_info['Qfs_N']):.3f}",
                    ])
                if sensor_info.get("sigma_rel") is not None:
                    writer.writerow([
                        "sensor noise fraction (-)",
                        f"{float(sensor_info['sigma_rel']):.6f}",
                    ])
                if sensor_info.get("preset_id"):
                    writer.writerow(["sensor preset id", sensor_info["preset_id"]])
                if sensor_info.get("vendor"):
                    writer.writerow(["sensor vendor", sensor_info["vendor"]])
                if sensor_info.get("official_name"):
                    writer.writerow(["sensor official name", sensor_info["official_name"]])
                if sensor_info.get("agnostic_label"):
                    writer.writerow(["sensor agnostic label", sensor_info["agnostic_label"]])

            writer.writerow([" "])

            # ── Column headers ───────────────────────────────────────
            writer.writerow(["time (ms)", "position (mm)", "horizontal force (N)"])

            # ── Data ────────────────────────────────────────────────
            for t, d, f in zip(time, displacement.ravel(), tangential_force.ravel()):
                writer.writerow([
                    round(float(t) * 1e3, 1),
                    round(float(d), 4),
                    round(float(f), 4),
                ])

        return csv_path

    # ------------------------------------------------------------------
    # Shelve (session save / load)
    # ------------------------------------------------------------------

    def save_session(self, data: dict[str, Any]) -> Path:
        """
        Persist all simulation variables to a shelve file.

        Parameters
        ----------
        data : dict
            Mapping of variable names to values.  Non-serialisable
            objects (modules, lambdas, etc.) are silently skipped.

        Returns
        -------
        Path
            Path to the shelve file (without extension).
        """
        recovery_dir = self.output_dir / f"{self.filename}_recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        shelf_path = recovery_dir / self.filename

        with shelve.open(str(shelf_path), flag="n") as shelf:
            for key, value in data.items():
                try:
                    shelf[key] = value
                except Exception:
                    pass  # skip un-serialisable objects

        return shelf_path


def load_session(shelf_path: str | Path) -> dict[str, Any]:
    """
    Load all variables from a shelve file previously saved by
    :meth:`Exporter.save_session`.

    Parameters
    ----------
    shelf_path : str | Path
        Path to the shelve file (without extension).

    Returns
    -------
    dict
        All key-value pairs stored in the shelf.
    """
    shelf_path = str(shelf_path)
    data: dict[str, Any] = {}
    with shelve.open(shelf_path, flag="r") as shelf:
        for key in shelf:
            data[key] = shelf[key]
    return data
