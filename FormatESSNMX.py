from __future__ import annotations

import logging

import h5py
import numpy as np
import re

import cctbx.array_family.flex as flex

import dxtbx_flumpy as flumpy
from dxtbx import IncorrectFormatError
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.model import Detector
from dxtbx.model.beam import BeamFactory, PolychromaticBeam, Probe
from dxtbx.model.goniometer import Goniometer, GoniometerFactory
from dxtbx.model.scan import Scan, ScanFactory
from dxtbx.model.tof_helpers import wavelength_from_tof

logger = logging.getLogger(__name__)

# Factor that converts a value expressed in <key> into microseconds.
_TO_US = {
    "s": 1e6, "sec": 1e6, "secs": 1e6, "second": 1e6, "seconds": 1e6,
    "ms": 1e3, "msec": 1e3, "millisecond": 1e3, "milliseconds": 1e3,
    "us": 1.0, "usec": 1.0, "microsecond": 1.0, "microseconds": 1.0,
    "µs": 1.0,   # U+00B5 MICRO SIGN
    "μs": 1.0,   # U+03BC GREEK SMALL LETTER MU
    "ns": 1e-3, "nsec": 1e-3, "nanosecond": 1e-3, "nanoseconds": 1e-3,
    "ps": 1e-6, "picosecond": 1e-6, "picoseconds": 1e-6,
}

# NeXus says "units"; real files in the wild also use these.
_UNIT_ATTRS = ("units", "unit", "Units", "Unit", "UNITS")

# Neutron TOF: t[us] = 252.7784 * lambda[A] * L[m]
TOF_CONSTANT = 252.7784


def _as_text(value) -> str:
    """HDF5 string attributes come back as bytes, str, or 0-d arrays of either."""
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", "replace")
    return str(value).strip()


def units_of(dataset, search_parent: bool = True) -> str | None:
    """Return the unit string attached to ``dataset``, or None if absent.

    Falls back to the containing group, since some writers put ``units`` on the
    NXdata/NXdetector group rather than on the field.
    """
    for attrs in ([dataset.attrs] + ([dataset.parent.attrs] if search_parent else [])):
        for key in _UNIT_ATTRS:
            if key in attrs:
                text = _as_text(attrs[key])
                if text:
                    return text
    return None


def time_factor_to_us(units: str) -> float:
    """Multiplicative factor taking a value in ``units`` to microseconds.

    Handles the Mantid/ISIS style scaled units too, e.g. ``100*ns``.
    """
    text = _as_text(units).replace(" ", "")
    scale = 1.0
    match = re.match(r"^([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\*?(.+)$", text)
    if match:
        scale, text = float(match.group(1)), match.group(2)
    key = text.lower().rstrip(".")
    try:
        return scale * _TO_US[key]
    except KeyError:
        raise ValueError(f"unrecognised time unit {units!r}") from None


def read_time_us(dataset, default_units: str | None = None) -> np.ndarray:
    """Read an HDF5 time dataset and return it in microseconds.

    ``default_units`` is used only when the file carries no units attribute at
    all; it is logged loudly, because a silent guess is exactly the failure mode
    this module exists to prevent.
    """
    values = np.asarray(dataset[()], dtype=float).ravel()
    units = units_of(dataset)
    if units is None:
        if default_units is None:
            raise ValueError(
                f"{dataset.name} has no units attribute and no default was given; "
                "refusing to guess the TOF scale"
            )
        units = default_units
        logger.warning(
            "%s has no units attribute; assuming %r. Fix the writer, or pass "
            "default_units explicitly.", dataset.name, units
        )
    return values * time_factor_to_us(units)


def expected_tof_range_us(distance_m: float, wavelength_range_a: tuple[float, float]):
    """Physically plausible TOF window for a given flight path and bandwidth."""
    lo, hi = sorted(wavelength_range_a)
    return (TOF_CONSTANT * lo * distance_m, TOF_CONSTANT * hi * distance_m)


def check_tof_plausible(tof_us, distance_m: float, wavelength_range_a, tol: float = 5.0):
    """Warn if the TOF array is off by a suspicious factor (usually 1000x).

    Returns the ratio of observed to expected magnitude; ~1 is healthy.
    """
    tof_us = np.asarray(tof_us, dtype=float)
    if tof_us.size == 0:
        return float("nan")
    lo, hi = expected_tof_range_us(distance_m, wavelength_range_a)
    observed = float(np.median(tof_us))
    ratio = observed / (0.5 * (lo + hi))
    if not (1.0 / tol) < ratio < tol:
        logger.warning(
            "TOF values look wrong by a factor of ~%.3g: median %.4g us, but a "
            "%.1f m flight path over %.2f-%.2f A implies %.4g-%.4g us. Check the "
            "units attribute on the TOF dataset.",
            ratio, observed, distance_m, wavelength_range_a[0],
            wavelength_range_a[1], lo, hi,
        )
    return ratio



class FormatESSNMX(FormatHDF5):
    """
    Class to read files from NMX
    https://europeanspallationsource.se/instruments/nmx
    preprocessed files in scipp to obtain binned data
    """

    def __init__(self, image_file, **kwargs) -> None:
        if not FormatESSNMX.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        self._nxs_file = h5py.File(image_file, "r")
        self._raw_data = None

    @staticmethod
    def understand(image_file: str) -> bool:
        try:
            return FormatESSNMX.is_nmx_file(image_file)
        except (OSError, KeyError):
            return False

    @staticmethod
    def is_nmx_file(image_file: str) -> bool:
        def get_name(image_file):
            try:
                with h5py.File(image_file, "r") as handle:
                    
                    return handle["entry/instrument/name"][...].item().decode()
            except (OSError, KeyError, AttributeError):
                return ""

        def is_McStas(image_file: str) -> bool:
            try:
                with h5py.File(image_file, "r") as handle:
                    if handle.get(
                        "entry/metadata/mcstas_weight2count_scale_factor", None
                    ):
                        return True
                    return False
            except (OSError, KeyError, AttributeError):
                return False

        return get_name(image_file) == "NMX" and not is_McStas(image_file)

    def get_instrument_name(self) -> str:
        return "NMX"

    def get_experiment_description(self) -> str:
        return "NMX Data"

    def _load_raw_data(self) -> None:
        raw_data = []
        for panel in self._get_panels():
            spectra = panel["data"][...]
            raw_data.append(flumpy.from_numpy(np.ascontiguousarray(spectra)))

        self._raw_data = tuple(raw_data)

    def get_raw_data(
        self, index: int, use_loaded_data: bool = False
    ) -> tuple[flex.int]:
        raw_data = []
        # image_size = self._get_image_size()
        # total_pixels = image_size[0] * image_size[1]

        if use_loaded_data:
            if self._raw_data is None:
                self._load_raw_data()
            for panel in self._raw_data:
                data = panel[:, :, index : index + 1].astype(np.int32)
                data.reshape(flex.grid(panel.all()[0], panel.all()[1]))
                data.matrix_transpose_in_place()
                raw_data.append(data)

        else:
            for panel in self._get_panels():
                spectra = panel["data"][:, :, index].astype(np.int32)
                raw_data.append(flumpy.from_numpy(np.ascontiguousarray(spectra)))

        return tuple(raw_data)

    def _get_time_channel_bins(self) -> list[float]:
        # (usec)
        # the tofs are recorded separately per panel but they
        # should all be the same
        for panel in self._get_panels():
            return panel["time_of_flight"][...] / 1e3

    def _get_time_of_flight(self) -> list[float]:
        # (usec)
        bins = self._get_time_channel_bins()
        return [float((bins[i] + bins[i + 1]) * 0.5) for i in range(len(bins) - 1)]

    def get_num_images(self) -> int:
        return len(self._get_time_of_flight())

    def get_detector(self, index: int = None) -> Detector:
        panel_names = self._get_panel_names()
        panel_type = self._get_panel_type()
        trusted_range = self._get_panel_trusted_range()
        pixel_size = self._get_pixel_size()
        gain = self._get_panel_gain()
        detector = Detector()
        root = detector.hierarchy()
        panels = self._get_panels()
        panel_projections = self._get_panel_projections_2d(panels)
        for panel_dset, panel_name in zip(panels, panel_names):
            panel = root.add_panel()
            panel.set_type(panel_type)
            panel.set_name(panel_name)
            panel.set_image_size(
                self._get_image_size(panel_dset)
            )  # XXX fix to include detectors possibly not being the same size
            panel.set_trusted_range(trusted_range)
            panel.set_pixel_size(pixel_size)
            fast_axis = self._get_panel_fast_axes(panel_dset)
            slow_axis = self._get_panel_slow_axes(panel_dset)
            panel_origin = self._get_panel_origins(panel_dset)
            panel.set_local_frame(fast_axis, slow_axis, panel_origin)

            panel.set_gain(gain)
            i = int(panel_name[-1])
            r, t = panel_projections[i]
            r = tuple(map(int, r))
            t = tuple(map(int, t))
            panel.set_projection_2d(r, t)

        return detector

    def _get_num_panels(self) -> int:
        return len(self._get_panels())

    def _get_panels(self) -> list[h5py._hl.group.Group]:
        """get the detector panel locations in file"""
        panels = []
        inst_dset = self._nxs_file["/entry/instrument/"]
        for _, dset in inst_dset.items():
            if dset.attrs.get("NX_class") == "NXdetector":
                panels.append(dset)
        return panels

    def _get_panel_names(self) -> list[str]:
        panel_names = []
        inst_dset = self._nxs_file["/entry/instrument/"]
        for name, dset in inst_dset.items():
            if dset.attrs.get("NX_class") == "NXdetector":
                panel_names.append(name)
        return panel_names

    def _get_panel_name(self, panel) -> str:
        return panel.name.split("/")[-1]

    def _get_panel_type(self) -> str:
        return "Triple_GEM_Gd"

    def _get_image_size(self, panel) -> tuple[int, int]:
        # (px)
        dset = panel["data"]
        return dset[:, :, 0].shape

    def _get_panel_trusted_range(self) -> tuple[int, int]:
        # 4 * 1280**2 plus buffer
        return (-1, 2**64 - 1)

    def _get_pixel_size(self) -> tuple[float, float]:
        # (mm)
        return (0.4, 0.4)

    def _get_panel_fast_axes(self, panel) -> tuple[float, float, float]:
        # return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
        fast_axis = panel["fast_axis"][...]
        return tuple(fast_axis)

    def _get_panel_slow_axes(self, panel) -> tuple[float, float, float]:
        # return ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        slow_axis = panel["slow_axis"][...]
        return tuple(slow_axis)

    def _get_panel_origins(self, panel) -> tuple[float, float, float]:
        # (mm)
        # return ((-250, -250.0, -292.0), (290, -250.0, -250), (-290, -250.0, 250.0))
        if panel["origin"].attrs.get("first_pixel_position", None) is not None:
            logger.debug("first_pixel_origin found in attrubiute, using...")
            first_pixel_position = panel["origin"].attrs.get("first_pixel_position")
            first_pixel_position *= 1000     # mm
            return tuple(first_pixel_position)
        else:
            origin = panel["origin"][...]
            panelnum = int(panel.name[-1])
            origin *= 1000  # convert to mm
            corrfact = self._get_correction_factor()[panelnum]

            corrorg = origin + corrfact
            return tuple(corrorg)

    def _get_correction_factor(self):
        return np.array([[0, -256, -256], [-256.0, -256.0, 0], [0, -256, 256]])

    def _get_panel_projections_2d(self, panels) -> dict[int : tuple[tuple, tuple]]:
        p_w, p_h = self._get_image_size(panels[0])  # XXX fix later
        p_w += 10
        p_h += 10
        panel_pos = {
            int(panels[0].name[-1]): ((-1, 0, 0, -1), (p_h, 0)),
            int(panels[1].name[-1]): ((-1, 0, 0, -1), (p_h, p_w)),
            int(panels[2].name[-1]): ((-1, 0, 0, -1), (p_h, -p_w)),
        }

        return panel_pos

    def get_beam(self, index: int = None) -> PolychromaticBeam:
        direction = self._get_sample_to_source_direction()
        distance = self._get_sample_to_source_distance()
        wavelength_range = self._get_wavelength_range()
        return BeamFactory.make_polychromatic_beam(
            direction=direction,
            sample_to_source_distance=distance,
            probe=Probe.neutron,
            wavelength_range=wavelength_range,
        )

    def _get_sample_to_source_direction(self) -> tuple[float, float, float]:
        return (0, 0, -1)

    # def _get_wavelength_range(self) -> tuple[float, float]:
    #     # (A)
    #     return (1.8, 2.55)

    def _get_wavelength_range(self) -> tuple[float, float]:
        # (A)
        tofs = np.array(self._get_time_of_flight()) / 1e6  # (s)
        tof_low = tofs[0]
        tof_high = tofs[-1]
        distance = self._get_sample_to_source_distance() / 1000  # (m)
        lambda_low = wavelength_from_tof(distance=distance, tof=tof_low)
        lambda_high = wavelength_from_tof(distance=distance, tof=tof_high)
        return (round(lambda_low, 2), round(lambda_high, 2))

    def _tof_to_lambda(self, tof):
        """given tof in s, return lambda in Angstrom"""
        neutron_mass = 1.67492749804e-27
        h = 6.62607015e-34
        distance = self._get_sample_to_source_distance() / 1000  # convert to m
        return h / (neutron_mass * (distance / tof)) * 1e10

    def _get_sample_to_source_distance(self) -> float:
        """get sample to source distance in mm"""
        try:
            dist = abs(self._nxs_file["entry/instrument/source/distance"][...]) * 1000
            return dist
        except (KeyError, ValueError):
            logger.warning("sample to moderator_distance not found, using dummy value")
            return 156714

    def _get_panel_gain(self) -> float:
        return 1.0

    def get_goniometer_phi_angle(self) -> float:
        return self.get_goniometer_orientations()[1]

    def get_goniometer(self, index: int = None) -> Goniometer:
        rotation_axis = (0.0, 1.0, 0.0)
        fixed_rotation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        goniometer = GoniometerFactory.make_goniometer(rotation_axis, fixed_rotation)
        try:
            angles = self.get_goniometer_orientations()
        except KeyError:
            logger.warning("crystal_rotation not found, using default")
            return goniometer
        axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for idx, angle in enumerate(angles):
            goniometer.rotate_around_origin(axes[idx], angle)
        return goniometer

    def get_goniometer_orientations(self) -> tuple[float, float, float]:
        # Angles in deg along x, y, z
        return self._nxs_file["entry/sample/crystal_rotation"][...]

    def get_scan(self, index=None) -> Scan:
        image_range = (1, self.get_num_images())
        properties = {"time_of_flight": tuple(self._get_time_of_flight())}
        return ScanFactory.make_scan_from_properties(
            image_range=image_range, properties=properties
        )

    def get_proton_charge(self) -> float:
        """McStas Simulations don't have a proton charge
        so this is a calculated value"""
        return self._nxs_file["entry/metadata/mcstas_weight2count_scale_factor"][
            ...
        ].item()


class FormatESSNMX_McStas(FormatHDF5):
    """
    Class to read files from NMX
    https://europeanspallationsource.se/instruments/nmx
    preprocessed files in scipp to obtain binned data

    Simulated data coming from McStas.
    """

    def __init__(self, image_file, **kwargs) -> None:
        if not FormatESSNMX_McStas.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        self._nxs_file = h5py.File(image_file, "r")
        self._raw_data = None

    @staticmethod
    def understand(image_file: str) -> bool:
        try:
            return FormatESSNMX_McStas.is_nmx_file(image_file)
        except (OSError, KeyError):
            return False

    @staticmethod
    def is_nmx_file(image_file: str) -> bool:
        def get_name(image_file):
            try:
                with h5py.File(image_file, "r") as handle:
                    return handle["entry/instrument/name"][...].item().decode()
            except (OSError, KeyError, AttributeError):
                return ""

        def is_McStas(image_file: str) -> bool:
            try:
                with h5py.File(image_file, "r") as handle:
                    if (
                        handle.get(
                            "entry/metadata/mcstas_weight2count_scale_factor", None
                        )
                        is not None
                    ):
                        return True
                    return False
            except (OSError, KeyError, AttributeError):
                return False

        return ("NMX" in get_name(image_file)) and (is_McStas(image_file))

    def get_instrument_name(self) -> str:
        return "NMX"

    def get_experiment_description(self) -> str:
        return "Simulated data"

    def _load_raw_data(self) -> None:
        raw_data = []
        for panel in self._get_panels():
            spectra = panel["data"][...]
            raw_data.append(flumpy.from_numpy(np.ascontiguousarray(spectra)))

        self._raw_data = tuple(raw_data)

    def get_raw_data(
        self, index: int, use_loaded_data: bool = False
    ) -> tuple[flex.int]:
        raw_data = []
        # image_size = self._get_image_size()
        # total_pixels = image_size[0] * image_size[1]

        if use_loaded_data:
            if self._raw_data is None:
                self._load_raw_data()
            for panel in self._raw_data:
                data = panel[:, :, index : index + 1].astype(np.int32)
                data.reshape(flex.grid(panel.all()[0], panel.all()[1]))
                data.matrix_transpose_in_place()
                raw_data.append(data)

        else:
            for panel in self._get_panels():
                spectra = panel["data"][:, :, index].astype(np.int32)
                raw_data.append(flumpy.from_numpy(np.ascontiguousarray(spectra)))

        return tuple(raw_data)

    def _get_time_channel_bins(self) -> list[float]:
        # (usec)
        # the tofs are recorded separately per panel but they
        # should all be the same
        for panel in self._get_panels():
            return panel["time_of_flight"][...] * 1e6

    def _get_time_of_flight(self) -> list[float]:
        # (usec)
        bins = self._get_time_channel_bins()
        return [float((bins[i] + bins[i + 1]) * 0.5) for i in range(len(bins) - 1)]

    def get_num_images(self) -> int:
        return len(self._get_time_of_flight())

    def get_detector(self, index: int = None) -> Detector:
        panel_names = self._get_panel_names()
        panel_type = self._get_panel_type()
        trusted_range = self._get_panel_trusted_range()
        pixel_size = self._get_pixel_size()
        gain = self._get_panel_gain()
        detector = Detector()
        root = detector.hierarchy()
        panels = self._get_panels()
        panel_projections = self._get_panel_projections_2d(panels)
        for panel_dset, panel_name in zip(panels, panel_names):
            panel = root.add_panel()
            panel.set_type(panel_type)
            panel.set_name(panel_name)
            panel.set_image_size(
                self._get_image_size(panel_dset)
            )  # XXX fix to include detectors possibly not being the same size
            panel.set_trusted_range(trusted_range)
            panel.set_pixel_size(pixel_size)
            fast_axis = self._get_panel_fast_axes(panel_dset)
            slow_axis = self._get_panel_slow_axes(panel_dset)
            panel_origin = self._get_panel_origins(panel_dset)
            panel.set_local_frame(fast_axis, slow_axis, panel_origin)

            panel.set_gain(gain)
            i = int(panel_name[-1])
            r, t = panel_projections[i]
            r = tuple(map(int, r))
            t = tuple(map(int, t))
            panel.set_projection_2d(r, t)

        return detector

    def _get_num_panels(self) -> int:
        return len(self._get_panels())

    def _get_panels(self) -> list[h5py._hl.group.Group]:
        """get the detector panel locations in file"""
        panels = []
        inst_dset = self._nxs_file["/entry/instrument/"]
        for _, dset in inst_dset.items():
            if dset.attrs.get("NX_class") == "NXdetector":
                panels.append(dset)
        return panels

    def _get_panel_names(self) -> list[str]:
        panel_names = []
        inst_dset = self._nxs_file["/entry/instrument/"]
        for name, dset in inst_dset.items():
            if dset.attrs.get("NX_class") == "NXdetector":
                panel_names.append(name)
        return panel_names

    def _get_panel_name(self, panel) -> str:
        return panel.name.split("/")[-1]

    def _get_panel_type(self) -> str:
        return "Triple_GEM_Gd"

    def _get_image_size(self, panel) -> tuple[int, int]:
        # (px)
        dset = panel["data"]
        return dset[:, :, 0].shape

    def _get_panel_trusted_range(self) -> tuple[int, int]:
        # 4 * 1280**2 plus buffer
        return (-1, 2**64 - 1)

    def _get_pixel_size(self) -> tuple[float, float]:
        # (mm)
        return (0.4, 0.4)

    def _get_panel_fast_axes(self, panel) -> tuple[float, float, float]:
        # return ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
        fast_axis = panel["fast_axis"][...]
        return tuple(fast_axis)

    def _get_panel_slow_axes(self, panel) -> tuple[float, float, float]:
        # return ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        slow_axis = panel["slow_axis"][...]
        return tuple(slow_axis)

    def _get_panel_origins(self, panel) -> tuple[float, float, float]:
        # (mm)
        # return ((-250, -250.0, -292.0), (290, -250.0, -250), (-290, -250.0, 250.0))

        origin = panel["origin"][...]
        panelnum = int(panel.name[-1])
        origin *= 1000  # convert to mm
        corrfact = self._get_correction_factor()[panelnum]

        corrorg = origin + corrfact
        return tuple(corrorg)

    def _get_correction_factor(self):
        return np.array([[0, -256, -256], [-256.0, -256.0, 0], [0, -256, 256]])

    def _get_panel_projections_2d(self, panels) -> dict[int : tuple[tuple, tuple]]:
        p_w, p_h = self._get_image_size(panels[0])  # XXX fix later
        p_w += 10
        p_h += 10
        panel_pos = {
            int(panels[0].name[-1]): ((-1, 0, 0, -1), (p_h, 0)),
            int(panels[1].name[-1]): ((-1, 0, 0, -1), (p_h, p_w)),
            int(panels[2].name[-1]): ((-1, 0, 0, -1), (p_h, -p_w)),
        }

        return panel_pos

    def get_beam(self, index: int = None) -> PolychromaticBeam:
        direction = self._get_sample_to_source_direction()
        distance = self._get_sample_to_source_distance()
        wavelength_range = self._get_wavelength_range()
        return BeamFactory.make_polychromatic_beam(
            direction=direction,
            sample_to_source_distance=distance,
            probe=Probe.neutron,
            wavelength_range=wavelength_range,
        )

    def _get_sample_to_source_direction(self) -> tuple[float, float, float]:
        return (0, 0, -1)

    def _get_wavelength_range(self) -> tuple[float, float]:
        # (A)
        tofs = np.array(self._get_time_of_flight()) / 1e6
        tof_low = tofs[0]
        tof_high = tofs[-1]
        lambda_low = self._tof_to_lambda(tof_low)
        lambda_high = self._tof_to_lambda(tof_high)
        return (round(lambda_low, 2), round(lambda_high, 2))

    def _tof_to_lambda(self, tof):
        """given tof in s, return lambda in Angstrom"""
        neutron_mass = 1.67492749804e-27
        h = 6.62607015e-34
        distance = self._get_sample_to_source_distance() / 1000  # convert to m
        return h / (neutron_mass * (distance / tof)) * 1e10

    def _get_sample_to_source_distance(self) -> float:
        """get sample to source distance in mm"""
        try:
            dist = abs(self._nxs_file["entry/instrument/source/distance"][...]) * 1000
            return dist
        except (KeyError, ValueError):
            logger.warning("sample to moderator_distance not found, using dummy value")
            return 156714

    def _get_panel_gain(self) -> float:
        return 1.0

    def get_goniometer_phi_angle(self) -> float:
        return self.get_goniometer_orientations()[1]

    def get_goniometer(self, index: int = None) -> Goniometer:
        rotation_axis = (0.0, 1.0, 0.0)
        fixed_rotation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        goniometer = GoniometerFactory.make_goniometer(rotation_axis, fixed_rotation)
        try:
            angles = self.get_goniometer_orientations()
        except KeyError:
            logger.warning("crystal_rotation not found, using default")
            return goniometer
        axes = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        for idx, angle in enumerate(angles):
            goniometer.rotate_around_origin(axes[idx], angle)
        return goniometer

    def get_goniometer_orientations(self) -> tuple[float, float, float]:
        # Angles in deg along x, y, z
        return self._nxs_file["entry/sample/crystal_rotation"][...]

    def get_scan(self, index=None) -> Scan:
        image_range = (1, self.get_num_images())
        properties = {"time_of_flight": tuple(self._get_time_of_flight())}
        return ScanFactory.make_scan_from_properties(
            image_range=image_range, properties=properties
        )

    def get_proton_charge(self) -> float:
        """McStas Simulations don't have a proton charge
        so this is a calculated value"""
        return self._nxs_file["entry/metadata/mcstas_weight2count_scale_factor"][
            ...
        ].item()
