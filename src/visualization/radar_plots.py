import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# Configuration
# ============================================================

# Column names in metadata.parquet / prediction dicts that describe a
# ground-truth or predicted object's footprint size, used to draw a real
# bounding box instead of a single point. Adjust these if the actual
# schema uses different names -- if none of these are present, the
# functions below fall back to drawing a marker at the center point.
GT_WIDTH_COLUMNS = ("width_m", "bbox_width", "extent_azimuth_m")
GT_LENGTH_COLUMNS = ("length_m", "bbox_length", "extent_range_m")

# Optional class_id -> human-readable name mapping for legend/labels.
CLASS_ID_TO_NAME = {
    0: "Clutter",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    4: "Class 4",
    5: "Class 5",
    6: "Class 6",
}


# ============================================================
# Core scaling / map helpers
# ============================================================

def magnitude_to_db(data):
    """Convert magnitude to decibels: 20*log10(|data| + eps)."""
    return 20 * np.log10(np.abs(data) + 1e-12)


def create_range_doppler_map(radar_cube):
    """
    Average the azimuth dimension and create a Range-Doppler matrix.

    Input shape:  (range_bins, doppler_bins, azimuth_bins) -> (64, 255, 64)
    Output shape: (range_bins, doppler_bins)
    """
    magnitude = np.abs(radar_cube)
    rdm = np.mean(magnitude, axis=2)
    return rdm


def create_range_azimuth_map(radar_cube):
    """
    Average the Doppler dimension and create a Range-Azimuth (BEV) matrix.

    Input shape:  (range_bins, doppler_bins, azimuth_bins) -> (64, 255, 64)
    Output shape: (range_bins, azimuth_bins)
    """
    magnitude = np.abs(radar_cube)
    ra_map = np.mean(magnitude, axis=1)
    return ra_map


# ============================================================
# Individual plot builders (all accept an optional `ax` and never
# call plt.show() themselves, so they compose into subplots/GIF frames)
# ============================================================

def plot_range_doppler_map(radar_cube, velocity_axis, range_axis, ax=None):
    """Plot the Range-Doppler Map."""
    rdm = create_range_doppler_map(radar_cube)
    rdm_db = magnitude_to_db(rdm)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        rdm_db,
        aspect="auto",
        origin="lower",
        extent=[velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]],
        cmap="jet"
    )

    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Range-Doppler Map")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    return ax


def plot_polar_scope(radar_cube, azimuth_deg, range_axis, ax=None):
    """
    Create an Azimuth-Range polar radar-style plot.
    Radius -> Range, Angle -> Azimuth.
    """
    ra_map = create_range_azimuth_map(radar_cube)

    angle_rad = np.radians(azimuth_deg)
    valid = ~np.isnan(angle_rad)

    theta = angle_rad[valid]
    data = ra_map[:, valid]

    theta_grid, range_grid = np.meshgrid(theta, range_axis)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")

    ax.pcolormesh(theta_grid, range_grid, data, shading="auto", cmap="jet")
    ax.set_title("Azimuth-Range Polar Scope", pad=20)
    return ax


def _extract_peak_coords(peaks, range_axis, azimuth_deg):
    """Convert a list of peaks (dicts or tuples of bin indices) into
    (range_m, azimuth_deg) coordinate lists."""
    det_r, det_az = [], []
    for p in peaks:
        r_bin = p["r_bin"] if isinstance(p, dict) else p[0]
        az_bin = p["az_bin"] if isinstance(p, dict) else p[2]
        det_r.append(range_axis[r_bin])
        det_az.append(azimuth_deg[az_bin])
    return det_r, det_az


def _first_available_column(row_or_dict, candidate_columns):
    """Return the value of the first matching column/key found (dict or
    pandas Series), skipping missing keys and NaN/None values. Returns
    None if no candidate column has a usable value."""
    for col in candidate_columns:
        if isinstance(row_or_dict, dict):
            has_col = col in row_or_dict
        else:
            has_col = col in row_or_dict.index

        if not has_col:
            continue

        value = row_or_dict[col]
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue

        return value

    return None


def _draw_bbox_or_point(ax, center_az, center_range, width, length, edgecolor, label=None, linewidth=2):
    """Draw a real Rectangle bounding box centered on (center_az, center_range)
    if width/length are both known; otherwise fall back to a marker point."""
    if width is not None and length is not None:
        rect = Rectangle(
            (center_az - width / 2, center_range - length / 2),
            width,
            length,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
            label=label
        )
        ax.add_patch(rect)
    else:
        ax.scatter(
            [center_az], [center_range],
            facecolors="none", edgecolors=edgecolor,
            s=100, linewidths=linewidth, label=label
        )


def plot_bev_detections(radar_cube, peaks, azimuth_deg, range_axis, gt_metadata=None, ax=None):
    """
    Plot the Bird's-Eye-View (BEV) Range-Azimuth heatmap overlaid with
    CFAR candidate centroids (points) and ground-truth objects. Ground
    truth is drawn as a real bounding box when width/length columns are
    present in gt_metadata (see GT_WIDTH_COLUMNS / GT_LENGTH_COLUMNS),
    otherwise as a center-point marker.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    ra_map = create_range_azimuth_map(radar_cube)
    ra_db = magnitude_to_db(ra_map)

    im = ax.imshow(
        ra_db,
        aspect="auto",
        origin="lower",
        extent=[azimuth_deg[0], azimuth_deg[-1], range_axis[0], range_axis[-1]],
        cmap="jet"
    )

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title("BEV Range-Azimuth Plane with Detections")

    # 1. Overlay CFAR peaks (centroids only -- CFAR proposals have no
    #    inherent extent, so they are always plotted as points).
    if peaks is not None and len(peaks) > 0:
        det_r, det_az = _extract_peak_coords(peaks, range_axis, azimuth_deg)
        ax.scatter(
            det_az, det_r, c="magenta", marker="x", s=60,
            linewidths=2, label="CFAR Candidates"
        )

    # 2. Overlay Ground Truth (bounding box if available, else point)
    if gt_metadata is not None and len(gt_metadata) > 0:
        first_row_label = "Ground Truth"
        for idx, row in gt_metadata.iterrows():
            width = _first_available_column(row, GT_WIDTH_COLUMNS)
            length = _first_available_column(row, GT_LENGTH_COLUMNS)
            _draw_bbox_or_point(
                ax,
                center_az=row["azimuth_deg"],
                center_range=row["range_m"],
                width=width,
                length=length,
                edgecolor="lime",
                label=first_row_label
            )
            first_row_label = None  # only label the first one to avoid duplicate legend entries

    _add_legend_if_any(ax)
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    return ax


def plot_inference_overlay(radar_cube, predictions, azimuth_deg, range_axis, ax=None):
    """
    Overlay Member 4's inference predictions (predicted class + bounding
    box / centroid) onto the Range-Azimuth BEV plane.

    `predictions` is expected to be an iterable of dicts, each with at
    least:
        - "range_m": float
        - "azimuth_deg": float
        - "class_id": int
    and optionally:
        - "width_m" / "length_m" (or any of GT_WIDTH_COLUMNS / GT_LENGTH_COLUMNS)
          for drawing a real bounding box
        - "confidence": float, appended to the label if present

    NOTE: the exact schema of Member 4's output has not been confirmed
    yet -- adjust the key names below once that contract is finalized.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    ra_map = create_range_azimuth_map(radar_cube)
    ra_db = magnitude_to_db(ra_map)

    im = ax.imshow(
        ra_db,
        aspect="auto",
        origin="lower",
        extent=[azimuth_deg[0], azimuth_deg[-1], range_axis[0], range_axis[-1]],
        cmap="jet"
    )

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Range (m)")
    ax.set_title("Inference Predictions Overlay")

    if predictions is not None:
        for pred in predictions:
            class_id = pred.get("class_id", -1)
            class_name = CLASS_ID_TO_NAME.get(class_id, f"class {class_id}")
            confidence = pred.get("confidence")
            label = f"{class_name}" + (f" ({confidence:.2f})" if confidence is not None else "")

            width = _first_available_column(pred, GT_WIDTH_COLUMNS)
            length = _first_available_column(pred, GT_LENGTH_COLUMNS)

            _draw_bbox_or_point(
                ax,
                center_az=pred["azimuth_deg"],
                center_range=pred["range_m"],
                width=width,
                length=length,
                edgecolor="white",
                label=None,
                linewidth=2.5
            )
            ax.annotate(
                label,
                xy=(pred["azimuth_deg"], pred["range_m"]),
                xytext=(5, 5),
                textcoords="offset points",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none")
            )

    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    return ax


def _add_legend_if_any(ax):
    """Call ax.legend() only if there's at least one labeled artist,
    to avoid matplotlib's 'no artists with labels found' UserWarning."""
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc="upper right")


# ============================================================
# Composite / side-by-side dashboard
# ============================================================

def create_verification_dashboard(
    radar_cube,
    peaks,
    velocity_axis,
    azimuth_deg,
    range_axis,
    gt_metadata=None,
    predictions=None,
    save_path=None
):
    """
    Build a single figure with side-by-side verification plots:
        1) Range-Doppler Map
        2) BEV Range-Azimuth with CFAR + Ground Truth overlays
        3) Inference predictions overlay (if provided)

    If `save_path` is given, the figure is saved to disk instead of /
    in addition to being returned (useful as a single frame when
    building a GIF -- see `export_detection_gif` below).
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    plot_range_doppler_map(radar_cube, velocity_axis, range_axis, ax=axes[0])
    plot_bev_detections(radar_cube, peaks, azimuth_deg, range_axis, gt_metadata=gt_metadata, ax=axes[1])

    if predictions is not None:
        plot_inference_overlay(radar_cube, predictions, azimuth_deg, range_axis, ax=axes[2])
    else:
        axes[2].axis("off")
        axes[2].set_title("No predictions available")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120)

    return fig, axes


# ============================================================
# GIF / video export of a detection sequence
# ============================================================

def export_detection_gif(
    frames,
    velocity_axis,
    azimuth_deg,
    range_axis,
    output_path="radar_detections.gif",
    fps=2
):
    """
    Render an animated GIF of the BEV detection view across a sequence
    of frames.

    `frames` is a list of dicts, one per frame, each with:
        - "radar_cube": (range_bins, doppler_bins, azimuth_bins) array
        - "peaks": list of CFAR peaks for that frame (optional)
        - "gt_metadata": ground-truth rows for that frame (optional)
        - "predictions": Member 4 predictions for that frame (optional)

    Uses matplotlib's PillowWriter, so it only depends on Pillow
    (already a matplotlib dependency) -- no extra packages required.
    """
    if len(frames) == 0:
        raise ValueError("`frames` must contain at least one frame to export a GIF.")

    fig, ax = plt.subplots(figsize=(9, 7))

    def _draw_frame(i):
        ax.clear()
        frame = frames[i]
        plot_bev_detections(
            frame["radar_cube"],
            frame.get("peaks"),
            azimuth_deg,
            range_axis,
            gt_metadata=frame.get("gt_metadata"),
            ax=ax
        )
        if frame.get("predictions") is not None:
            # Overlay predictions on top of the same axes as annotations only
            for pred in frame["predictions"]:
                class_id = pred.get("class_id", -1)
                class_name = CLASS_ID_TO_NAME.get(class_id, f"class {class_id}")
                ax.annotate(
                    class_name,
                    xy=(pred["azimuth_deg"], pred["range_m"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none")
                )
        ax.set_title(f"Frame {i} — BEV Detections")
        return ax.images + ax.collections + ax.patches

    anim = FuncAnimation(fig, _draw_frame, frames=len(frames), blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"Saved detection GIF: {output_path} ({len(frames)} frames)")
    return output_path


 
if __name__ == "__main__":
    # This block is intentionally left without a real call -- plug in
    # real (or mock) radar_cube / peaks / metadata / predictions here
    # once available, e.g.:
    #
    # fig, axes = create_verification_dashboard(
    #     radar_cube, peaks, velocity_axis, azimuth_deg, range_axis,
    #     gt_metadata=metadata_slice, predictions=member4_predictions
    # )
    # plt.show()
    pass