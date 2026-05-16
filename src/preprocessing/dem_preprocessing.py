"""
DEM Preprocessing Pipeline for NeuralDelta
==========================================
Processes raw SRTM GeoTIFF into analysis-ready terrain products.

Pipeline:
    1. Void filling (NoData gaps → interpolated values)
    2. Reprojection WGS84 → UTM Zone 51N (EPSG:32651)
    3. Resampling to 5 resolution levels (30, 50, 100, 200, 500 m)
    4. Bed slope field computation (Sox, Soy) for SWE

Scientific rationale:
    The 2D SWE require spatially continuous bed slope fields Sox = -dz/dx
    and Soy = -dz/dy in metric coordinates (m/m). SRTM voids, if unfilled,
    propagate as NaN through gradient computation, corrupting the physics
    residuals in the PINN loss function.

Author: [YOUR NAME]
Date: [TODAY'S DATE]
Reference: Horritt & Bates (2002); Raissi et al. (2019)
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ── Path configuration ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DEM      = PROJECT_ROOT / "data" / "raw" / "dem" / "cagayan_srtm_raw.tif"
PROCESSED    = PROJECT_ROOT / "data" / "processed" / "dem"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Target CRS and resolution levels ───────────────────────────────────────────
TARGET_CRS = CRS.from_epsg(32651)          # UTM Zone 51N — metric, Philippines

# Five resolution levels for H2 (speed vs accuracy scaling experiment)
RESOLUTIONS = {
    "30m":  30,
    "50m":  50,
    "100m": 100,
    "200m": 200,
    "500m": 500,
}


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1: Void Filling
# ══════════════════════════════════════════════════════════════════════════════

def fill_voids(dem_array: np.ndarray, nodata_value: float) -> np.ndarray:
    """
    Fill NoData voids using nearest-neighbor interpolation.

    Scientific basis:
        SRTM voids occur in radar shadow zones (steep terrain, water surfaces).
        In the Cagayan delta, voids most commonly appear over water bodies and
        steep terrain at domain edges. Nearest-neighbor fill is appropriate
        here because void regions are small relative to domain size; kriging
        or spline interpolation would be excessive for this application.

    Parameters
    ----------
    dem_array : np.ndarray
        2D elevation array with NoData values
    nodata_value : float
        The value representing NoData (typically -32768 for SRTM INT16)

    Returns
    -------
    np.ndarray
        Void-filled elevation array (float32)
    """
    arr = dem_array.astype(np.float32)
    
    # Identify void pixels
    if nodata_value is not None:
        void_mask = (arr == nodata_value) | np.isnan(arr)
    else:
        void_mask = np.isnan(arr)
    
    n_voids = void_mask.sum()
    
    if n_voids == 0:
        log.info("No void pixels detected — skipping fill step")
        return arr
    
    log.info(f"Filling {n_voids:,} void pixels ({100*n_voids/arr.size:.2f}% of domain)")
    
    # Distance transform: for each void pixel, find the nearest valid pixel
    # distance_transform_edt returns indices of nearest valid pixels
    distance, nearest_idx = distance_transform_edt(
        void_mask, return_indices=True
    )
    
    # Fill voids using nearest valid elevation
    filled = arr.copy()
    filled[void_mask] = arr[nearest_idx[0][void_mask], nearest_idx[1][void_mask]]
    
    # Remove any remaining NaN (safety check)
    filled = np.where(np.isnan(filled), 0.0, filled)
    
    log.info(f"Void fill complete — max gap distance: {distance.max():.1f} pixels")
    return filled


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2: Reproject and Resample
# ══════════════════════════════════════════════════════════════════════════════

def reproject_and_resample(
    src_path: Path,
    dst_path: Path,
    target_res_m: int
) -> dict:
    """
    Reproject DEM from WGS84 to UTM 51N and resample to target resolution.

    Scientific basis:
        The SWE are formulated in Cartesian (x, y) coordinates in meters.
        Working in geographic coordinates (degrees) would make the spatial
        gradient dz/dx dimensionally inconsistent with the momentum equations.
        UTM Zone 51N is the standard metric projection for 120-126°E longitude.

        Bilinear resampling is used (not nearest-neighbor) because elevation
        is a continuous field. Nearest-neighbor resampling creates staircase
        artifacts in slope fields, which corrupt the SWE bed slope terms.

    Parameters
    ----------
    src_path : Path
        Input GeoTIFF (void-filled, WGS84)
    dst_path : Path
        Output GeoTIFF path
    target_res_m : int
        Target pixel size in meters

    Returns
    -------
    dict
        Metadata about the reprojected raster
    """
    log.info(f"Reprojecting → UTM 51N at {target_res_m}m resolution ...")

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_res_m    # Force exact pixel size in meters
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': TARGET_CRS,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'nodata': np.nan,
            'compress': 'lzw'          # Lossless compression for GeoTIFFs
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear
            )

    meta = {
        'resolution_m': target_res_m,
        'width_px': width,
        'height_px': height,
        'domain_width_km': width * target_res_m / 1000,
        'domain_height_km': height * target_res_m / 1000
    }

    log.info(
        f"  → {width}×{height} pixels | "
        f"{meta['domain_width_km']:.1f}×{meta['domain_height_km']:.1f} km"
    )
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 3: Bed Slope Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_bed_slopes(dem_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute SWE bed slope fields Sox and Soy from DEM.

    Mathematical formulation:
        The 2D SWE momentum equations contain source terms:
            g * h * Sox  (x-momentum, gravitational forcing)
            g * h * Soy  (y-momentum, gravitational forcing)

        Where:
            Sox = -∂z/∂x   (negative x-gradient of bed elevation)
            Soy = -∂z/∂y   (negative y-gradient of bed elevation)

        These are computed using second-order central finite differences
        (numpy.gradient), which is equivalent to what FEniCS uses for
        the FEM benchmark — ensuring methodological consistency.

        Units: m/m (dimensionless slope ratio)

    Parameters
    ----------
    dem_path : Path
        Input GeoTIFF (UTM projected, meters)

    Returns
    -------
    Sox, Soy : np.ndarray
        Bed slope arrays in x and y directions
    stats : dict
        Summary statistics for validation
    """
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float32)
        res_x = src.res[0]   # pixel width in meters (UTM)
        res_y = src.res[1]   # pixel height in meters (UTM)

    # Replace NaN with nearest valid (safety)
    nan_mask = np.isnan(z)
    if nan_mask.any():
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(nan_mask, return_indices=True)
        z[nan_mask] = z[idx[0][nan_mask], idx[1][nan_mask]]

    # Second-order central differences
    # np.gradient returns [dz/drow, dz/dcol]
    # row increases southward → dz/drow = -dz/dy
    # col increases eastward  → dz/dcol =  dz/dx
    dz_drow, dz_dcol = np.gradient(z, res_y, res_x)

    # Convert from image to geographic convention
    # Sox = -dz/dx = -dz_dcol
    # Soy = -dz/dy = +dz_drow  (row increases south, y increases north)
    Sox = -dz_dcol
    Soy =  dz_drow

    stats = {
        'Sox_min':  float(Sox.min()),
        'Sox_max':  float(Sox.max()),
        'Sox_mean': float(Sox.mean()),
        'Soy_min':  float(Soy.min()),
        'Soy_max':  float(Soy.max()),
        'Soy_mean': float(Soy.mean()),
    }

    return Sox, Soy, stats


def save_slope_geotiff(array: np.ndarray, ref_path: Path, out_path: Path,
                       band_name: str) -> None:
    """Save a slope array as a GeoTIFF, inheriting projection from reference."""
    with rasterio.open(ref_path) as src:
        meta = src.meta.copy()
    meta.update({'dtype': 'float32', 'nodata': np.nan, 'compress': 'lzw'})
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(array.astype(np.float32), 1)
    log.info(f"Saved {band_name} → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_pipeline():
    """
    Execute the full DEM preprocessing pipeline.
    
    Output files produced:
        data/processed/dem/cagayan_dem_filled_wgs84.tif   ← void-filled, WGS84
        data/processed/dem/cagayan_dem_30m.tif            ← 30m UTM
        data/processed/dem/cagayan_dem_50m.tif            ← 50m UTM
        data/processed/dem/cagayan_dem_100m.tif           ← 100m UTM (primary)
        data/processed/dem/cagayan_dem_200m.tif           ← 200m UTM
        data/processed/dem/cagayan_dem_500m.tif           ← 500m UTM
        data/processed/dem/cagayan_Sox_100m.tif           ← x bed slope
        data/processed/dem/cagayan_Soy_100m.tif           ← y bed slope
    """
    log.info("=" * 60)
    log.info("NeuralDelta DEM Preprocessing Pipeline")
    log.info("=" * 60)

    # ── Step 1: Void filling ────────────────────────────────────────────────
    log.info("STEP 1: Void filling")
    filled_wgs84_path = PROCESSED / "cagayan_dem_filled_wgs84.tif"

    with rasterio.open(RAW_DEM) as src:
        dem_raw = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata

    dem_filled = fill_voids(dem_raw, nodata)

    meta.update({'dtype': 'float32', 'nodata': np.nan, 'compress': 'lzw'})
    with rasterio.open(filled_wgs84_path, 'w', **meta) as dst:
        dst.write(dem_filled, 1)
    log.info(f"Void-filled DEM saved → {filled_wgs84_path.name}")

    # ── Step 2: Reproject and resample to all 5 resolution levels ──────────
    log.info("STEP 2: Reprojection and resampling")
    resolution_meta = {}

    for name, res_m in RESOLUTIONS.items():
        out_path = PROCESSED / f"cagayan_dem_{name}.tif"
        meta_info = reproject_and_resample(filled_wgs84_path, out_path, res_m)
        resolution_meta[name] = meta_info

    # ── Step 3: Compute bed slopes at primary resolution (100m) ────────────
    log.info("STEP 3: Bed slope computation (100m primary resolution)")
    dem_100m = PROCESSED / "cagayan_dem_100m.tif"
    Sox, Soy, slope_stats = compute_bed_slopes(dem_100m)

    sox_path = PROCESSED / "cagayan_Sox_100m.tif"
    soy_path = PROCESSED / "cagayan_Soy_100m.tif"
    save_slope_geotiff(Sox, dem_100m, sox_path, "Sox")
    save_slope_geotiff(Soy, dem_100m, soy_path, "Soy")

    log.info(f"Slope stats: Sox ∈ [{slope_stats['Sox_min']:.4f}, "
             f"{slope_stats['Sox_max']:.4f}], "
             f"Soy ∈ [{slope_stats['Soy_min']:.4f}, "
             f"{slope_stats['Soy_max']:.4f}]")

    # ── Step 4: Generate validation figure ─────────────────────────────────
    log.info("STEP 4: Generating validation figures")
    _plot_preprocessing_summary(dem_100m, Sox, Soy, slope_stats)

    # ── Step 5: Save metadata report ───────────────────────────────────────
    import json
    report = {
        'pipeline': 'DEM Preprocessing',
        'source': 'SRTM GL1 (OpenTopography)',
        'crs_input': 'EPSG:4326 (WGS84)',
        'crs_output': 'EPSG:32651 (UTM Zone 51N)',
        'void_fill_method': 'nearest_neighbor (scipy distance_transform_edt)',
        'resample_method': 'bilinear',
        'resolutions': resolution_meta,
        'slope_stats_100m': slope_stats,
    }
    report_path = PROCESSED / "preprocessing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"Output directory: {PROCESSED}")
    log.info("=" * 60)

    return report


def _plot_preprocessing_summary(dem_path, Sox, Soy, stats):
    """Generate a 4-panel validation figure."""
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float32)
        z[z == src.nodata] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DEM Preprocessing Validation — Cagayan River Delta\n'
                 '(UTM Zone 51N, 100m resolution)', fontsize=13)

    # Panel 1: Elevation
    im1 = axes[0, 0].imshow(z, cmap='terrain', aspect='auto')
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
    axes[0, 0].set_title('Bed Elevation z (m)')

    # Panel 2: Sox
    vmax = max(abs(Sox.min()), abs(Sox.max()))
    im2 = axes[0, 1].imshow(Sox, cmap='RdBu_r', aspect='auto',
                             vmin=-vmax, vmax=vmax)
    plt.colorbar(im2, ax=axes[0, 1], label='Sox (m/m)')
    axes[0, 1].set_title('Bed Slope Sox = -∂z/∂x')

    # Panel 3: Soy
    vmax = max(abs(Soy.min()), abs(Soy.max()))
    im3 = axes[1, 0].imshow(Soy, cmap='RdBu_r', aspect='auto',
                             vmin=-vmax, vmax=vmax)
    plt.colorbar(im3, ax=axes[1, 0], label='Soy (m/m)')
    axes[1, 0].set_title('Bed Slope Soy = -∂z/∂y')

    # Panel 4: Slope magnitude
    slope_mag = np.sqrt(Sox**2 + Soy**2)
    im4 = axes[1, 1].imshow(slope_mag, cmap='hot_r', aspect='auto')
    plt.colorbar(im4, ax=axes[1, 1], label='|S| (m/m)')
    axes[1, 1].set_title('Slope Magnitude |S| = √(Sox² + Soy²)')

    plt.tight_layout()
    fig_path = (Path(__file__).resolve().parents[2] / "outputs" / "figures"
                / "dem_preprocessing_validation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Validation figure saved → {fig_path.name}")


if __name__ == "__main__":
    report = run_preprocessing_pipeline()