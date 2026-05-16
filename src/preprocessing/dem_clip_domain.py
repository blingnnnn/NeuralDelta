"""
Computational Domain Clipping for NeuralDelta
==============================================
Clips the full-extent DEM to the flood-relevant study domain:
the lower Cagayan River valley and delta floodplain.

Scientific rationale:
    The 2D SWE PINN computational domain must be restricted to terrain
    where flood inundation physically occurs. Including high-elevation
    mountain terrain would:
      1. Waste collocation points on non-floodable areas
      2. Corrupt slope statistics with mountain-scale gradients
      3. Misalign SAR flood mask coverage with model domain
      4. Violate the SWE shallow-water assumption (SWE are only valid
         for h << horizontal length scale, which breaks in steep terrain)

    The domain is defined by the lower Cagayan River corridor from
    Tuguegarao City (upstream gauge) to Aparri (river mouth).

Author: Gerald Del Rosario
Date: 05.16.2026
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED    = PROJECT_ROOT / "data" / "processed" / "dem"

# ── Study domain definition (WGS84) ──────────────────────────────────────────
# Lower Cagayan River valley: Tuguegarao → Aparri delta
# Chosen to capture the flood inundation zone documented during
# Typhoon Ulysses (Nov 2020), Lawin (Oct 2016), Mangkhut (Sep 2018)

DOMAIN = {
    "west":  121.55,   # Excludes Cordillera Central
    "east":  122.00,   # Excludes Sierra Madre high terrain
    "south": 17.55,    # Just below Tuguegarao City
    "north": 18.45,    # Just past Aparri river mouth
}

TARGET_CRS = CRS.from_epsg(32651)   # UTM Zone 51N

RESOLUTIONS = {
    "30m":  30,
    "50m":  50,
    "100m": 100,
    "200m": 200,
    "500m": 500,
}

# Elevation ceiling for the floodplain domain
# Pixels above this are mountain/upland and not relevant to flood inundation
ELEVATION_CEILING_M = 150.0


def clip_and_reproject(source_tif: Path, out_path: Path,
                       target_res_m: int) -> dict:
    """
    Clip to study domain, reproject to UTM 51N, resample to target resolution,
    and apply elevation ceiling to remove mountain artifacts.
    """
    bbox = box(DOMAIN["west"], DOMAIN["south"],
               DOMAIN["east"], DOMAIN["north"])
    geom = [mapping(bbox)]

    with rasterio.open(source_tif) as src:
        # Clip to bounding box
        clipped, clip_transform = rio_mask(src, geom, crop=True)
        clipped = clipped[0].astype(np.float32)

        # Replace SRTM nodata (-32768) with NaN
        clipped = np.where(clipped == -32768, np.nan, clipped)

        # Clamp negative elevations to 0 (coastal/water artifacts)
        n_negative = np.sum(clipped < 0)
        if n_negative > 0:
            log.info(f"  Clamping {n_negative} negative elevation pixels to 0.0m")
            clipped = np.where(clipped < 0, 0.0, clipped)

        clip_crs = src.crs

    # Write clipped WGS84 version temporarily
    tmp_path = PROCESSED / "_tmp_clipped_wgs84.tif"
    rows, cols = clipped.shape
    with rasterio.open(
        tmp_path, 'w',
        driver='GTiff',
        height=rows, width=cols,
        count=1, dtype='float32',
        crs=clip_crs,
        transform=clip_transform,
        nodata=np.nan
    ) as tmp:
        tmp.write(clipped, 1)

    # Reproject clipped raster to UTM 51N at target resolution
    with rasterio.open(tmp_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS,
            src.width, src.height,
            *src.bounds,
            resolution=target_res_m
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': TARGET_CRS,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'nodata': np.nan,
            'compress': 'lzw'
        })
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear
            )

    tmp_path.unlink()  # Remove temp file

    # Read result and compute stats
    with rasterio.open(out_path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == src.nodata] = np.nan

    valid = arr[~np.isnan(arr)]
    meta = {
        'resolution_m':       target_res_m,
        'width_px':           width,
        'height_px':          height,
        'domain_width_km':    round(width  * target_res_m / 1000, 1),
        'domain_height_km':   round(height * target_res_m / 1000, 1),
        'elev_min_m':         round(float(valid.min()), 2),
        'elev_max_m':         round(float(valid.max()), 2),
        'elev_mean_m':        round(float(valid.mean()), 2),
        'total_pixels':       int(arr.size),
    }

    log.info(
        f"  {target_res_m:>3}m → {width}×{height} px | "
        f"{meta['domain_width_km']}×{meta['domain_height_km']} km | "
        f"elev {meta['elev_min_m']}–{meta['elev_max_m']} m"
    )
    return meta


def compute_bed_slopes(dem_path: Path) -> tuple:
    """
    Compute SWE bed slope fields Sox = -∂z/∂x and Soy = -∂z/∂y.
    Identical formulation to Phase 3 but applied to clipped domain.
    """
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float32)
        res_x = src.res[0]
        res_y = src.res[1]
        meta  = src.meta.copy()

    z = np.where(np.isnan(z), 0.0, z)

    dz_drow, dz_dcol = np.gradient(z, res_y, res_x)
    Sox = -dz_dcol
    Soy =  dz_drow

    stats = {
        'Sox_min':  round(float(Sox.min()),  6),
        'Sox_max':  round(float(Sox.max()),  6),
        'Sox_mean': round(float(Sox.mean()), 6),
        'Soy_min':  round(float(Soy.min()),  6),
        'Soy_max':  round(float(Soy.max()),  6),
        'Soy_mean': round(float(Soy.mean()), 6),
    }
    return Sox, Soy, stats, meta


def save_geotiff(array, meta, out_path):
    m = meta.copy()
    m.update({'dtype': 'float32', 'nodata': np.nan,
              'count': 1, 'compress': 'lzw'})
    with rasterio.open(out_path, 'w', **m) as dst:
        dst.write(array.astype(np.float32), 1)


def plot_domain_validation(dem_path, Sox, Soy):
    """4-panel validation figure for the clipped floodplain domain."""
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype(np.float32)
        bounds = src.bounds
        res_m  = src.res[0]

    z[z == -9999] = np.nan

    # Axis labels in km from SW corner
    ny, nx = z.shape
    x_km = np.linspace(0, nx * res_m / 1000, nx)
    y_km = np.linspace(ny * res_m / 1000, 0, ny)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'Computational Domain — Lower Cagayan River Floodplain\n'
        '(Clipped, UTM Zone 51N, 100m resolution)\n'
        f'Domain: {nx * res_m / 1000:.0f} km × {ny * res_m / 1000:.0f} km',
        fontsize=13
    )

    ext = [0, nx * res_m / 1000, 0, ny * res_m / 1000]

    # Panel 1: Elevation with low-elevation emphasis
    im1 = axes[0, 0].imshow(
        z, cmap='terrain', aspect='auto', origin='upper',
        extent=ext, vmin=0, vmax=min(z[~np.isnan(z)].max(), 200)
    )
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')
    axes[0, 0].set_title('Bed Elevation z (m)\n[colorscale capped at 200m]')
    axes[0, 0].set_xlabel('Easting (km)')
    axes[0, 0].set_ylabel('Northing (km)')

    # Panel 2: Sox
    vmax = np.percentile(np.abs(Sox), 98)
    im2 = axes[0, 1].imshow(
        Sox, cmap='RdBu_r', aspect='auto', origin='upper',
        extent=ext, vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im2, ax=axes[0, 1], label='Sox (m/m)')
    axes[0, 1].set_title('Bed Slope Sox = −∂z/∂x\n[98th percentile scale]')
    axes[0, 1].set_xlabel('Easting (km)')

    # Panel 3: Soy
    vmax = np.percentile(np.abs(Soy), 98)
    im3 = axes[1, 0].imshow(
        Soy, cmap='RdBu_r', aspect='auto', origin='upper',
        extent=ext, vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im3, ax=axes[1, 0], label='Soy (m/m)')
    axes[1, 0].set_title('Bed Slope Soy = −∂z/∂y\n[98th percentile scale]')
    axes[1, 0].set_xlabel('Easting (km)')
    axes[1, 0].set_ylabel('Northing (km)')

    # Panel 4: Elevation histogram (floodplain zone only)
    valid = z[~np.isnan(z)].flatten()
    floodplain = valid[valid <= 50]
    axes[1, 1].hist(valid[valid <= 200], bins=80,
                    color='steelblue', edgecolor='white', linewidth=0.3,
                    label='All valid (≤200m)')
    axes[1, 1].hist(floodplain, bins=60,
                    color='coral', edgecolor='white', linewidth=0.3,
                    alpha=0.7, label='Floodplain (≤50m)')
    axes[1, 1].axvline(x=0, color='navy', linestyle='--', label='Sea level')
    axes[1, 1].axvline(x=30, color='red', linestyle='--',
                       label='~Flood threshold 30m')
    axes[1, 1].set_xlabel('Elevation (m)')
    axes[1, 1].set_ylabel('Pixel count')
    axes[1, 1].set_title('Elevation Distribution\n(focus on floodplain zone)')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = (PROJECT_ROOT / "outputs" / "figures"
                / "dem_domain_clipped_validation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Validation figure → {fig_path.name}")


def run():
    log.info("=" * 60)
    log.info("Computational Domain Clipping Pipeline")
    log.info(f"Domain: {DOMAIN}")
    log.info("=" * 60)

    source = PROCESSED / "cagayan_dem_filled_wgs84.tif"
    resolution_meta = {}

    # Clip and reproject all 5 resolution levels
    log.info("Clipping and reprojecting all resolution levels ...")
    for name, res_m in RESOLUTIONS.items():
        out_path = PROCESSED / f"cagayan_dem_{name}.tif"
        log.info(f"Processing {name} ...")
        meta = clip_and_reproject(source, out_path, res_m)
        resolution_meta[name] = meta

    # Compute bed slopes for primary resolution
    log.info("Computing bed slopes at 100m ...")
    dem_100m = PROCESSED / "cagayan_dem_100m.tif"
    Sox, Soy, slope_stats, raster_meta = compute_bed_slopes(dem_100m)

    sox_path = PROCESSED / "cagayan_Sox_100m.tif"
    soy_path = PROCESSED / "cagayan_Soy_100m.tif"
    save_geotiff(Sox, raster_meta, sox_path)
    save_geotiff(Soy, raster_meta, soy_path)

    log.info(f"Sox ∈ [{slope_stats['Sox_min']}, {slope_stats['Sox_max']}]")
    log.info(f"Soy ∈ [{slope_stats['Soy_min']}, {slope_stats['Soy_max']}]")

    # Validation figure
    log.info("Generating validation figure ...")
    plot_domain_validation(dem_100m, Sox, Soy)

    # Save report
    report = {
        'pipeline':       'Domain Clipping',
        'domain_wgs84':   DOMAIN,
        'crs_output':     'EPSG:32651 (UTM Zone 51N)',
        'resolutions':    resolution_meta,
        'slope_stats':    slope_stats,
        'notes': (
            'Domain clipped to lower Cagayan floodplain. '
            'Negative elevations clamped to 0.0m (coastal artifacts). '
            'Mountain terrain excluded above domain bounds.'
        )
    }
    report_path = PROCESSED / "domain_clipping_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    log.info("=" * 60)
    log.info("DOMAIN CLIPPING COMPLETE")
    log.info("=" * 60)
    return report


if __name__ == "__main__":
    run()