import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
import os

def _minmax_norm(a: np.ndarray) -> np.ndarray:
    """Convert non-nan data to range [0,1]"""
    a = a.astype(np.float64)
    mask = np.isfinite(a)
    if not np.any(mask):
        return np.full_like(a, np.nan, dtype=np.float64)
    vmin = np.nanmin(a[mask])
    vmax = np.nanmax(a[mask])
    if vmax > vmin:
        out = (a - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(a)
    out[~mask] = np.nan
    return out

def tif_band1_to_excel(
    in_tif: str,
    out_xlsx: str,
    target_rows: int | None = None,
    target_cols: int | None = None,
    resampling: str = "average",   # "average" | "bilinear" | "nearest"
    sheet_name: str = "E"
):
    """
    Always read the band 1 of GeoTIFF → min–max norm → (opt)resampling → to Excel。
    The excel represents a matrix (rows×cols) ；read it with pd.read_excel(..., index_col=0)。
    """
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    if resampling not in resampling_map:
        raise ValueError(f"Unknown resampling: {resampling}")

    with rasterio.open(in_tif) as ds:
        band1 = ds.read(1, masked=False).astype(np.float64)
        nodata = ds.nodata
        if nodata is not None:
            band1 = np.where(band1 == nodata, np.nan, band1)

        # Min-max with original resolution
        band1 = _minmax_norm(band1)

        if target_rows is not None and target_cols is not None:
            dst = np.full((target_rows, target_cols), np.nan, dtype=np.float64)
            reproject(
                source=band1,
                destination=dst,
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=rasterio.transform.from_bounds(
                    *ds.bounds, width=target_cols, height=target_rows
                ),
                dst_crs=ds.crs,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=resampling_map[resampling],
            )
            arr = dst
        else:
            arr = band1

    # export as xlsx： "1..rows", "1..cols"
    rows, cols = arr.shape
    df = pd.DataFrame(
        arr,
        index=[str(i+1) for i in range(rows)],
        columns=[str(j+1) for j in range(cols)]
    )
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name)

    print(f"Saved: {out_xlsx}  ({rows} x {cols})")

# —— Usage example ——
# 1) Original
# tif_band1_to_excel(r"E:\data\env.tif", r"E:\TheoEcoFramework\config_spatial_1.xlsx")

# 2) Resampling
# tif_band1_to_excel(r"E:\data\env.tif", r"E:\TheoEcoFramework\config_spatial_1.xlsx",
#                    target_rows=32, target_cols=32, resampling="average")


if __name__ == "__main__":

    os.chdir('E:\\TheoEcoFramework')
    tif_band1_to_excel(r"TND7ZQ.tiff", r"config_resistance_2.xlsx")

