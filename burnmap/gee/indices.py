"""
Spectral index computation from S2 pre/post composites.

Returns pre, post, and delta for: NDVI, NDMI, NDRE, NBR.
Also returns dB12 (raw SWIR difference, used in label gates only).

Note: dNBR is for label construction only — it should NOT appear
in classifier feature sets (circularity).
"""

import ee


def _nd(img: ee.Image, b1: str, b2: str, name: str) -> ee.Image:
    """Normalized difference: (b1 - b2) / (b1 + b2)."""
    return img.normalizedDifference([b1, b2]).rename(name)


def compute_all(pre: ee.Image, post: ee.Image) -> dict:
    """
    Compute all spectral indices from S2 pre/post composites.

    Parameters
    ----------
    pre  : ee.Image with original S2 band names (B2..B12)
    post : ee.Image with original S2 band names

    Returns
    -------
    dict of ee.Image keyed by name (e.g. "NDVI_pre", "dNDVI", "dNBR", "dB12")
    """
    # NDVI: (B8 - B4) / (B8 + B4)
    ndvi_pre  = _nd(pre,  "B8", "B4", "NDVI_pre")
    ndvi_post = _nd(post, "B8", "B4", "NDVI_post")
    d_ndvi    = ndvi_post.subtract(ndvi_pre).rename("dNDVI")

    # NDMI: (B8 - B11) / (B8 + B11)
    ndmi_pre  = _nd(pre,  "B8", "B11", "NDMI_pre")
    ndmi_post = _nd(post, "B8", "B11", "NDMI_post")
    d_ndmi    = ndmi_post.subtract(ndmi_pre).rename("dNDMI")

    # NDRE: (B8 - B5) / (B8 + B5)
    ndre_pre  = _nd(pre,  "B8", "B5", "NDRE_pre")
    ndre_post = _nd(post, "B8", "B5", "NDRE_post")
    d_ndre    = ndre_post.subtract(ndre_pre).rename("dNDRE")

    # NBR: (B8 - B12) / (B8 + B12) — for labels only
    nbr_pre  = _nd(pre,  "B8", "B12", "NBR_pre")
    nbr_post = _nd(post, "B8", "B12", "NBR_post")
    d_nbr    = nbr_pre.subtract(nbr_post).rename("dNBR")  # pre - post convention

    # dB12: raw SWIR difference (post - pre)
    d_b12 = post.select("B12").subtract(pre.select("B12")).rename("dB12")

    return {
        "NDVI_pre": ndvi_pre, "NDVI_post": ndvi_post, "dNDVI": d_ndvi,
        "NDMI_pre": ndmi_pre, "NDMI_post": ndmi_post, "dNDMI": d_ndmi,
        "NDRE_pre": ndre_pre, "NDRE_post": ndre_post, "dNDRE": d_ndre,
        "NBR_pre":  nbr_pre,  "NBR_post":  nbr_post,  "dNBR":  d_nbr,
        "dB12":     d_b12,
    }


def delta_stack(indices: dict) -> ee.Image:
    """Stack all delta index bands into a single image (for quick export)."""
    return (
        indices["dNBR"]
        .addBands(indices["dNDVI"])
        .addBands(indices["dNDMI"])
        .addBands(indices["dNDRE"])
        .addBands(indices["dB12"])
        .toFloat()  # cast all bands to Float32 for consistent export
    )