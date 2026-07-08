"""
SGA.photoz
==========

Photometric-redshift estimation for the SGA-2025 sample using a random
forest regressor, trained on the spectroscopic subsample (``Z_IVAR>0``,
roughly two-thirds of the sample). Runs on all group members (not just
``GROUP_PRIMARY``) -- SGA groups are angular associations, not physical
ones, so every member galaxy needs its own redshift.

Typical workflow (see ``bin/SGA2025-photoz`` for the full pipeline):

    from SGA.SGA import read_sga_sample
    from SGA.photoz import (select_spec_subsample, build_features,
                            train_and_evaluate, fit_final_model, nmad,
                            outlier_fraction)

    _, sample = read_sga_sample()  # all group members, not just GROUP_PRIMARY
    _, tractor = read_sga_sample(tractor=True)  # row-matched to sample

    X, feature_names, row_mask = build_features(sample, tractor=tractor)
    train_mask = select_spec_subsample(sample) & row_mask

    z_spec = sample['Z'][train_mask]
    z_phot_cv, diagnostics = train_and_evaluate(X[train_mask], z_spec)
    print(diagnostics)  # {'nmad': ..., 'outlier_frac': ...}

    final_model = fit_final_model(X[train_mask], z_spec)
    z_phot_all = final_model.predict(X)

"""
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from SGA.SGA import Z_FLAG
from SGA.coadds import REGIONBITS
from SGA.logger import log

# Photometric bands used to build consecutive colors (NUV-g, g-r, r-i,
# i-z, z-W1, W1-W2). Includes both optical neighbors of r (g and i) so
# that objects missing FLUX_R (~2% of the sample) still get a usable
# adjacent color instead of losing the whole g/r/i/z chain. NUV is
# included (76% coverage) since it measurably improves CV performance
# (NMAD 0.01137->0.01108, checked against the SGA2025-v1.0 sample); FUV
# was tested too but adds nothing on top of NUV (62% coverage, NMAD
# 0.01108->0.01107) so it's left out.
FEATURE_BANDS = ['NUV', 'G', 'R', 'I', 'Z', 'W1', 'W2']

# Priority order for the single-band "apparent brightness"/surface-brightness
# magnitude: r is usually best-measured, but when it's missing i covers far
# more objects than g (85% vs. 47%, checked against the SGA2025-v1.0 sample).
BRIGHTNESS_BANDS = ['R', 'I', 'G']

MAG_MAX = 30.   # magnitudes fainter than this are treated as non-detections
MAG_FILL = 30.  # fill value substituted for non-detections/bad flux

# Tractor TYPE values for which SERSIC is not a meaningful concentration
# index: PSF (point source, no profile fit) and '' (no Tractor match at
# all). EXP/REX/DEV are kept as informative -- their SERSIC is a fixed
# Tractor convention (1.0/1.0/4.0), not a real fit, but it's still a
# legitimate coarse profile-shape value, unlike PSF's SERSIC=0.
SERSIC_MISSING_TYPES = ('PSF', '')
SERSIC_FILL = -1.  # fill value substituted for SERSIC_MISSING_TYPES

# Z_FLAG bits that mark an adopted redshift as untrustworthy for training:
# any cross-source disagreement, or a sole NED redshift that NED itself
# flagged as unresolved/missing (see SGA.SGA.Z_FLAG for the full bitmask).
DEFAULT_BAD_ZFLAG = Z_FLAG['DISCREPANCY'] | Z_FLAG['NED_UNRESOLVED'] | Z_FLAG['NED_MISSING']


def select_spec_subsample(sample, bad_zflag=DEFAULT_BAD_ZFLAG):
    """Boolean mask selecting galaxies with a trustworthy spectroscopic
    redshift, suitable for training/cross-validation.

    Parameters
    ----------
    sample : :class:`~astropy.table.Table`
        Table with ``Z_IVAR`` and ``Z_FLAG`` columns (e.g. from
        :func:`SGA.SGA.read_sga_sample`).
    bad_zflag : :class:`int`
        Bitwise-OR of :data:`SGA.SGA.Z_FLAG` bits that disqualify a row
        even though ``Z_IVAR>0``. Pass 0 to disable this filter.

    Returns
    -------
    :class:`~numpy.ndarray` of :class:`bool`

    """
    good = np.asarray(sample['Z_IVAR']) > 0
    if bad_zflag:
        good &= (np.asarray(sample['Z_FLAG']).astype(np.int64) & bad_zflag) == 0
    return good


def _flux_to_mag(flux, mag_max=MAG_MAX, mag_fill=MAG_FILL):
    """Convert nanomaggies to AB mag; non-positive flux or mag>mag_max is
    replaced by mag_fill (mirrors the sentinel handling in the reference
    random_forest_photo-z_example.ipynb)."""
    flux = np.asarray(flux, dtype=np.float64)
    mag = np.full(flux.shape, mag_fill, dtype=np.float64)
    good = flux > 0
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    mag[mag > mag_max] = mag_fill
    return mag


def _censor_north_z(sample, flux_z):
    """Zero out ``FLUX_Z`` for dr11-north-only rows (``REGION`` exactly
    equal to ``REGIONBITS['dr11-north']``, i.e. unambiguously north-sourced
    photometry -- overlap rows with both region bits set are left alone,
    since the actual adopted photometry there may already be dr11-south).

    dr11-north z-band (BASS+MzLS) is systematically redder relative to
    south with increasing D26 -- median z-W1 drifts from about -0.15 mag
    at D26<0.5 arcmin to +0.15 mag at D26>10 arcmin, versus a roughly flat
    -0.26 to -0.85 mag in dr11-south over the same range (checked against
    the SGA2025-v1.0 sample) -- so it's censored outright rather than used
    with a size cut.

    """
    flux_z = np.array(flux_z, dtype=np.float64, copy=True)
    region = np.asarray(sample['REGION'])
    north_only = region == REGIONBITS['dr11-north']
    flux_z[north_only] = 0.
    return flux_z


def build_features(sample, tractor=None, bands=FEATURE_BANDS,
                   brightness_bands=BRIGHTNESS_BANDS, mag_max=MAG_MAX,
                   censor_north_z=True):
    """Assemble the random-forest feature matrix from the base SGA2025
    columns (``FLUX_{band}``, ``MW_TRANSMISSION_{band}``, ``D26``, ``BA``)
    -- no ``ELLIPSEPHOT`` join needed.

    Parameters
    ----------
    sample : :class:`~astropy.table.Table`
        Table with ``FLUX_{band}`` and ``MW_TRANSMISSION_{band}`` for each
        of ``bands``, plus ``REGION``, ``D26``, ``BA`` (e.g. from
        :func:`SGA.SGA.read_sga_sample`).
    tractor : :class:`~astropy.table.Table`, optional
        Row-matched TRACTOR table (e.g. from
        ``SGA.SGA.read_sga_sample(tractor=True)``), used to add a
        ``SERSIC`` feature. If None (default), no ``SERSIC``/
        ``detected_sersic`` features are added.
    bands : :class:`list` of :class:`str`
        Bands to form consecutive colors from (default NUV,g,r,i,z,W1,W2).
    brightness_bands : :class:`list` of :class:`str`
        Priority-ordered fallback bands for the single apparent-magnitude
        and surface-brightness features (default r, then i, then g).
    mag_max : :class:`float`
        Passed to :func:`_flux_to_mag`.
    censor_north_z : :class:`bool`
        If True (default), zero out ``FLUX_Z`` for dr11-north-only rows
        before computing anything (see :func:`_censor_north_z`).

    Returns
    -------
    X : :class:`~numpy.ndarray`, shape (nobj, nfeat)
        Feature matrix: consecutive colors, a fallback brightness
        magnitude, ``D26``, ``BA``, a surface-brightness proxy, a
        detected/not-detected flag per band, and (when ``tractor`` is
        given) ``SERSIC``/``detected_sersic``. All fluxes/magnitudes are
        Milky Way extinction-corrected via ``MW_TRANSMISSION_{band}``;
        there is no separate ``EBV`` feature, since the dereddened
        photometry already absorbs its effect.
    feature_names : :class:`list` of :class:`str`
    row_mask : :class:`~numpy.ndarray` of :class:`bool`, shape (nobj,)
        True where all features are finite and ``D26``/``BA`` are
        positive; callers should combine this with
        :func:`select_spec_subsample` before training.

    Notes
    -----
    A color or the brightness magnitude is only as good as the bands
    behind it; rather than dropping a whole row when one band is missing
    (the previous behavior), a missing band's contribution to any color is
    zeroed out and its own ``detected_{band}`` flag is set to 0, so the
    model can learn to discount it instead of losing the row entirely.
    Checked against the full SGA2025-v1.0 sample (470,625 objects, all
    group members): every single row now gets a usable feature vector --
    ``row_mask`` is only ever False for ``D26<=0``/``BA<=0``, which does
    not occur in practice.

    ``SERSIC`` (Tractor Sersic index) uses the same sentinel+flag pattern:
    it's a real fit for ``TYPE='SER'`` and a fixed-but-informative Tractor
    convention for ``EXP``/``REX``/``DEV`` (1.0/1.0/4.0), so all four are
    kept as-is; only ``PSF`` (point source, no profile) and unmatched rows
    (``TYPE=''``) get :data:`SERSIC_FILL` and ``detected_sersic=0``.
    Checked against the SGA2025-v1.0 sample: only 155/470,625 objects
    (0.03%) fall in that missing bucket, and adding SERSIC improves 5-fold
    CV NMAD from 0.01091 to 0.01053 (outlier fraction 0.12%->0.11%), with
    SERSIC ranking 7th of 19 by feature importance.

    """
    flux = {b: np.asarray(sample[f'FLUX_{b}'], dtype=np.float64) for b in bands}
    if censor_north_z and 'Z' in bands:
        flux['Z'] = _censor_north_z(sample, flux['Z'])

    # Detection is judged on the raw (observed) flux; dereddening only
    # rescales positive values, so it can't flip a non-detection to a
    # detection or vice versa.
    detected = {b: flux[b] > 0 for b in bands}

    mw_transmission = {b: np.asarray(sample[f'MW_TRANSMISSION_{b}'], dtype=np.float64)
                       for b in bands}
    dered_flux = {b: flux[b] / mw_transmission[b] for b in bands}
    mags = {b: _flux_to_mag(dered_flux[b], mag_max=mag_max) for b in bands}

    color_names, colors = [], []
    for b1, b2 in zip(bands[:-1], bands[1:]):
        valid = detected[b1] & detected[b2]
        color = np.where(valid, mags[b1] - mags[b2], 0.)
        colors.append(color)
        color_names.append(f'{b1.lower()}-{b2.lower()}')

    n = len(sample)
    brightness_mag = np.full(n, MAG_FILL, dtype=np.float64)
    brightness_detected = np.zeros(n, dtype=bool)
    for b in brightness_bands:
        use = detected[b] & ~brightness_detected
        brightness_mag[use] = mags[b][use]
        brightness_detected |= detected[b]

    d26 = np.asarray(sample['D26'], dtype=np.float64)
    ba = np.asarray(sample['BA'], dtype=np.float64)

    # Average surface brightness within the D26 isophote (mag/arcsec^2),
    # using whichever band fed the brightness magnitude above.
    area_arcsec2 = (np.pi / 4.) * ba * (d26 * 60.)**2
    good_area = brightness_detected & (area_arcsec2 > 0)
    sb = np.full(n, MAG_FILL, dtype=np.float64)
    sb[good_area] = brightness_mag[good_area] + 2.5 * np.log10(area_arcsec2[good_area])

    detected_names = [f'detected_{b.lower()}' for b in bands]
    detected_arrs = [detected[b].astype(np.float64) for b in bands]

    feature_names = (color_names + ['mag', 'D26', 'BA', 'SB'] + detected_names)
    feature_arrs = colors + [brightness_mag, d26, ba, sb] + detected_arrs

    if tractor is not None:
        typ = np.array([t.strip() for t in np.asarray(tractor['TYPE']).astype(str)])
        sersic_missing = np.isin(typ, SERSIC_MISSING_TYPES)
        sersic = np.where(sersic_missing, SERSIC_FILL,
                          np.asarray(tractor['SERSIC'], dtype=np.float64))
        detected_sersic = (~sersic_missing).astype(np.float64)
        feature_names += ['SERSIC', 'detected_sersic']
        feature_arrs += [sersic, detected_sersic]

    X = np.column_stack(feature_arrs)

    row_mask = np.all(np.isfinite(X), axis=1) & (d26 > 0) & (ba > 0)

    return X, feature_names, row_mask


def train_and_evaluate(X, y, n_folds=5, random_state=1456, outlier_threshold=0.1,
                       **rf_kwargs):
    """K-fold cross-validated ``RandomForestRegressor`` training and
    diagnostics (NMAD, outlier fraction) on out-of-fold predictions.

    Parameters
    ----------
    X : :class:`~numpy.ndarray`, shape (nobj, nfeat)
    y : :class:`~numpy.ndarray`, shape (nobj,)
        Spectroscopic redshift (``Z``).
    n_folds : :class:`int`
    random_state : :class:`int`
    outlier_threshold : :class:`float`
        Passed to :func:`outlier_fraction`.
    rf_kwargs : passed to :class:`~sklearn.ensemble.RandomForestRegressor`
        (defaults: ``n_estimators=100``, ``n_jobs=-1``).

    Returns
    -------
    z_phot_cv : :class:`~numpy.ndarray`, shape (nobj,)
        Out-of-fold predictions, aligned with ``X``/``y``.
    diagnostics : :class:`dict`
        ``{'nmad': float, 'outlier_frac': float}``.

    """
    rf_kwargs.setdefault('n_estimators', 100)
    rf_kwargs.setdefault('n_jobs', -1)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    z_phot_cv = np.zeros(len(y))
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        log.info(f'Training fold {fold}/{n_folds} ({len(train_idx):,d} objects)...')
        model = RandomForestRegressor(random_state=random_state, **rf_kwargs)
        model.fit(X[train_idx], y[train_idx])
        z_phot_cv[test_idx] = model.predict(X[test_idx])

    diagnostics = {
        'nmad': nmad(z_phot_cv, y),
        'outlier_frac': outlier_fraction(z_phot_cv, y, threshold=outlier_threshold),
    }
    return z_phot_cv, diagnostics


def fit_final_model(X, y, random_state=1456, **rf_kwargs):
    """Fit a single ``RandomForestRegressor`` on the full training set
    (e.g. for production predictions, after :func:`train_and_evaluate`
    has been used to validate the approach).

    Parameters
    ----------
    X, y : see :func:`train_and_evaluate`.
    rf_kwargs : see :func:`train_and_evaluate`.

    Returns
    -------
    :class:`~sklearn.ensemble.RandomForestRegressor`

    """
    rf_kwargs.setdefault('n_estimators', 100)
    rf_kwargs.setdefault('n_jobs', -1)
    model = RandomForestRegressor(random_state=random_state, **rf_kwargs)
    model.fit(X, y)
    return model


def predict_with_uncertainty(model, X):
    """Point prediction and a first-cut uncertainty estimate from a
    fitted :class:`~sklearn.ensemble.RandomForestRegressor`, using the
    spread of predictions across its individual trees.

    Parameters
    ----------
    model : :class:`~sklearn.ensemble.RandomForestRegressor`
        Fitted model (e.g. from :func:`fit_final_model`).
    X : :class:`~numpy.ndarray`, shape (nobj, nfeat)

    Returns
    -------
    z_phot : :class:`~numpy.ndarray`, shape (nobj,)
        Mean prediction across trees (matches ``model.predict(X)``).
    z_phot_err : :class:`~numpy.ndarray`, shape (nobj,)
        Standard deviation of the per-tree predictions.

    """
    tree_preds = np.stack([tree.predict(X) for tree in model.estimators_], axis=0)
    return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def nmad(z_phot, z_spec):
    """Normalized median absolute deviation of ``dz/(1+z_spec)``, the
    standard robust photo-z scatter metric."""
    z_phot, z_spec = np.asarray(z_phot), np.asarray(z_spec)
    return 1.48 * np.median(np.abs((z_phot - z_spec) / (1. + z_spec)))


def outlier_fraction(z_phot, z_spec, threshold=0.1):
    """Fraction of objects with ``|dz|/(1+z_spec) > threshold``."""
    z_phot, z_spec = np.asarray(z_phot), np.asarray(z_spec)
    dz = np.abs((z_phot - z_spec) / (1. + z_spec))
    return np.count_nonzero(dz > threshold) / len(z_spec)
