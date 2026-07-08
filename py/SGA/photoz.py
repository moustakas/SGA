"""
SGA.photoz
==========

Photometric-redshift estimation for the SGA-2025 sample using a random
forest regressor, trained on the spectroscopic subsample (``Z_IVAR>0``,
roughly two-thirds of ``GROUP_PRIMARY`` galaxies).

Typical workflow (see ``bin/SGA2025-photoz`` for the full pipeline):

    from SGA.SGA import read_sga_sample
    from SGA.photoz import (select_spec_subsample, build_features,
                            train_and_evaluate, fit_final_model, nmad,
                            outlier_fraction)

    sample, _ = read_sga_sample()

    X, feature_names, row_mask = build_features(sample)
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
from SGA.logger import log

# Photometric bands used to build colors, in the order used to form
# consecutive colors (g-r, r-z, z-W1, W1-W2), matching the reference
# random_forest_photo-z_example.ipynb.
FEATURE_BANDS = ['G', 'R', 'Z', 'W1', 'W2']

MAG_MAX = 30.   # magnitudes fainter than this are treated as non-detections
MAG_FILL = 30.  # fill value substituted for non-detections/bad flux

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


def build_features(sample, bands=FEATURE_BANDS, mag_max=MAG_MAX):
    """Assemble the random-forest feature matrix from the base SGA2025
    columns (``FLUX_{band}``, ``D26``, ``BA``, ``EBV``) -- no
    ``ELLIPSEPHOT`` join needed.

    Parameters
    ----------
    sample : :class:`~astropy.table.Table`
        Table with ``FLUX_{band}`` for each of ``bands``, plus ``D26``,
        ``BA``, ``EBV`` (e.g. from :func:`SGA.SGA.read_sga_sample`).
    bands : :class:`list` of :class:`str`
        Bands to form consecutive colors from (default g,r,z,W1,W2).
    mag_max : :class:`float`
        Passed to :func:`_flux_to_mag`.

    Returns
    -------
    X : :class:`~numpy.ndarray`, shape (nobj, nfeat)
        Feature matrix: colors, the reddest-of-the-first-pair magnitude
        (e.g. r), ``D26``, ``BA``, ``EBV``.
    feature_names : :class:`list` of :class:`str`
    row_mask : :class:`~numpy.ndarray` of :class:`bool`, shape (nobj,)
        True where all features are finite and non-degenerate (see notes
        below); callers should combine this with
        :func:`select_spec_subsample` before training.

    Notes
    -----
    Following the reference notebook, a color that is exactly zero
    (both bands hit ``mag_fill``, or the two bands are otherwise
    identical) is flagged as degenerate for the first three colors only
    (g-r, r-z, z-W1); a zero W1-W2 color is common for real sources near
    the unWISE detection limit and is kept.

    """
    mags = {b: _flux_to_mag(sample[f'FLUX_{b}'], mag_max=mag_max) for b in bands}

    colors = [mags[b1] - mags[b2] for b1, b2 in zip(bands[:-1], bands[1:])]
    color_names = [f'{b1.lower()}-{b2.lower()}' for b1, b2 in zip(bands[:-1], bands[1:])]

    d26 = np.asarray(sample['D26'], dtype=np.float64)
    ba = np.asarray(sample['BA'], dtype=np.float64)
    ebv = np.asarray(sample['EBV'], dtype=np.float64)

    feature_names = color_names + [bands[1].lower(), 'D26', 'BA', 'EBV']
    X = np.column_stack(colors + [mags[bands[1]], d26, ba, ebv])

    row_mask = np.all(np.isfinite(X), axis=1) & (d26 > 0) & (ba > 0)
    for color in colors[:3]:
        row_mask &= (color != 0)

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
