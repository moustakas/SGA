"""
==========
SGA.logger
==========

Unique logger object, distinct from the one used by the DESI libraries,
allocated at startup and used throughout code base.  We need to create
our own logger to avoid the DESI libraries accidentally changing the
log level out from under us.

Having just one logger allows us to initialize its level on startup and have
those changes propagate everywhere.

This needs to be in its own file to prevent circular imports with other code.

Notes
-----
The ``from SGA.logger import LOGFMT`` statement below is a self-import
(this *is* ``SGA.logger``) -- it works because ``LOGFMT`` is already
defined earlier in this same module by the time this line executes
(Python registers a module in ``sys.modules`` before finishing its
execution, so a self-import can see names already bound above it), but
it's redundant (``LOGFMT`` is already in scope without it) and fragile
to reordering; likely a copy-paste artifact from code that originated
in a different module.

"""
import sys, logging

LOGFMT = logging.Formatter(
    '%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S')


import sys, logging
from SGA.logger import LOGFMT  # your formatter


def hook_legacypipe_and_root(fh, debug=False):
    """Attach a file handler to the ``legacypipe``/``legacypipe.runbrick``
    loggers and the root logger, so their output is captured to the
    same log file as SGA's own logger.

    Adds ``fh`` to the root logger *before* legacypipe's ``runbrick``
    has a chance to call ``logging.basicConfig(stream=stdout)`` (which
    is a no-op once the root logger already has a handler), preventing
    a duplicate console stream. Also routes Python's ``warnings``
    module output to ``fh``, and stops the ``legacypipe`` loggers from
    propagating to the root logger (to avoid double-logging once both
    have ``fh`` attached). Unless ``debug``, removes any pre-existing
    console (``StreamHandler``) handlers from both the root and
    legacypipe loggers, so their output only goes to the file.

    Parameters
    ----------
    fh : :class:`logging.FileHandler`
        File handler to attach.
    debug : :class:`bool`
        If True, leave any existing console handlers in place (so
        legacypipe/root output still also appears on the console).

    Returns
    -------
    None

    """
    # 1) Root: add file handler first so basicConfig is a no-op
    root = logging.getLogger()
    if fh not in root.handlers:
        root.addHandler(fh)
    root.setLevel(logging.DEBUG)

    # If not debugging, remove any existing root console handlers to stdout/stderr
    if not debug:
        for h in list(root.handlers):
            if (type(h) is logging.StreamHandler) and h is not fh:
                root.removeHandler(h)

    # Route Python warnings to the same file
    logging.captureWarnings(True)
    pyw = logging.getLogger('py.warnings')
    if fh not in pyw.handlers:
        pyw.addHandler(fh)

    # 2) legacypipe loggers: add file handler, stop propagation
    for name in ('legacypipe', 'legacypipe.runbrick'):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        if fh not in lg.handlers:
            lg.addHandler(fh)
        if not debug:
            # drop any console stream handlers they might have added elsewhere
            for h in list(lg.handlers):
                if (type(h) is logging.StreamHandler) and h is not fh:
                    lg.removeHandler(h)


def unhook_legacypipe_and_root(fh, restore_console=True, debug=False):
    """Detach a file handler from the ``legacypipe``/``py.warnings``/root
    loggers (undoing :func:`hook_legacypipe_and_root`), optionally
    restoring a plain console handler on the root logger.

    Parameters
    ----------
    fh : :class:`logging.FileHandler`
        File handler to detach.
    restore_console : :class:`bool`
        If True and the root logger has no remaining console
        (``StreamHandler``) handler, add a fresh one (level DEBUG if
        ``debug``, else INFO), so non-SGA log output becomes visible on
        the console again.
    debug : :class:`bool`
        Level to use for the restored console handler, if any (see
        ``restore_console``).

    Returns
    -------
    None

    """
    pyw = logging.getLogger('py.warnings')
    if fh in pyw.handlers:
        pyw.removeHandler(fh)

    for name in ('legacypipe', 'legacypipe.runbrick'):
        lg = logging.getLogger(name)
        if fh in lg.handlers:
            lg.removeHandler(fh)

    root = logging.getLogger()
    if fh in root.handlers:
        root.removeHandler(fh)

    if restore_console:
        # Put a clean console handler on root (so non-SGA logs show up again)
        has_console = any(type(h) is logging.StreamHandler for h in root.handlers)
        if not has_console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG if debug else logging.INFO)
            ch.setFormatter(LOGFMT)
            root.addHandler(ch)


def setup_logging(logfile, debug=False):
    """(Re)configure the SGA singleton logger (:data:`log`) to write to
    a file instead of (or in addition to) the console, and capture
    ``legacypipe``/Python-``warnings`` output into the same file.

    Removes any existing ``FileHandler``s from :data:`log` first. If
    ``debug``, leaves console (``StreamHandler``) output in place, bumps
    it to DEBUG level, and returns without creating a file handler
    (console-only debug mode). Otherwise, creates a new ``FileHandler``
    on ``logfile`` (INFO level), attaches it to :data:`log`, hooks it
    into the legacypipe/root loggers (:func:`hook_legacypipe_and_root`)
    so ``runbrick`` output lands in the same file, removes
    :data:`log`'s own console handlers (file-only mode), and routes
    Python ``warnings`` to the file too.

    Parameters
    ----------
    logfile : :class:`str`
        Path to the log file to create (opened in write/truncate mode).
    debug : :class:`bool`
        If True, keep logging to the console at DEBUG level instead of
        setting up file logging.

    Returns
    -------
    :class:`logging.FileHandler` or None
        The created file handler, or None if ``debug`` (no file handler
        created).

    """
    # remove any existing FileHandlers
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            log.removeHandler(h)
            h.close()

    if debug:
        # console at DEBUG
        for h in log.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(logging.DEBUG)
        return None

    # file-only
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO) # writes INFO, WARNING, ERROR, CRITICAL
    fh.setFormatter(LOGFMT)
    log.addHandler(fh)

    # prevent runbrick’s basicConfig from installing stdout handler; capture its logs
    hook_legacypipe_and_root(fh, debug=debug)

    if not debug:
        for h in list(log.handlers):
            if type(h) is logging.StreamHandler and getattr(h, "stream", None) in (sys.stdout, sys.stderr):
                log.removeHandler(h)

    # route Python warnings -> logging file
    logging.captureWarnings(True)
    logging.getLogger('py.warnings').addHandler(fh)

    return fh


def getSgaLogger():
    """Create (or retrieve) the singleton logger unique to SGA,
    configured to send its log messages to stdout and formatted
    identically to the DESIUtil defaults.

    Every call to this function returns the *same* underlying
    ``logging.Logger`` object (Python's ``logging.getLogger('SGA')``
    is itself a singleton by name), and resets its level and
    propagation setting each time. It should really only be called
    once at the start of the program, as is done here at module import
    time to create the module-level :data:`log` singleton.

    Notes
    -----
    Despite this docstring's earlier claim of "default level INFO",
    the logger itself is actually set to DEBUG (``log.setLevel(logging.DEBUG)``,
    "allow everything; handlers decide what to write") -- the effective
    filtering happens at the handler level (e.g. via
    :func:`setup_logging`'s file handler, set to INFO), not on the
    logger itself.

    Returns
    -------
    :class:`logging.Logger`
        The ``'SGA'`` logger, with a stdout ``StreamHandler`` attached
        (only added the first time, if none exists yet), level DEBUG,
        and ``propagate=False`` (to avoid duplicate messages via the
        root logger).

    """
    log = logging.getLogger('SGA')
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(LOGFMT)
        log.addHandler(ch)
    log.setLevel(logging.DEBUG) # allow everything; handlers decide what to write
    log.propagate = False  # don’t duplicate to root

    return log


log = getSgaLogger()
