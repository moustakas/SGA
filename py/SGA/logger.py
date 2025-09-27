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

"""
import sys, logging

LOGFMT = logging.Formatter(
    '%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S')


import sys, logging
from SGA.logger import LOGFMT  # your formatter


def hook_legacypipe_and_root(fh, debug=False):
    """
    Attach `fh` to legacypipe loggers AND the root logger.
    Ensures runbrick's basicConfig(stream=stdout) can't add a console handler.

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
    """Detach `fh` from legacypipe and root; optionally restore a root console handler."""
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
    """Create a logging object unique to the SGA.  Configure it to
    send its log messages to stdout and to format them identically to
    the DESIUtil defaults.

    Note that every call to this function returns the *same*
    log object, so will reset its properties including log level.
    Hence, it should really only be called once at the start
    of the program, as we do here, to create a singleton
    log object.

    Returns a log object with default level INFO.

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
