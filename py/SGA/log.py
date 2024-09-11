# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
============
desiutil.log
============

DESI-specific utility functions that wrap the standard :mod:`logging`
module.

This module is intended to support three different logging use patterns:

1. Just get an easy-to-use, pre-configured logging object.
2. Easily change the log level temporarily within a function.  This is
   provided by a context manager.
3. Change the default log level on the command-line.  This can actually be
   accomplished in two ways: the command-line interpreter can call
   :func:`~desiutil.log.get_logger` with the appropriate level, or
   the environment variable :envvar:`DESI_LOGLEVEL` can be set.

In addition, it is possible to add timestamps and change the delimiter of
log messages as needed.  See the optional arguments to
:func:`~desiutil.log.get_logger`.

Examples
--------

Simplest possible use:

>>> from desiutil.log import log
>>> log.info('This is some information.')

This is exactly equivalent to:

>>> from desiutil.log import get_logger
>>> log = get_logger()
>>> log.info('This is some information.')

Temporarily change the log level with a context manager:

>>> from desiutil.log import get_logger, DesiLogContext, DEBUG
>>> log = get_logger()  # defaults to INFO
>>> log.info('This is some information.')
>>> log.debug("This won't be logged.")
>>> with DesiLogContext(log, DEBUG):
...     log.debug("This will be logged.")
>>> log.debug("This won't be logged.")

Create the logger with a different log level:

>>> from desiutil.log import get_logger, DEBUG
>>> if options.debug:
...     log = get_logger(DEBUG)
>>> else:
...     log = get_logger()

"""
import os
import sys
import logging
from warnings import warn


_desiutil_log_root = dict()
_good_levels = {'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL,
                logging.DEBUG: logging.DEBUG,
                logging.INFO: logging.INFO,
                logging.WARNING: logging.WARNING,
                logging.ERROR: logging.ERROR,
                logging.CRITICAL: logging.CRITICAL,
                }
_level_children = {logging.DEBUG: 'debug',
                   logging.INFO: 'info',
                   logging.WARNING: 'warning',
                   logging.ERROR: 'error',
                   logging.CRITICAL: 'critical',
                   }


# Just for convenience to avoid importing logging, we duplicate the logging levels
# Detailed information, typically of interest only when diagnosing problems.
DEBUG = logging.DEBUG
# Confirmation that things are working as expected.
INFO = logging.INFO
# An indication that something unexpected happened, or indicative of some problem
# in the near future (e.g. "disk space low"). The software is still working as expected.
WARNING = logging.WARNING
# Due to a more serious problem, the software has not been able to perform some function.
ERROR = logging.ERROR
# A serious error, indicating that the program itself may be unable to continue running.
CRITICAL = logging.CRITICAL

# see example of usage in test/test_log.py


class DesiLogWarning(UserWarning):
    """Warnings related to misconfiguration of the DESI logging object.
    """
    pass


class DesiLogContext(object):
    """Provides a context manager to temporarily change the log level of
    an existing logging object.

    Parameters
    ----------
    logger : :class:`logging.Logger`
        Logging object.
    level : :class:`int`, optional
        The logging level to set.  If it is not set, this whole class
        does nothing.
    """
    def __init__(self, logger, level=None):  # , handler=None, close=True):
        self.logger = logger
        self.level = level
        # self.handler = handler
        # self.close = close

    def __enter__(self):
        if self.level is None:
            warn("This context manager will not actually do anything!",
                 DesiLogWarning)
        else:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        # if self.handler:
        #     self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        # if self.handler:
        #     self.logger.removeHandler(self.handler)
        # if self.handler and self.close:
        #     self.handler.close()


def _configure_root_logger(timestamp=False, delimiter=':'):
    """Configure a root logger.

    Parameters
    ----------
    timestamp : :class:`bool`, optional
        If ``True``, add a timestamp to the log message.
    delimiter : :class:`str`, optional
        Use `delimiter` to separate fields in the log message (default ``:``).

    Returns
    -------
    :class:`str`
        The name of the root logger, suitable for input to :func:`logging.getLogger`.
    """
    root_name = "desiutil.log.dlm" + ''.join(map(str, map(ord, delimiter)))
    if timestamp:
        root_name += 'timestamp'
    if root_name not in _desiutil_log_root:
        ch = logging.StreamHandler(sys.stdout)
        fmtfields = ['%(levelname)s', '%(filename)s', '%(lineno)s', '%(funcName)s']
        if timestamp:
            fmtfields.append('%(asctime)s')
        fmtfields.append(' %(message)s')
        formatter = logging.Formatter(delimiter.join(fmtfields),
                                      datefmt='%Y-%m-%dT%H:%M:%S')
        ch.setFormatter(formatter)
        _desiutil_log_root[root_name] = logging.getLogger(root_name)
        _desiutil_log_root[root_name].addHandler(ch)
        _desiutil_log_root[root_name].setLevel(logging.INFO)
    return root_name


def get_logger(level=None, timestamp=False, delimiter=':'):
    """Returns a default DESI logger.

    Parameters
    ----------
    level : :class:`int` or :class:`str`, optional
        Set the logging level (default ``INFO``).
    timestamp : :class:`bool`, optional
        If ``True``, add a timestamp to the log message.
    delimiter : :class:`str`, optional
        Use `delimiter` to separate fields in the log messages (default ``:``).

    Returns
    -------
    :class:`logging.Logger`
        A logging object configured with the DESI defaults.

    Notes
    -----
    * If `level` is not ``None``, that sets the log level, overriding anything
      else.
    * If `level` is not set, and if the environment variable
      :envvar:`DESI_LOGLEVEL` exists and has value
      DEBUG, INFO, WARNING, ERROR or CRITICAL (upper or lower case),
      that is used to set the log level.
    * If :envvar:`DESI_LOGLEVEL` is not set and `level` is ``None``,
      the default level is set to INFO.
    """
    root_name = _configure_root_logger(timestamp=timestamp, delimiter=delimiter)
    if level is None:
        try:
            ul = os.environ["DESI_LOGLEVEL"].upper()
        except KeyError:
            ul = logging.INFO
    else:
        try:
            ul = level.upper()
        except AttributeError:
            # level should be an integer in this case.
            ul = level
    try:
        gl = _good_levels[ul]
    except KeyError:
        message = "Invalid level='{0}' ignored.  Setting INFO.".format(str(ul))
        warn(message, DesiLogWarning)
        gl = logging.INFO
    log = logging.getLogger(root_name + '.' + _level_children[gl])
    log.setLevel(gl)
    return log


log = get_logger()
