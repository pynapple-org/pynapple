#!/usr/bin/env python

"""
Various io functions

@author: Guillaume Viejo
"""
import os
from .neurosuite import NeuroSuite
from .phy import Phy
from .loader import BaseLoader


def load_session(path=None, session_type=None):
    """
    General Loader for Neurosuite, Phy or default session.

    Parameters
    ----------
    path : str, optional
        The path to load the data
    session_type : str, optional
        Can be 'neurosuite', 'phy' or None for default loader.

    Returns
    -------
    Session
        A class holding all the data from the session.

    """
    if path:
        if not os.path.isdir(path):
            raise RuntimeError("Path {} is not found.".format(path))

    if session_type == 'neurosuite':
        return NeuroSuite(path)
    elif session_type == 'phy':
      return Phy(path)
    else:
        return BaseLoader(path)
