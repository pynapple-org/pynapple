#!/usr/bin/env python

"""
Various io functions

@author: Guillaume Viejo
"""
import os
from .neurosuite import NeuroSuite
from .loader import BaseLoader


def load_session(path=None, session_type=None):
    """
    General Loader for Neurosuite, Phy or default session.

    Parameters
    ----------
    path : str, optional
        The path to load the data
    session_type : str, optional
        For the moment, pynapple support only 'neurosuite'.

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
    # elif session_type == 'phy':
    #   return NeuroSuite(path)
    else:
        return BaseLoader(path)
