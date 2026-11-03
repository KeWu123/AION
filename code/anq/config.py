#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for the AION run configuration.

Every experiment imports CANON from here.  The value is the tightest valid point
of the 108-configuration sweep in code/tune_cs.py (kappa0=0.01, growth=1.4,
nk=16, weight_pow=0.4), so the width studies, the cost study and the session
studies all refer to the *same* sequence.  (Earlier drafts used four different
configs across drivers; numbers from them were not comparable.)

The constant is defined in anq.bounds, next to CSConfig, so that classes there
can default to it without a circular import; this module re-exports it.
"""
from .bounds import CSConfig, CANON  # noqa: F401
