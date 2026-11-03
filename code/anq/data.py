#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPC-H data access (PostgreSQL) -> numpy, plus exact ground truth.

Connection settings are read from the environment so that no credential is baked into
the artifact.  ``AION_PG*`` variables take precedence over the standard libpq names
(``PGHOST``, ``PGPORT``, ``PGUSER``, ``PGPASSWORD``, ``PGDATABASE``).
"""
from __future__ import annotations
import os
import numpy as np

PG = dict(
    host=os.environ.get("AION_PGHOST", os.environ.get("PGHOST", "127.0.0.1")),
    port=int(os.environ.get("AION_PGPORT", os.environ.get("PGPORT", "5432"))),
    user=os.environ.get("AION_PGUSER", os.environ.get("PGUSER", "postgres")),
    password=os.environ.get("AION_PGPASSWORD", os.environ.get("PGPASSWORD", "")),
    dbname=os.environ.get("AION_PGDATABASE", os.environ.get("PGDATABASE", "postgres")),
)


def connect():
    import psycopg2
    return psycopg2.connect(**PG)


def load_columns(sql_cols: str, table: str, where: str = "") -> dict:
    """Load the named columns of table into numpy arrays."""
    con = connect()
    q = f"SELECT {sql_cols} FROM {table} " + (f"WHERE {where}" if where else "")
    with con.cursor() as cur:
        cur.execute(q)
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    con.close()
    out = {}
    for j, nm in enumerate(names):
        col = [r[j] for r in rows]
        try:
            out[nm] = np.asarray(col, dtype=np.float64)
        except (TypeError, ValueError):
            out[nm] = np.asarray(col, dtype=object)
    return out


def exact(sql: str):
    con = connect()
    with con.cursor() as cur:
        cur.execute(sql)
        r = cur.fetchall()
    con.close()
    return r


def build_lineitem_orders(limit: int | None = None) -> dict:
    """Materialise a lineitem x orders star join (foreign-key join on o_orderkey)."""
    cols = ("l_orderkey, l_quantity, l_extendedprice, l_discount, l_tax, "
            "l_returnflag, o_custkey, o_totalprice, o_orderdate")
    q = f"""SELECT {cols} FROM lineitem JOIN orders ON l_orderkey = o_orderkey"""
    con = connect()
    import numpy as _np
    with con.cursor() as cur:
        cur.execute(q + (f" LIMIT {limit}" if limit else ""))
        rows = cur.fetchall()
        names = [d.name for d in cur.description]
    con.close()
    out = {}
    arr = _np.array(rows, dtype=object)
    for j, nm in enumerate(names):
        c = arr[:, j]
        if nm in ("l_returnflag",):
            out[nm] = c.astype(str)
        else:
            out[nm] = c.astype(_np.float64)
    return out


if __name__ == "__main__":
    d = load_columns("l_quantity, l_extendedprice, l_discount", "lineitem")
    print({k: (v.shape, v.dtype) for k, v in d.items()})
    print(exact("select count(*), avg(l_quantity) from lineitem"))
