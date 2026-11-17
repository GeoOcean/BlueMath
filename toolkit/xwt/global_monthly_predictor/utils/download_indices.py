"""
Download, parse and store the monthly climate indices used in the AWT analysis.

Raw files are kept in `p_raw` (one per index) and all indices are stored together
in a single NetCDF (variable per index, monthly time, NaN for missing values).
"""

import html
import os
import os.path as op
import re
import shutil
import urllib.request
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


# url     : source file
# format  : 'wide' (year + 12 months), 'long' (year, month, value) or
#           'cpc_fixed' (CPC fixed-width table, section given by `section`),
#           'eli_html' (ggweather html table, Jul-Jun seasons) or
#           'roni_html' (CPC html table, 3-month running means DJF..NDJ)
# missing : missing-value flags
# col     : value column for 'long' files (default 2)
# skip    : leading lines to ignore (PSL files start with "first_year last_year")
SOURCES = {
    "NAO": dict(url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table",
                format="wide", missing=[-99.9, -99.99]),
    "AO":  dict(url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table",
                format="wide", missing=[-99.9, -99.99]),
    "AMO": dict(url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.amo.dat",
                format="long", missing=[-99.9, -99.99, 99.99]),
    "SCA": dict(url="https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/scand_index.tim",
                format="long", missing=[-99.9, -99.99]),
    "OLR": dict(url="https://www.cpc.ncep.noaa.gov/data/indices/olr",
                format="cpc_fixed", section="STANDARDIZED", missing=[-999.9]),
    "N34": dict(url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.el_nino.sst.dat",
                format="long", col=4, missing=[-99.99, 99.99]),
    "PDO": dict(url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
                format="wide", missing=[99.99, -99.99]),
    "SOI": dict(url="https://www.cpc.ncep.noaa.gov/data/indices/soi",
                format="cpc_fixed", section="STANDARDIZED", missing=[-999.9]),
    "PNA": dict(url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table",
                format="wide", missing=[-99.9, -99.99]),
    "SAM": dict(url="https://www.bas.ac.uk/met/gjma/sam.txt",
                format="wide", missing=[-99.9, -99.99]),
    "ELI": dict(url="https://ggweather.com/enso/eli.htm",
                format="eli_html", missing=[]),
    "RONI": dict(url="https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/",
                 format="roni_html", missing=[]),
}

_NUM = r"-?\d+\.?\d*"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_raw(name, p_raw, timeout=60):
    """
    Download the raw file of an index to `p_raw/<name>.txt`.
    If the download fails, the existing raw file (if any) is kept.
    Returns True if the file was updated.
    """
    os.makedirs(p_raw, exist_ok=True)
    p_file = op.join(p_raw, f"{name}.txt")
    req = urllib.request.Request(SOURCES[name]["url"], headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode("utf-8", errors="replace")
        if not text.strip():
            raise ValueError("empty response (the server may block automatic downloads)")
        parse_index(name, text)  # check it is a valid file before overwriting
    except Exception as e:
        status = "keeping previous raw file" if op.isfile(p_file) else "NO raw file available"
        warnings.warn(f"{name}: download failed ({e}); {status}")
        return False
    with open(p_file, "w") as f:
        f.write(text)
    return True


def seed_raw_from_folder(p_old, p_raw, files):
    """Copy old raw files ({name: filename}) to `p_raw` when no raw file exists yet."""
    os.makedirs(p_raw, exist_ok=True)
    for name, fname in files.items():
        dst = op.join(p_raw, f"{name}.txt")
        if not op.isfile(dst) and op.isfile(op.join(p_old, fname)):
            shutil.copy(op.join(p_old, fname), dst)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_wide(lines):
    rows = []
    for line in lines:
        tok = re.findall(_NUM, line)
        if len(tok) >= 2 and re.match(r"^\s*\d{4}\s", line) and "." in line:
            vals = [float(v) for v in tok[1:13]]
            rows.append([int(tok[0])] + vals + [np.nan] * (12 - len(vals)))
    data = np.array(rows)
    years = np.repeat(data[:, 0], 12)
    months = np.tile(np.arange(1, 13), len(data))
    return years, months, data[:, 1:].reshape(-1)


def _parse_long(lines, col=2):
    rows = [re.findall(_NUM, l) for l in lines if re.match(r"^\s*\d{4}\s+\d{1,2}\s+" + _NUM, l)]
    data = np.array([[float(r[0]), float(r[1]), float(r[col])] for r in rows])
    return data[:, 0], data[:, 1], data[:, 2]


def _parse_cpc_fixed(lines, section):
    """CPC tables: 4-char year followed by 12 fields of 6 chars (values may touch)."""
    i0 = next(i for i, l in enumerate(lines) if section in l)
    rows, started = [], False
    for line in lines[i0 + 1:]:
        if re.match(r"^\d{4}", line):
            started = True
            vals = [line[4 + 6 * k: 10 + 6 * k].strip() for k in range(12)]
            rows.append([int(line[:4])] + [float(v) if v else np.nan for v in vals])
        elif started and line.strip():
            break
    data = np.array(rows)
    years = np.repeat(data[:, 0], 12)
    months = np.tile(np.arange(1, 13), len(data))
    return years, months, data[:, 1:].reshape(-1)


def _html_rows(text):
    """Cells (plain text, empty ones removed) of every row of the html tables."""
    text = re.sub(r"(?s)<style.*?</style>|<script.*?</script>|<!--.*?-->", "", text)
    rows = []
    for tr in re.findall(r"(?is)<tr.*?</tr>", text):
        cells = re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", tr)
        cells = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", c))).strip() for c in cells]
        rows.append([c for c in cells if c])
    return rows


def _parse_eli_html(text):
    """ELI (ggweather): one row per season, Jul-Dec of the first year and Jan-Jun of the second."""
    years, months, values = [], [], []
    for row in _html_rows(text):
        season = [c for c in row if re.fullmatch(r"\d{4}-\d{4}", c)]
        if not season:
            continue
        y0 = int(season[0][:4])
        vals = [float(c) for c in row[row.index(season[0]) + 1:] if re.fullmatch(_NUM, c)][:12]
        for k, v in enumerate(vals):
            m = (6 + k) % 12 + 1  # Jul, Aug, ..., Jun
            years.append(y0 + (m < 7))
            months.append(m)
            values.append(v)
    return np.array(years), np.array(months), np.array(values)


def _parse_roni_html(text):
    """RONI (CPC): one row per year, 3-month running means (DJF..NDJ) assigned to the central month."""
    years, months, values = [], [], []
    for row in _html_rows(text):
        if not row or not re.fullmatch(r"\d{4}", row[0]):
            continue
        for k, c in enumerate(row[1:13]):
            if re.fullmatch(_NUM, c):
                years.append(int(row[0]))
                months.append(k + 1)
                values.append(float(c))
    return np.array(years), np.array(months), np.array(values)


def parse_index(name, text):
    """Parse the raw text of an index into a pandas Series (monthly, NaN = missing)."""
    cfg = SOURCES[name]
    lines = text.splitlines()[cfg.get("skip", 0):]

    if cfg["format"] == "wide":
        years, months, values = _parse_wide(lines)
    elif cfg["format"] == "long":
        years, months, values = _parse_long(lines, cfg.get("col", 2))
    elif cfg["format"] == "eli_html":
        years, months, values = _parse_eli_html(text)
    elif cfg["format"] == "roni_html":
        years, months, values = _parse_roni_html(text)
    else:
        years, months, values = _parse_cpc_fixed(lines, cfg["section"])

    for m in cfg["missing"]:
        values = np.where(np.isclose(values, m), np.nan, values)

    time = pd.to_datetime({"year": years.astype(int), "month": months.astype(int), "day": 1})
    s = pd.Series(values, index=time.values, name=name)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    if s.isna().all():
        raise ValueError(f"{name}: no valid values parsed")
    return s.loc[s.first_valid_index():s.last_valid_index()]


# ---------------------------------------------------------------------------
# Update all
# ---------------------------------------------------------------------------

def update_indices(p_raw, p_out, names=None):
    """Download (when possible) and parse every index; save them in one NetCDF."""
    names = names or list(SOURCES)
    series, updated = {}, {}
    for name in names:
        updated[name] = download_raw(name, p_raw)
        p_file = op.join(p_raw, f"{name}.txt")
        if op.isfile(p_file):
            with open(p_file) as f:
                series[name] = parse_index(name, f.read())

    df = pd.DataFrame(series)
    df.index.name = "time"
    ds = xr.Dataset.from_dataframe(df)
    for name in series:
        ds[name].attrs["source"] = SOURCES[name]["url"]
        ds[name].attrs["downloaded"] = str(updated[name])
    ds.attrs["created"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    ds.to_netcdf(p_out)

    summary = pd.DataFrame({
        "start": [series[n].index[0].strftime("%Y-%m") for n in series],
        "end": [series[n].index[-1].strftime("%Y-%m") for n in series],
        "updated": [updated[n] for n in series],
    }, index=list(series))
    return ds, summary
