"""Direct downloads for Colab (no browser / login needed).

* CaloChallenge DS2: Zenodo record 6366271, files fetched with wget and md5-checked.
* Quark/gluon jets: CERNBox public share. The share is listed over the public
  WebDAV endpoint (remote.php/dav/public-files/<token>) and each file is fetched
  with wget. If listing is blocked, whole-share archive endpoints are tried, then
  any direct URLs in cfg["data"]["extra_urls"].
"""
import hashlib
import os
import shutil
import subprocess
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

import requests

CERNBOX = "https://cernbox.cern.ch"
DATA_EXTS = (".pt", ".pth", ".parquet", ".h5", ".hdf5", ".npz", ".npy")


def raw_dir(cfg):
    d = os.path.join(cfg["paths"]["raw_root"], "calogan" if cfg.get("family") == "calogan" else cfg["dataset"])
    os.makedirs(d, exist_ok=True)
    return d


def fetch(url, dest, expected_size=None):
    """Resumable download with wget (falls back to requests)."""
    if os.path.exists(dest) and expected_size and os.path.getsize(dest) == expected_size:
        print(f"[skip] {os.path.basename(dest)} already downloaded")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[get ] {url}\n    -> {dest}")
    if shutil.which("wget"):
        subprocess.run(["wget", "-c", "--progress=dot:giga", "-O", dest, url], check=True)
    else:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
    return dest


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- Zenodo
def download_calo(cfg):
    d = cfg["data"]
    out = raw_dir(cfg)
    record = d["zenodo_record"]
    wanted = set(d["files"])
    meta = {}
    try:
        r = requests.get(f"https://zenodo.org/api/records/{record}", timeout=30)
        r.raise_for_status()
        for f in r.json()["files"]:
            if f["key"] in wanted:
                meta[f["key"]] = f
    except Exception as e:  # API down: fall back to the known file names
        print(f"Zenodo API unavailable ({e}); using known file names")
    paths = []
    for name in d["files"]:
        info = meta.get(name, {})
        url = f"https://zenodo.org/records/{record}/files/{name}?download=1"
        dest = os.path.join(out, name)
        fetch(url, dest, info.get("size"))
        checksum = info.get("checksum", "")
        if checksum.startswith("md5:"):
            ok = md5sum(dest) == checksum[4:]
            print(f"[md5 ] {name}: {'OK' if ok else 'MISMATCH - delete and re-run'}")
            if not ok:
                raise RuntimeError(f"checksum mismatch for {dest}")
        paths.append(dest)
    return paths


# -------------------------------------------------------------------- CERNBox
_PROPFIND = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop>'
             '<d:resourcetype/><d:getcontentlength/></d:prop></d:propfind>')


def list_cernbox(token, sub=""):
    """Recursively list a public CERNBox share -> [(relative_path, size)]."""
    base = f"{CERNBOX}/remote.php/dav/public-files/{token}"
    url = f"{base}/{urllib.parse.quote(sub)}" if sub else base + "/"
    r = requests.request("PROPFIND", url, data=_PROPFIND, headers={"Depth": "1"}, timeout=60)
    r.raise_for_status()
    ns = {"d": "DAV:"}
    files = []
    for resp in ET.fromstring(r.content).findall("d:response", ns):
        href = urllib.parse.unquote(resp.find("d:href", ns).text)
        rel = href.split(token, 1)[-1].strip("/")
        if rel == sub.strip("/"):
            continue  # the folder itself
        if resp.find(".//d:resourcetype/d:collection", ns) is not None:
            files += list_cernbox(token, rel)
        else:
            size = resp.find(".//d:getcontentlength", ns)
            files.append((rel, int(size.text) if size is not None and size.text else None))
    return files


def _extract(archive, out):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(out)
    else:
        raise RuntimeError(f"{archive} is neither zip nor tar")
    os.remove(archive)


def find_data_files(directory):
    found = []
    for root, _, names in os.walk(directory):
        found += [os.path.join(root, n) for n in names if n.lower().endswith(DATA_EXTS)]
    return sorted(found)


def download_qg(cfg):
    d = cfg["data"]
    out = raw_dir(cfg)
    token = d["cernbox_token"]
    try:
        listing = list_cernbox(token)
        print(f"CERNBox share contains {len(listing)} files:")
        for rel, size in listing:
            print(f"   {rel}  ({(size or 0) / 1e9:.2f} GB)")
        for rel, size in listing:
            if rel.lower().endswith(DATA_EXTS):
                url = f"{CERNBOX}/remote.php/dav/public-files/{token}/{urllib.parse.quote(rel)}"
                fetch(url, os.path.join(out, rel), size)
    except Exception as e:
        print(f"WebDAV listing failed ({e}); trying archive download of the whole share")
        archive = os.path.join(out, "cernbox_share.archive")
        for url in (f"{CERNBOX}/archiver?public-token={token}",
                    f"{CERNBOX}/index.php/s/{token}/download"):
            try:
                fetch(url, archive)
                _extract(archive, out)
                break
            except Exception as e2:
                print(f"   {url} failed: {e2}")
                if os.path.exists(archive):
                    os.remove(archive)
    for url in d.get("extra_urls", []):
        name = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path)) or "file"
        fetch(url, os.path.join(out, name))
    files = find_data_files(out)
    if not files:
        raise RuntimeError(
            "No jet files found. Download them from the CERNBox page manually, put them in "
            f"{out} (or on Drive and set paths.raw_root), or add direct links to data.extra_urls.")
    print("Jet data files:", *files, sep="\n   ")
    return files


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download_calogan(cfg):
    """CaloGAN (Mendeley Data, DOI 10.17632/pvn3xc3wy5.1): eplus / gamma / piplus .hdf5, 406 MB each, fetched with
    a resumable wget and checked against the sha256 Mendeley lists. Files already in data.source_dir (default
    {raw_root}/calogan) are used as they are; with data.source_dir set nothing is downloaded. A smoke run uses
    whatever files are there."""
    d = cfg["data"]
    out = d["source_dir"] or raw_dir(cfg)
    found = {}
    for particle in d["particles"]:
        info = d["files"][particle]
        dest = os.path.join(out, f"{particle}.hdf5")
        have = os.path.exists(dest)
        if d["source_dir"] or cfg.get("smoke"):
            if not have:
                raise RuntimeError(f"{dest} not found (data.source_dir is set, so nothing is downloaded)")
        elif not have or os.path.getsize(dest) != info["bytes"]:
            fetch(d["url"].format(id=info["id"]), dest, info["bytes"])
        if not (d["source_dir"] or cfg.get("smoke")):
            digest = sha256sum(dest)
            if digest != info["sha256"]:
                os.remove(dest)   # a corrupt file would otherwise be resumed from forever
                raise RuntimeError(f"{dest}: sha256 {digest} != {info['sha256']}; deleted, run the download again")
        print(f"  {particle}: {dest} ({os.path.getsize(dest) / 1e6:.0f} MB)")
        found[particle] = dest
    return found


def download(cfg):
    if cfg.get("family") == "calogan":
        return download_calogan(cfg)
    return download_calo(cfg) if cfg["dataset"] == "calo" else download_qg(cfg)
