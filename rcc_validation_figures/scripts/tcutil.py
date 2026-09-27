"""Shared helpers for the final TRACER campaign analyses.

Everything here exists because both SLURM allocations are exhausted and the
login cgroup is 5 GB shared with other processes, so large arrays are streamed
to disk-backed memmaps rather than materialised.
"""
from __future__ import annotations
import ctypes, gc, io, os, time, zipfile
from pathlib import Path
import numpy as np

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None
_CG = Path("/sys/fs/cgroup/memory/user.slice") / f"user-{os.getuid()}.slice"


def trim():
    """Return freed memory to the OS.

    Both glibc arenas and the pyarrow memory pool otherwise retain hundreds of
    MB after a streaming parquet pass, which matters inside a shared 5 GB login
    cgroup.
    """
    gc.collect()
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def rss_mb():
    try:
        return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2 ** 20
    except Exception:
        return -1.0


def cgroup_free_mb():
    """Headroom against ANONYMOUS memory only.

    memory.usage_in_bytes includes the page cache, which this pipeline fills
    with its own streaming reads and which the kernel reclaims on demand; using
    it as the headroom signal deadlocks the wait loop.
    """
    try:
        lim = int((_CG / "memory.limit_in_bytes").read_text())
        stat = {}
        for line in (_CG / "memory.stat").read_text().splitlines():
            k, _, v = line.partition(" ")
            stat[k] = int(v)
        rss = stat.get("total_rss", 0) + stat.get("total_swap", 0)
        return (lim - rss) / 2 ** 20
    except Exception:
        return 1e9


def mklog(tag):
    def log(m):
        print(f"[{tag}] rss={rss_mb():5.0f}M free={cgroup_free_mb():5.0f}M {m}",
              flush=True)
    return log


def wait_for_headroom(need_mb, tries=240, log=print):
    for i in range(tries):
        if cgroup_free_mb() >= need_mb:
            return True
        if i % 20 == 0:
            log(f"waiting for {need_mb:.0f}M headroom")
        time.sleep(15)
    return False


def npz_to_memmap(npz_path, key, out_path, chunk=1 << 22):
    """Stream one member of an .npz to a .npy on disk without materialising it.

    savez_compressed members are deflate streams; zipfile hands back a
    decompressing file object, so the array can be copied through a small
    buffer.  Returns (memmap, shape, dtype).
    """
    out_path = Path(out_path)
    with zipfile.ZipFile(npz_path) as z:
        name = key if key in z.namelist() else key + ".npy"
        with z.open(name) as f:
            ver = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
            assert not fortran, "unexpected Fortran order"
            if not out_path.exists() or out_path.stat().st_size != int(
                    np.prod(shape)) * dtype.itemsize:
                with open(out_path, "wb") as o:
                    while True:
                        b = f.read(chunk)
                        if not b:
                            break
                        o.write(b)
    return np.memmap(out_path, dtype=dtype, mode="r", shape=shape), shape, dtype


def npz_small(npz_path, key):
    """Load a small member of an .npz normally."""
    with np.load(npz_path, allow_pickle=True) as z:
        return z[key]


def cliffs_delta(a, b, cap=200_000, seed=0):
    """P(a>b) - P(a<b), subsampled for large n."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if len(a) > cap:
        a = rng.choice(a, cap, replace=False)
    if len(b) > cap:
        b = rng.choice(b, cap, replace=False)
    a = np.sort(a)
    gt = np.searchsorted(a, b, side="left").sum()
    lt = len(a) * len(b) - np.searchsorted(a, b, side="right").sum()
    return float((lt - gt) / (len(a) * len(b)))


def boot_ci(v, stat=np.median, n=2000, alpha=0.05, seed=0, cap=20_000,
            chunk=64):
    """Percentile bootstrap CI, drawn in chunks so the index matrix stays small.

    Large samples are first subsampled to `cap`; the CI then refers to that
    subsample, which is conservative (wider) rather than optimistic.
    """
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    if len(v) > cap:
        v = rng.choice(v, cap, replace=False)
    out = np.empty(n)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        idx = rng.integers(0, len(v), size=(b - a, len(v)))
        out[a:b] = stat(v[idx], axis=1)
        del idx
    return (float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    v = p[ok]
    if len(v) == 0:
        return q
    o = np.argsort(v)
    r = np.empty(len(v), int); r[o] = np.arange(1, len(v) + 1)
    adj = v * len(v) / r
    s = np.argsort(-v)
    run = np.minimum.accumulate(adj[s])
    out = np.empty(len(v)); out[s] = run
    q[ok] = np.clip(out, 0, 1)
    return q


def stars(q):
    if not np.isfinite(q):
        return "ns"
    return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "ns"


def signflip_exact(d):
    """Exact two-sided paired sign-flip (permutation) test on differences.

    With n <= 6 patients this enumerates all 2^n sign assignments; the smallest
    attainable two-sided p is 2^-(n-1), i.e. 0.031 at n = 6.
    """
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    n = len(d)
    if n == 0:
        return np.nan, 0
    obs = abs(d.mean())
    cnt = 0
    for m in range(1 << n):
        sg = np.array([1.0 if (m >> i) & 1 else -1.0 for i in range(n)])
        if abs((d * sg).mean()) >= obs - 1e-12:
            cnt += 1
    return cnt / (1 << n), n


def csr_subset_from_npz(npz_path, row_idx, tmpdir):
    """CSR rows `row_idx` from a scipy .npz without materialising the matrix.

    data/indices are streamed to disk-backed memmaps first, then only the
    requested rows are copied into RAM.  A cervical arm is 836k x 5.1k with
    ~62M nonzeros (~500 MB); the 150k scored rows are ~90 MB.
    """
    import scipy.sparse as sp
    tmpdir = Path(tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    stem = Path(npz_path).stem
    shape = tuple(int(v) for v in npz_small(npz_path, "shape"))
    indptr = npz_small(npz_path, "indptr").astype(np.int64)
    data, _, _ = npz_to_memmap(npz_path, "data", tmpdir / f"{stem}.data")
    idx, _, _ = npz_to_memmap(npz_path, "indices", tmpdir / f"{stem}.idx")
    row_idx = np.asarray(row_idx, np.int64)
    lens = (indptr[row_idx + 1] - indptr[row_idx]).astype(np.int64)
    out_ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    nd = np.empty(int(out_ptr[-1]), data.dtype)
    ni = np.empty(int(out_ptr[-1]), idx.dtype)
    for k, r in enumerate(row_idx):
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b > a:
            s = int(out_ptr[k])
            nd[s:s + (b - a)] = data[a:b]
            ni[s:s + (b - a)] = idx[a:b]
    del data, idx
    trim()
    return sp.csr_matrix((nd, ni, out_ptr), shape=(len(row_idx), shape[1]))
