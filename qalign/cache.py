"""
Packed-blob image cache — the dataloading speedup, model-agnostic.

QA training/eval reads the same images every epoch / every checkpoint. On a
network/FUSE mount each small jpg/png is a round-trip, so a quarter-million tiny
opens dominate wall-clock and starve the GPU (we measured ~40-50% util; packing
took it to ~64% and an ~18x dataloading speedup).

Fix: concatenate every referenced image **byte-for-byte** (no resize — pixel
fidelity is the QA signal) into one ``<name>.blob`` + an index
``index.json = {orig_path: [blob, offset, length]}``. At run time we mmap the
blob once and serve each image as a RAM slice (the OS page cache, or explicit
prefault, keeps the hot bytes resident). One file handle, no small-file storm,
byte-identical to the originals.

Two install points (the only swift/torch touch, both lazy):
  * ``install_train_hook`` patches ``swift.template.vision_utils.load_file``
    (used by the dataloader) — call from a ``--custom_register_path`` shim.
  * ``install_eval_hook`` patches ``qalign.scorer.open_image`` (used by the
    scorer) and forces the blob resident in RAM up front.
"""
import os, json, mmap, sys, threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

INDEX_NAME = "index.json"


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #
def _manifest_image_paths(manifest):
    """Every image path a manifest references (key 'images' for train, 'src' for eval)."""
    paths = []
    for line in open(manifest):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "images" in r:
            paths.extend(r["images"])
        elif r.get("modality") == "image" and "src" in r:
            paths.append(r["src"])
        # video 'src' entries are sampled at eval time, not packed here
    return paths


def _read_bytes(path):
    try:
        with open(path, "rb") as f:
            return path, f.read()
    except Exception:
        return path, None


def pack(paths, cache_dir, blob_name, workers=64):
    """Concatenate ``paths`` (deduped, first-seen order) into cache_dir/<blob_name>.

    Writes ``<blob_name>`` + merges into ``index.json``. Returns (blob_path, n, errors).
    """
    os.makedirs(cache_dir, exist_ok=True)
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)

    blob_path = os.path.join(cache_dir, blob_name)
    local = {}
    n_err = offset = done = 0
    # read source bytes in parallel (IO-bound); WRITE from the main thread so
    # offsets are deterministic and the blob is one contiguous stream.
    with open(blob_path, "wb") as blob, ThreadPoolExecutor(max_workers=workers) as ex:
        for path, data in ex.map(_read_bytes, uniq):
            if data is None:
                n_err += 1
                continue
            blob.write(data)
            local[path] = [blob_name, offset, len(data)]
            offset += len(data)
            done += 1
            if done % 20000 == 0:
                print(f"  [{blob_name}] {done}/{len(uniq)}  {offset/1e9:.1f}GB", file=sys.stderr)

    # merge into the global index (re-packing one blob must not drop others)
    gi = os.path.join(cache_dir, INDEX_NAME)
    combined = {}
    if os.path.exists(gi):
        try:
            combined = json.load(open(gi))
        except Exception:
            combined = {}
    combined.update(local)
    json.dump(combined, open(gi, "w"))
    print(f"[pack {blob_name}] {len(local)} imgs, {offset/1e9:.2f} GB, errors={n_err} "
          f"-> {blob_path} (index now {len(combined)})", file=sys.stderr)
    return blob_path, len(local), n_err


def pack_manifests(manifests, cache_dir, blob_name="images.blob", workers=64):
    """Pack every image referenced across a list of manifest files into one blob."""
    paths = []
    for m in manifests:
        if os.path.exists(m):
            paths.extend(_manifest_image_paths(m))
        else:
            print(f"[pack] manifest missing, skipped: {m}", file=sys.stderr)
    return pack(paths, cache_dir, blob_name, workers)


# --------------------------------------------------------------------------- #
# Runtime hooks
# --------------------------------------------------------------------------- #
_lock = threading.Lock()


class _BlobStore:
    """Lazily mmaps blobs in a cache dir and serves index slices as bytes."""
    def __init__(self, cache_dir, populate=False):
        self.cache_dir = cache_dir
        self.populate = populate
        self.index = json.load(open(os.path.join(cache_dir, INDEX_NAME)))
        self._blobs = {}

    def _blob(self, name):
        mm = self._blobs.get(name)
        if mm is None:
            with _lock:
                mm = self._blobs.get(name)
                if mm is None:
                    fd = os.open(os.path.join(self.cache_dir, name), os.O_RDONLY)
                    try:
                        flags = mmap.MAP_PRIVATE
                        if self.populate and hasattr(mmap, "MAP_POPULATE"):
                            flags |= mmap.MAP_POPULATE     # prefault all pages on map
                        mm = mmap.mmap(fd, 0, flags=flags, prot=mmap.PROT_READ)
                    finally:
                        os.close(fd)
                    if self.populate:
                        _residentize(mm)
                    self._blobs[name] = mm
        return mm

    def get(self, path):
        hit = self.index.get(path) or (isinstance(path, str) and self.index.get(path.strip()))
        if not hit:
            return None
        name, off, ln = hit
        return self._blob(name)[off:off + ln]

    def total_resident(self):
        return sum(len(mm) for mm in self._blobs.values())


def _residentize(mm):
    """Force the whole mmap into physical RAM (prefault every page)."""
    try:
        mm.madvise(mmap.MADV_WILLNEED)
    except Exception:
        pass
    n, s, chunk = len(mm), 0, mmap.PAGESIZE * 4096   # ~16 MiB strides
    while s < n:
        _ = bytes(mm[s:min(s + chunk, n)])
        s += chunk


def install_train_hook(cache_dir):
    """Patch swift's dataloader image loader to serve blob slices. Best-effort.

    Returns True if active. Safe to call when the cache is absent (falls back to
    normal file IO). Intended to be invoked from a ``--custom_register_path`` shim.
    """
    gi = os.path.join(cache_dir, INDEX_NAME)
    if not os.path.exists(gi):
        print(f"[qalign-cache] no index at {gi}; training uses normal file IO", flush=True)
        return False
    from swift.template import vision_utils
    store = _BlobStore(cache_dir, populate=False)
    orig = vision_utils.load_file

    def cached_load_file(path):
        if isinstance(path, str):
            b = store.get(path)
            if b is not None:
                return BytesIO(b)               # exact original bytes
        return orig(path)

    vision_utils.load_file = cached_load_file
    print(f"[qalign-cache] train cache active: {len(store.index)} images, {cache_dir}", flush=True)
    return True


def install_eval_hook(cache_dir):
    """Patch qalign.scorer.open_image to serve RAM-resident blob slices. Idempotent."""
    gi = os.path.join(cache_dir, INDEX_NAME)
    if not os.path.exists(gi):
        print(f"[qalign-eval-cache] no index at {gi}; eval uses normal file IO", flush=True)
        return False
    from PIL import Image
    from . import scorer
    store = _BlobStore(cache_dir, populate=True)
    for name in {v[0] for v in store.index.values()}:    # pull every blob resident now
        store._blob(name)
    orig = scorer.open_image

    def cached_open_image(path):
        if isinstance(path, str):
            b = store.get(path)
            if b is not None:
                return Image.open(BytesIO(b)).convert("RGB")
        return orig(path)

    scorer.open_image = cached_open_image
    print(f"[qalign-eval-cache] RAM-resident eval cache: {len(store.index)} files, "
          f"{store.total_resident()/1e9:.2f} GB pinned, {cache_dir}", flush=True)
    return True
