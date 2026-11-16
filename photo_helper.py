"""
No-Nonsense Photo Sorter
Pick a folder and a job (everything, or just one step):
  copies set aside, photos sorted by date (memes/GIFs sorted separately), faces named, names saved in the photos.
Face detection/recognition: OpenCV (YuNet + SFace). Metadata: exiftool (the copy inside digiKam is fine).
digiKam reads the saved names (people tags + face regions) the next time it scans the folder.
"""
import sys, os, re, json, shutil, hashlib, subprocess, tempfile, traceback, urllib.request
import base64, threading, time, ctypes, uuid, sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PySide6.QtCore import Qt, QThread, Signal, QStringListModel, QTimer, QSize, QUrl, QPoint, QObject, QEvent
from PySide6.QtGui import QFontDatabase, QPixmap, QImage, QPainter, QPainterPath, QIcon, QDesktopServices, QColor, QPen
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QLineEdit, QProgressBar, QStackedWidget, QFileDialog, QMessageBox,
                               QCompleter, QDialog, QComboBox, QListWidget, QListWidgetItem, QCheckBox,
                               QFrame, QSpinBox, QScrollArea, QProgressDialog)

APP_VERSION = "0.7.1"
APP_NAME = "No-Nonsense Photo Sorter"
APP_SHORT_NAME = "NNPS"
FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent   # next to the .exe
RES_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))                                        # the "bin" folder
MODEL_DIR = RES_DIR / "models"
CONFIG = APP_DIR / "config.json"
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"
MODEL_REV = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODEL_URLS = {name: f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_REV}/models/{folder}/{name}"
              for name, folder in ((YUNET, "face_detection_yunet"), (SFACE, "face_recognition_sface"))}
MODEL_SHA256 = {YUNET: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                SFACE: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}          # faces can be looked for in these
RAW_EXT = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".orf", ".rw2", ".raf",
           ".pef", ".srw", ".x3f", ".3fr", ".erf", ".kdc", ".dcr", ".mrw", ".rwl", ".iiq"}   # camera RAW (CR2 = Canon)
MEDIA_EXT = IMG_EXT | RAW_EXT | {".heic", ".heif", ".hif", ".dng", ".gif", ".bmp",   # iPhone photos, RAW etc: sorted + copies only
                       ".mov", ".mp4", ".m4v", ".avi", ".3gp", ".mts", ".m2ts", ".mkv", ".wmv", ".mpg", ".mpeg"}
MAX_SIDE = 1280      # photos are shrunk to this for face finding (speed)
MIN_FACE = 48        # ignore tiny faces (pixels, after shrinking)
DET_SCORE = 0.85
CLUSTER_T = 0.42     # similarity needed to group two faces as "same person"
MATCH_T = 0.40       # similarity needed to tag a face from a named example
SAMPLES = 6          # pictures shown per person
MIN_CLUSTER = 3      # people with fewer photos than this are not asked about
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
INFO_DIR = "_Photo Helper"   # keep the existing per-library folder name so prior runs remain readable
MOVE_LOG = "moves.jsonl"
_run_ids = {}
_move_lock = threading.Lock()
CANCEL_EVENT = threading.Event()
HASH_WORKERS = 4             # files read at the same time when checking copies
CPUS = os.cpu_count() or 2
FACE_WORKERS = max(1, min(4, CPUS - 1))   # photos looked at for faces at the same time
EXIF_WORKERS = max(1, min(3, CPUS - 1))   # exiftool runs at the same time
EXIF_CHUNK = 400                          # files per exiftool run (each run has a start-up cost, so bigger is faster)
FP_BLOCK = 64 * 1024         # size of each sample read for the quick look at a file
BUSY = {"scan", "naming", "waiting", "saving", "filing", "compressing", "removing_faces"}
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".3gp", ".mts", ".m2ts", ".mkv", ".wmv", ".mpg", ".mpeg"}
PEOPLE_EXT = MEDIA_EXT - VIDEO_EXT   # files that can carry saved names
SHOW_LIMIT = 300                     # most pictures shown in the grid (copying always uses all matches)

# Memes / GIFs / screenshots are sorted into their own folder, separate from real photos.
# ---- options (the panel on the right). The module-level numbers above are the live values; apply_options() sets them.
DEFAULT_OPTS = {"accuracy": "balanced", "matching": "normal", "speed": "auto",
                "samples": 6, "min_photos": 3, "keep_awake": True, "open_folder": False,
                "video_quality": "balanced", "video_codec": "h265", "video_min_mb": 20, "video_replace": False,
                "meme_filter": "strict", "sort_preview": False, "remember_people": True,
                "review_faces": False, "use_calibration": True}
OPTS = dict(DEFAULT_OPTS)
ACCURACY = [   # key, label, explanation, (MAX_SIDE, MIN_FACE, DET_SCORE)
    ("fast", "Fast", "Quicker. May miss small or blurry faces.", (960, 56, 0.90)),
    ("balanced", "Balanced (default)", "A good mix of speed and finding faces.", (1280, 48, 0.85)),
    ("thorough", "Thorough", "Finds smaller and harder faces, but is slower and may pick up a few things that are not faces.",
     (1920, 36, 0.80)),
]
MATCHING = [   # key, label, explanation, (CLUSTER_T, MATCH_T)
    ("strict", "Strict", "Fewer wrong names. The same person may be asked about more than once.", (0.48, 0.46)),
    ("normal", "Normal (default)", "A good mix.", (0.42, 0.40)),
    ("loose", "Loose", "Names more photos, but a few may get the wrong name.", (0.36, 0.34)),
]
SPEEDS = [     # key, label, explanation, (FACE_WORKERS, EXIF_WORKERS)
    ("gentle", "Gentle", "One thing at a time. Slowest, but the PC stays free for other work.", (1, 1)),
    ("auto", "Balanced (default)", "Uses some of your PC's power.",
     (max(1, min(4, CPUS - 1)), max(1, min(3, CPUS - 1)))),
    ("fast", "Fastest", "Uses most of your PC. It may feel slow to do other things meanwhile.",
     (max(1, min(8, CPUS)), max(1, min(4, CPUS)))),
]
VIDEO_QUALITY = [   # key, label, explanation, CRF (lower = better picture, bigger file) for (H.265, H.264)
    ("small", "Smaller file", "Smallest files. Fine for phone videos; fast action may look a little softer.", (30, 26)),
    ("balanced", "Balanced (default)", "Much smaller, and hard to tell apart from the original.", (27, 23)),
    ("best", "Best quality", "Looks almost the same as the original. Files are not as small.", (23, 20)),
]
VIDEO_CODECS = [    # key, label, explanation, ffmpeg encoder
    ("h265", "H.265 (smallest)", "Smallest files. Very old TVs and phones may not play it.", "libx265"),
    ("h264", "H.264 (plays everywhere)", "Bigger than H.265, but plays on practically everything.", "libx264"),
]
MEME_FILTERS = [   # key, label, explanation, (unused)
    ("standard", "Standard", "Only GIFs, screenshots, PNG/WebP pictures with no camera information, and files or folders "
     "with a word like meme or sticker in the name.", None),
    ("strict", "Strict (default)", "Also looks inside pictures with no camera information for caption blocks, "
     "meme lettering and quote cards.", None),
    ("strictest", "Strictest", "Every picture with no camera information counts as a meme. This also catches photos "
     "sent to you by WhatsApp or Messenger, because they lose that information.", None),
]
OPT_W = 290          # width of the options panel


def apply_options(o):
    """Check the saved options and set the live numbers from them."""
    global MAX_SIDE, MIN_FACE, DET_SCORE, CLUSTER_T, MATCH_T, FACE_WORKERS, EXIF_WORKERS, SAMPLES, MIN_CLUSTER
    new = dict(DEFAULT_OPTS)
    for k, v in (o or {}).items():
        if k in new:
            new[k] = v
    for k, table in (("accuracy", ACCURACY), ("matching", MATCHING), ("speed", SPEEDS)):
        if new[k] not in [row[0] for row in table]:
            new[k] = DEFAULT_OPTS[k]
    for k, table in (("video_quality", VIDEO_QUALITY), ("video_codec", VIDEO_CODECS), ("meme_filter", MEME_FILTERS)):
        if new[k] not in [row[0] for row in table]:
            new[k] = DEFAULT_OPTS[k]
    for k, lo, hi in (("samples", 3, 12), ("min_photos", 2, 20), ("video_min_mb", 0, 1000)):
        try:
            new[k] = max(lo, min(hi, int(new[k])))
        except (TypeError, ValueError):
            new[k] = DEFAULT_OPTS[k]
    new["keep_awake"], new["open_folder"] = bool(new["keep_awake"]), bool(new["open_folder"])
    new["video_replace"] = bool(new["video_replace"])
    new["sort_preview"] = bool(new["sort_preview"])
    for key in ("remember_people", "review_faces", "use_calibration"):
        new[key] = bool(new[key])
    OPTS.clear()
    OPTS.update(new)
    MAX_SIDE, MIN_FACE, DET_SCORE = next(r[3] for r in ACCURACY if r[0] == new["accuracy"])
    CLUSTER_T, MATCH_T = next(r[3] for r in MATCHING if r[0] == new["matching"])
    FACE_WORKERS, EXIF_WORKERS = next(r[3] for r in SPEEDS if r[0] == new["speed"])
    SAMPLES, MIN_CLUSTER = new["samples"], new["min_photos"]


MEMES_DIR = "Sorted Memes & GIFs"
WEB_EXT = {".png", ".webp", ".bmp"}   # count as memes unless the file says it came from a camera
# A folder or file name containing one of these words counts as a meme/GIF (gifts, gifted etc. do not match)
MEME_WORDS = re.compile(r"(?<![a-z])(memes?|gifs?|stickers?|screenshots?)(?![a-z])", re.I)
# extra words used by the Strict and Strictest filters
MEME_WORDS_MORE = re.compile(r"(?<![a-z])(captions?|quotes?|funny|humou?r|jokes?|reddit|9gag|imgflip|giphy|tenor|"
                             r"dank|wholesome|motivational|inspirational)(?![a-z])", re.I)
# picture-editing / meme apps that write their name into the file (only trusted when there is no camera information)
MEME_SOFTWARE = re.compile(r"canva|photoshop|gimp|pixlr|picsart|imgflip|phonto|photoroom|adobe express|fotor|kapwing|"
                           r"meme|instagram|facebook|whatsapp|snapchat|tiktok|giphy|paint", re.I)
CAMERA_TAGS = ("Model", "Make", "ExposureTime", "FNumber", "ISO", "FocalLength", "LensModel")


# The job list on the first screen: (key, label, explanation). Order = order in the list.
# copies / sort / faces can be ticked in any mix; "people" and "broken" are separate jobs that run on their own.
TASKS = [
    ("copies", "Find copies",
     "Moves exact copies and look-alikes (re-saved, resized or renamed like “photo (1).jpg”) into “_Duplicates” (exact) and “_Verify” (lower-quality look-alikes)."),
    ("sort", "Sort by date",
     "Sorts into date folders (photos and memes/GIFs separately)."),
    ("faces", "Name faces",
     "Looks for faces and saves the names into the photos."),
    ("people", "Find photos of people",
     "Pick the people you want, look at their photos, and copy them into a folder with their name."),
    ("broken", "Find broken photos",
     "Checks every photo and video for damage and moves the broken ones into “_Broken”, with a report. Nothing is deleted."),
    ("compress", "Compress videos",
     "Makes videos smaller (needs ffmpeg). Your originals are kept. See Options for quality and for replacing the originals."),
    ("names_off", "Remove names from photos",
     "Takes the face names, face boxes and people tags out of your photos. Back up your photos first; image backups are not made."),
]
COMBINABLE = ("copies", "sort", "faces")
STANDALONE = ("people", "broken", "compress", "names_off")
TASK_LABEL = {k: t for k, t, _ in TASKS}
OLD_MODES = {"all": ["copies", "sort", "faces"], "copies": ["copies"], "sort": ["sort"], "faces": ["faces"],
             "people": ["people"], "broken": ["broken"]}   # from older config.json files


class Cancelled(Exception):
    """Raised inside a worker when the app is being closed on purpose."""


# ------------------------------------------------------------------ helpers
def norm(p):
    return os.path.normcase(os.path.normpath(str(p)))


def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg):
    try:
        CONFIG.write_text(json.dumps(cfg), encoding="utf-8")
    except Exception:
        pass


def find_ffmpeg():
    cfg = load_config()
    configured = cfg.get("ffmpeg")
    if isinstance(configured, str) and Path(configured).name.lower() == "ffmpeg.exe" and Path(configured).is_file():
        return configured
    for cand in (RES_DIR / "ffmpeg" / "ffmpeg.exe", RES_DIR / "ffmpeg.exe", RES_DIR / "bin" / "ffmpeg.exe",
                 APP_DIR / "ffmpeg.exe", APP_DIR / "ffmpeg" / "ffmpeg.exe", APP_DIR / "bin" / "ffmpeg.exe",
                 APP_DIR / "bin" / "ffmpeg" / "ffmpeg.exe", APP_DIR.parent / "bin" / "ffmpeg.exe",
                 APP_DIR.parent / "bin" / "ffmpeg" / "ffmpeg.exe"):
        if cand.exists():
            return str(cand)
    found = shutil.which("ffmpeg")
    if found:
        return found
    for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if not base:
            continue
        for sub in ("digiKam", "digikam", "Programs/digiKam"):
            root = Path(base) / sub
            if root.exists():
                for dp, _, files in os.walk(root):
                    if "ffmpeg.exe" in files:
                        return os.path.join(dp, "ffmpeg.exe")
    return None


def find_exiftool():
    cfg = load_config()
    configured = cfg.get("exiftool")
    if isinstance(configured, str) and Path(configured).name.lower() == "exiftool.exe" and Path(configured).is_file():
        return configured
    bundled = RES_DIR / "exiftool" / "exiftool.exe"
    if bundled.exists():
        return str(bundled)
    here = APP_DIR / "exiftool.exe"
    if here.exists():
        return str(here)
    found = shutil.which("exiftool")
    if found:
        return found
    for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if not base:
            continue
        for sub in ("digiKam", "digikam", "Programs/digiKam"):
            root = Path(base) / sub
            if root.exists():
                for dp, _, files in os.walk(root):
                    if "exiftool.exe" in files:
                        return os.path.join(dp, "exiftool.exe")
    return None


def run_exif(exe, lines, tolerate_file_errors=False):
    """Run exiftool with an argument file (UTF-8, one argument per line).
    tolerate_file_errors: for read-only JSON runs, exit code 1 only means some files had an Error, which the JSON
    already reports per file; keep the output instead of failing the whole batch."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        name = f.name
    try:
        if CANCEL_EVENT.is_set():
            raise Cancelled()
        proc = subprocess.Popen([exe, "-@", name], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=os.path.dirname(exe) or None, creationflags=NO_WINDOW)
        try:
            while True:
                if CANCEL_EVENT.is_set():
                    raise Cancelled()
                try:
                    stdout, stderr = proc.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        if proc.returncode and not (tolerate_file_errors and proc.returncode == 1 and stdout.strip()):
            raise RuntimeError("ExifTool could not complete the metadata operation: " +
                               stderr.decode("utf-8", "replace")[-500:])
        return stdout.decode("utf-8", "replace")
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def unique_path(dest):
    dest = Path(dest)
    if not dest.exists():
        return dest
    i = 1
    while True:
        cand = dest.with_name(f"{dest.stem}_{i}{dest.suffix}")
        if not cand.exists():
            return cand
        i += 1


def start_move_run(root):
    """Give every launch of a folder job its own undo group."""
    root = Path(root).resolve()
    run_id = uuid.uuid4().hex
    log = root / INFO_DIR / MOVE_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    with _move_lock, log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "run": run_id}) + "\n")
    _run_ids[norm(root)] = run_id


def _move_entries(root):
    path = Path(root) / INFO_DIR / MOVE_LOG
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if isinstance(entry, dict):
                        yield entry
                except (ValueError, TypeError):
                    continue
    except OSError:
        return


def pending_moves(root):
    """Moves in the latest run that have not already been undone."""
    entries = list(_move_entries(root))
    runs = [e.get("run") for e in entries if e.get("event") in ("start", "move") and e.get("run")]
    if not runs:
        return []
    latest = runs[-1]
    undone = {e.get("id") for e in entries if e.get("event") == "undo" and e.get("run") == latest}
    return [e for e in entries if e.get("event") == "move" and e.get("run") == latest
            and e.get("id") not in undone]


def same_volume(src, directory):
    return Path(src).stat().st_dev == Path(directory).stat().st_dev


def verified_move(src, dst):
    """Copy across volumes into a temporary file and verify before removing the source."""
    src, dst = Path(src), Path(dst)
    if same_volume(src, dst.parent):
        shutil.move(str(src), str(dst))
        return
    before = src.stat()
    if shutil.disk_usage(dst.parent).free < before.st_size + (64 << 20):
        raise OSError("Not enough free space on the destination drive")
    temp = dst.with_name(dst.name + ".moving-" + uuid.uuid4().hex)
    try:
        shutil.copy2(src, temp)
        if sha256(src) != sha256(temp):
            raise OSError("Copied file failed verification; source retained")
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("Source changed during copying; source retained")
        # Windows rename refuses an existing destination rather than replacing it.
        if dst.exists():
            raise FileExistsError(str(dst))
        os.rename(temp, dst)
        src.unlink()
    finally:
        temp.unlink(missing_ok=True)


def move_logged(src, dst, root=None):
    """Record a planned move before moving, so an interrupted run can be undone."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    root = Path(root).resolve() if root else Path(os.path.commonpath((src.parent, dst.parent)))
    if not src.is_relative_to(root) or not dst.is_relative_to(root):
        raise ValueError("A logged move must stay inside the chosen folder")
    entry = {"event": "move", "id": uuid.uuid4().hex, "run": _run_ids.setdefault(norm(root), uuid.uuid4().hex),
             "time": datetime.now().isoformat(timespec="seconds"), "src": str(src), "dst": str(dst)}
    with _move_lock:
        log = root / INFO_DIR / MOVE_LOG
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        verified_move(src, dst)
    return dst


def undo_last_run(root):
    root = Path(root).resolve()
    moves = pending_moves(root)
    restored = skipped = 0
    for entry in reversed(moves):
        try:
            src, dst = Path(entry["src"]).resolve(), Path(entry["dst"]).resolve()
            if not src.is_relative_to(root) or not dst.is_relative_to(root) or src.exists() or not dst.is_file():
                skipped += 1
                continue
            src.parent.mkdir(parents=True, exist_ok=True)
            verified_move(dst, src)
            with (root / INFO_DIR / MOVE_LOG).open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "undo", "run": entry["run"], "id": entry["id"]}) + "\n")
            restored += 1
        except (OSError, KeyError, TypeError, ValueError):
            skipped += 1
    return {"restored": restored, "skipped": skipped}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exif_json(exe, args, paths, on_progress=None):
    """Run exiftool (-j) over many files, a few runs at the same time. Returns the list of JSON entries.
    on_progress(files_done, files_total) is called as each run finishes."""
    paths = [str(p) for p in paths]
    chunks = [paths[i:i + EXIF_CHUNK] for i in range(0, len(paths), EXIF_CHUNK)]

    def one(chunk):
        try:
            result = json.loads(run_exif(exe, ["-charset", "utf8", "-charset", "filename=utf8", "-j"] + args + chunk,
                                    tolerate_file_errors=True))
            if not isinstance(result, list) or not all(isinstance(d, dict) for d in result):
                raise ValueError("Expected a metadata list")
            return result
        except (ValueError, OSError) as exc:
            raise RuntimeError("Metadata could not be read. No date fallback was applied to this batch.") from exc

    out = []
    if not chunks:
        return out
    ex = ThreadPoolExecutor(min(EXIF_WORKERS, len(chunks)))
    try:
        for n, data in enumerate(ex.map(one, chunks), 1):
            out.extend(data)
            if on_progress:
                on_progress(min(n * EXIF_CHUNK, len(paths)), len(paths))
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return out


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def read_info(exe, paths, on_progress=None):
    """Return {normalized path: (month 'YYYY-MM' or None, has camera information, Software tag)} using the
    file's own data (exiftool)."""
    out = {}
    tags = ["-d", "%Y-%m", "-DateTimeOriginal", "-CreationDate", "-CreateDate", "-MediaCreateDate",
            "-Model", "-Make", "-ExposureTime", "-FNumber", "-ISO", "-FocalLength", "-LensModel", "-Software",
            "-ImageWidth", "-ImageHeight"]
    for d in exif_json(exe, tags, paths, on_progress):
        if d.get("Error"):
            continue
        month = next((str(d[k]) for k in ("DateTimeOriginal", "CreationDate", "CreateDate", "MediaCreateDate")
                      if re.fullmatch(r"(?:19|20)\d\d-(?:0[1-9]|1[0-2])", str(d.get(k, "")))), None)
        out[norm(d.get("SourceFile", ""))] = (month, any(d.get(k) for k in CAMERA_TAGS), str(d.get("Software") or ""),
                                       (_num(d.get("ImageWidth")), _num(d.get("ImageHeight"))))
    return out


# ---- meme detection inside the picture
def _small(img, long_side=640):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _flat_band(gray, from_top):
    """A plain white or black block along the top (or bottom) edge that has lettering in it (caption pictures)."""
    g = gray if from_top else gray[::-1]
    H, W = g.shape
    med = int(np.median(g[0]))
    if not (med >= 225 or med <= 30):
        return False
    colour = 255 if med >= 225 else 0
    bg = np.abs(g[: int(H * 0.6)].astype(np.int16) - colour) <= 25
    frac = bg.mean(axis=1)
    rows = 0
    while rows < len(frac) and frac[rows] >= 0.22:
        rows += 1
    if rows < H * 0.07 or rows > H * 0.5:
        return False
    block = bg[:rows]
    if block.mean() < 0.72:
        return False
    ink = (~block).astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _text_rows(ink, H, W)


def _text_rows(mask, H, W, min_h=0.025):
    """True when the mask holds a row of letter-sized blobs lined up on one line (i.e. text)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 6:
        return False
    cand = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if min_h * H <= h <= 0.2 * H and 0.004 * W <= w <= 0.16 * W and a >= 0.25 * w * h * 0.35:
            cand.append((cen[i][1], h))
    if len(cand) < 6:
        return False
    ys = np.array([c[0] for c in cand])
    hs = np.array([c[1] for c in cand])
    # look for >=6 blobs whose centres sit on one line and are of similar height
    for y0 in ys:
        near = (np.abs(ys - y0) < 0.35 * np.median(hs)) & (np.abs(hs - np.median(hs)) < 0.6 * np.median(hs))
        if near.sum() >= 6:
            return True
    return False


def _overlay_text(img, part):
    """Meme-style lettering (a bright fill with a dark outline) in the top, middle or bottom of the picture."""
    H, W = img.shape[:2]
    y0, y1 = {"top": (0, 0.30), "mid": (0.30, 0.70), "bot": (0.70, 1.0)}[part]
    strip = img[int(H * y0): int(H * y1)]
    b, g, r = strip[..., 0].astype(np.int16), strip[..., 1].astype(np.int16), strip[..., 2].astype(np.int16)
    dark = (strip.max(axis=2) <= 45).astype(np.uint8)
    fills = ((strip.min(axis=2) >= 235),                                     # white lettering
             ((np.minimum(r, g) >= 200) & (b <= 120)))                       # yellow lettering
    for fill in fills:
        fill = fill.astype(np.uint8)
        fa = fill.sum()
        if fa < 0.01 * fill.size or fa > 0.35 * fill.size:
            continue
        touching = cv2.dilate(fill, np.ones((7, 7), np.uint8)) & dark
        if touching.sum() < 0.3 * fa:                    # lettering has a dark outline hugging it
            continue
        if _text_rows(fill * 255, H, W, min_h=0.05):
            return True
    return False


def _flat_graphic(img):
    """A quote card / sticker: most of the picture is one flat colour, with lettering on it."""
    q = (img // 24).astype(np.int32)
    code = q[..., 0] * 100 + q[..., 1] * 10 + q[..., 2]
    vals, counts = np.unique(code, return_counts=True)
    top = vals[counts.argmax()]
    share = counts.max() / code.size
    if share < 0.5:
        return False
    other = (code != top).astype(np.uint8) * 255
    other = cv2.morphologyEx(other, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _text_rows(other, img.shape[0], img.shape[1])


def looks_like_meme(img):
    """img: BGR picture. True for caption pictures, meme lettering, quote cards."""
    if img is None or min(img.shape[:2]) < 120:
        return False
    img = _small(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if _flat_band(gray, True) or _flat_band(gray, False):
        return True
    if _overlay_text(img, "top") or _overlay_text(img, "bot") or _overlay_text(img, "mid"):
        return True
    return _flat_graphic(img)


MEME_MAX_SIDE = 2600        # memes and caption pictures are small; bigger pictures are not opened for the check
MEME_MAX_BYTES = 3_000_000


def meme_pixels(path, side=0):
    """The picture as BGR for the meme check, opened at a smaller size (much faster than a full open)."""
    try:
        flag = cv2.IMREAD_REDUCED_COLOR_2 if side >= 1300 else cv2.IMREAD_COLOR
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    except Exception:
        return None


def meme_pixel_check(job):
    path, side = job
    try:
        return looks_like_meme(meme_pixels(path, side))
    except Exception:
        return False


def classify(files, root, info, level=None, on_progress=None):
    """Return the set of files that are memes / GIFs / screenshots rather than real photos.
    Videos, RAW files and pictures with camera information stay with photos. Standard: a GIF; a folder/file
    name with a word like meme, gif, sticker, screenshot; or a PNG/WebP/BMP. Strict adds: more name words, and pictures that have a caption
    block, meme lettering or a quote card in them (or were saved by a meme/editing app). Strictest: every picture
    with no camera information."""
    level = level or OPTS.get("meme_filter", "strict")
    root = Path(root)
    memes, to_look = set(), []
    for p in files:
        ext = p.suffix.lower()
        rec = info.get(norm(p))
        if ext in VIDEO_EXT or ext in RAW_EXT:
            continue
        if ext == ".gif":
            memes.add(p)
            continue
        if rec is None or rec[1]:
            continue
        try:
            parts = list(p.relative_to(root).parts[:-1])
            names = [d for d in parts if d not in ("Sorted", MEMES_DIR)] + [p.stem]
        except ValueError:
            names = [p.stem]
        text = " ".join(names)
        rec = info.get(norm(p), (None, False, ""))
        if MEME_WORDS.search(text) or (level != "standard" and MEME_WORDS_MORE.search(text)):
            memes.add(p)
        elif ext in WEB_EXT and not rec[1]:
            memes.add(p)
        elif level != "standard" and ext in IMG_EXT and not rec[1]:
            if level == "strictest" or MEME_SOFTWARE.search(rec[2] if len(rec) > 2 else ""):
                memes.add(p)
            else:
                w, h = rec[3] if len(rec) > 3 else (0, 0)
                try:
                    small_file = os.path.getsize(p) <= MEME_MAX_BYTES
                except OSError:
                    small_file = False
                if small_file and max(w, h) <= MEME_MAX_SIDE:        # big pictures are real photos or scans
                    to_look.append((p, max(w, h)))
    if to_look:                                            # strict: look inside the picture
        ex = ThreadPoolExecutor(max(2, min(6, CPUS - 1)))
        try:
            for n, ((p, _), hit) in enumerate(zip(to_look, ex.map(meme_pixel_check, to_look)), 1):
                if hit:
                    memes.add(p)
                if on_progress:
                    on_progress(n, len(to_look))
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
    return memes


def meme_set(files, exe, root, on_progress=None):
    """classify() for files that are not being sorted (only reads camera info where it is needed)."""
    level = OPTS.get("meme_filter", "strict")
    need = [p for p in files if p.suffix.lower() in (WEB_EXT if level == "standard" else IMG_EXT)]
    return classify(files, root, read_info(exe, need) if need else {}, level, on_progress)


def imread(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def make_thumb(img, x, y, w, h):
    H, W = img.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = min(max(w, h) * 1.8, W, H)
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    crop = img[int(y0):int(y0 + side), int(x0):int(x0 + side)]
    crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


class Face:
    __slots__ = ("path", "box", "orig", "emb", "quality", "thumb")

    def __init__(self, path, box, orig, emb, quality, thumb):
        self.path, self.box, self.orig, self.emb, self.quality, self.thumb = path, box, orig, emb, quality, thumb


# ------------------------------------------------------------------ pipeline
def ensure_models(progress):
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = MODEL_DIR / name
        if dest.exists() and sha256(dest) == MODEL_SHA256[name]:
            continue
        progress(0.0, "Getting ready (first time only)…")
        part = str(dest) + ".part"
        try:
            with urllib.request.urlopen(url, timeout=30) as response, open(part, "wb") as output:
                shutil.copyfileobj(response, output)
            if sha256(part) != MODEL_SHA256[name]:
                raise RuntimeError("Face model verification failed. Please retry the download.")
            os.replace(part, dest)
        finally:
            Path(part).unlink(missing_ok=True)


# ---- status note (so anyone can see how far it got, even if the PC was switched off)
class Status:
    """Keeps "_Photo Helper/Status.txt" (and status.json) up to date inside the photo folder."""
    HEAD = {"working": "NOT FINISHED. No-Nonsense Photo Sorter was working on: {d}",
            "naming": "NOT FINISHED. No-Nonsense Photo Sorter was waiting for names to be typed in. {d}",
            "saving": "NOT FINISHED. No-Nonsense Photo Sorter was saving names into the photos.",
            "closed": "NOT FINISHED. No-Nonsense Photo Sorter was closed on purpose during: {d}",
            "error": "NOT FINISHED. No-Nonsense Photo Sorter stopped because of a problem. {d}",
            "done": "FINISHED. {d}"}

    def __init__(self, root):
        self.dir = Path(root) / INFO_DIR
        self.state, self.detail = "working", "starting"
        self.last_write = 0.0
        self.lock = threading.Lock()

    def update(self, state=None, detail=None, force=False):
        if state is not None:
            self.state = state
        if detail is not None:
            self.detail = detail
        if force or time.time() - self.last_write >= 5:
            self.write()

    def write(self):
        with self.lock:
            self.last_write = time.time()
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            head = self.HEAD.get(self.state, "{d}").format(d=self.detail)
            try:
                self.dir.mkdir(exist_ok=True)
                (self.dir / "status.json").write_text(
                    json.dumps({"state": self.state, "detail": self.detail, "updated": stamp}), encoding="utf-8")
                (self.dir / "Status.txt").write_text(
                    head + f"\n\nLast update: {stamp}\n\n"
                    "If this does not say FINISHED and the last update was a while ago, No-Nonsense Photo Sorter is not\n"
                    "running any more (it was closed, or the computer was switched off).\n"
                    "Nothing is deleted either way. Run No-Nonsense Photo Sorter again, choose this same folder,\n"
                    "and it carries on where it left off.\n", encoding="utf-8")
            except OSError:
                pass


def read_status(root):
    try:
        return json.loads((Path(root) / INFO_DIR / "status.json").read_text(encoding="utf-8"))
    except Exception:
        return None


# ---- remembered faces (so a second run doesn't redo the slow part)
def face_cache_id():
    return ("faces", 2, MAX_SIDE, MIN_FACE, DET_SCORE)


def face_cache_name(ident=None):
    """The default setting keeps the file name used before; other accuracy settings each get their own file.
    `ident` is a cache header (see face_cache_id); without it the current settings are used."""
    ms, mf, ds = (ident[2], ident[3], ident[4]) if ident else (MAX_SIDE, MIN_FACE, DET_SCORE)
    if (ms, mf, ds) == (1280, 48, 0.85):
        return "faces.jsonl"
    return f"faces_{ms}_{mf}_{int(round(ds * 100))}.jsonl"


def load_face_cache(root):
    folder = Path(root) / INFO_DIR
    if list(folder.glob("faces*.cache")):
        face_report(root, "Old pickle face caches were ignored. Photos will be scanned again.")
    header, items = read_cache_file(folder / face_cache_name())
    return items if header == face_cache_id() else {}


def face_report(root, message):
    folder = Path(root) / INFO_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "Face Report.txt").open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{message}\n")


def _face_encode(value):
    if isinstance(value, np.ndarray):
        return {"array": value.tolist()}
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _face_encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_face_encode(v) for v in value]
    return value


def _face_decode(value):
    if isinstance(value, dict):
        if set(value) == {"array"}:
            return np.asarray(value["array"], dtype=np.float32)
        if set(value) == {"bytes"}:
            return base64.b64decode(value["bytes"], validate=True)
        return {k: _face_decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_face_decode(v) for v in value]
    return value


def append_face_cache(root, fresh):
    d = Path(root) / INFO_DIR
    try:
        d.mkdir(exist_ok=True)
        path = d / face_cache_name()
        new = not path.exists() or path.stat().st_size == 0
        needs_newline = False
        if not new:
            with path.open("rb") as previous:
                previous.seek(-1, os.SEEK_END)
                needs_newline = previous.read(1) != b"\n"
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(json.dumps(face_cache_id()) + "\n")
            elif needs_newline:
                f.write("\n")
            f.write(json.dumps(_face_encode(fresh), allow_nan=False) + "\n")
    except OSError:
        pass


def read_cache_file(path):
    """(header, items) of one faces cache file, or (None, {}) if it is not one. A cut-off last chunk is ignored."""
    items, head = {}, None
    try:
        with open(path, encoding="utf-8") as f:
            head = tuple(json.loads(next(f)))
            if (len(head) != 5 or head[:2] != ("faces", 2)
                    or head[2:] not in [row[3] for row in ACCURACY]):
                return None, {}
            for line in f:
                try:
                    value = _face_decode(json.loads(line))
                    if isinstance(value, dict):
                        items.update(value)
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return head, items


def merge_face_caches(files, root, progress):
    """Join JSON face caches into the cache of folder `root`. Faces are keyed by photo content,
    so they still match after folders are merged. Caches made with
    different accuracy settings stay separate (one merged file per setting). Nothing is overwritten: an existing
    cache in `root` is joined in and a backup copy of it is kept next to it.
    Returns [(file name, photos in merged cache, photos added to what was there before)]."""
    groups, skipped = {}, 0
    for n, fp in enumerate(files):
        progress(0.7 * n / max(1, len(files)), f"Reading {Path(fp).name}…")
        head, items = read_cache_file(fp)
        if head is None or not items:
            skipped += 1
            continue
        groups.setdefault(head, {}).update(items)
    if not groups:
        raise RuntimeError("None of the chosen files held remembered faces.")
    d = Path(root) / INFO_DIR
    d.mkdir(exist_ok=True)
    out = []
    for k, (head, items) in enumerate(groups.items()):
        progress(0.7 + 0.3 * k / len(groups), "Saving the merged file…")
        target = d / face_cache_name(head)
        before = {}
        if target.exists():
            old_head, before = read_cache_file(target)
            if old_head != head:
                before = {}
            shutil.copy2(target, str(target) + ".before-merge-" + uuid.uuid4().hex)
        merged = dict(before)
        merged.update({key: v for key, v in items.items() if key not in merged})
        tmp = str(target) + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(head) + "\n")
            keys = list(merged)
            for i in range(0, len(keys), 500):
                f.write(json.dumps(_face_encode({key: merged[key] for key in keys[i:i + 500]}), allow_nan=False) + "\n")
        os.replace(tmp, target)
        out.append((target.name, len(merged), len(merged) - len(before)))
    return out


# ---- copies: quick look first, careful look later
def quick_fp(path, size):
    """Fingerprint from the start, middle and end of a file (cheap). Equal fingerprints = "probably a copy"."""
    with open(path, "rb") as f:
        parts = [f.read(FP_BLOCK)]
        if size > FP_BLOCK * 3:
            f.seek(size // 2 - FP_BLOCK // 2)
            parts.append(f.read(FP_BLOCK))
        if size > FP_BLOCK:
            f.seek(max(0, size - FP_BLOCK))
            parts.append(f.read(FP_BLOCK))
    return hashlib.blake2b(b"".join(parts), digest_size=16).digest()


def run_parallel(fn, items, on_progress):
    """fn(item) for every item, a few at a time. Items that cannot be read are left out of the result."""
    out = {}
    if not items:
        return out
    ex = ThreadPoolExecutor(HASH_WORKERS)
    try:
        futs = {ex.submit(fn, it): it for it in items}
        for n, fut in enumerate(as_completed(futs), 1):
            it = futs[fut]
            try:
                out[it] = fut.result()
            except OSError:
                pass
            on_progress(n, it)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return out


def rough_copies(files, progress, lo, hi):
    """Fast first pass. Only files of the exact same size can be copies, and of those only the ones that also
    look the same at the start/middle/end. Returns (main, groups): main = one file per look-alike group
    (these go on to sorting + faces); groups = {kept file: [look-alikes]} to be checked properly later."""
    progress(lo, "Looking for copies…")
    size_of, by_size = {}, defaultdict(list)
    for p in files:
        try:
            size_of[p] = p.stat().st_size
        except OSError:
            continue
        by_size[size_of[p]].append(p)
    suspects = [p for g in by_size.values() if len(g) > 1 for p in g]
    total = max(1, len(suspects))
    fps = run_parallel(lambda p: quick_fp(p, size_of[p]), suspects,
                       lambda n, _: progress(lo + (hi - lo) * n / total, f"Looking for copies… {n} of {len(suspects)}"))
    suspect_set = set(suspects)
    main, groups, seen = [], {}, {}
    for p in files:
        if p not in size_of or (p in suspect_set and p not in fps):
            continue                                   # unreadable: leave it alone
        key = (size_of[p], fps.get(p))
        if key in seen:
            groups.setdefault(seen[key], []).append(p)
        else:
            seen[key] = p
            main.append(p)
    return main, groups


# ---- sorting and faces
class SortedPaths(list):
    failures = 0


def sort_files(files, exe, root, progress, lo, hi, preview=False):
    """Move each photo into Sorted/YEAR/YEAR-MONTH, and each meme/GIF into "Sorted Memes & GIFs"/YEAR/YEAR-MONTH,
    using the file's own date. Related media and editor sidecars move together.
    Returns (paths in the same order, meme destinations); paths.failures counts media left unsorted."""
    root = Path(os.path.abspath(str(root)))          # always the folder that was chosen, never a subfolder
    sorted_dir, memes_dir = root / "Sorted", root / MEMES_DIR
    span = hi - lo
    t0 = time.time()

    def eta(n, tot):
        if n < 100 or n >= tot:
            return ""
        left = (time.time() - t0) / n * (tot - n)
        return f" (about {max(1, round(left / 60))} min left)" if left >= 60 else ""

    progress(lo, "Reading dates…")
    cache = CopyCache(root)
    try:
        info = {norm(p): cache.get(p, "metadata", lambda _: None) for p in files}
        missing = [p for p in files if info[norm(p)] is None]
        fresh = read_info(exe, missing, lambda n, tot: progress(lo + span * 0.30 * n / max(1, tot),
                                                              f"Reading dates… {n:,} of {tot:,}")) if missing else {}
        for p in missing:
            info[norm(p)] = cache.get(p, "metadata", lambda p: fresh.get(norm(p)))
        info = {k: v for k, v in info.items() if v is not None}
        level = OPTS.get("meme_filter", "strict")
        kind = "meme:" + level + ":" + norm(root)
        known = {p: cache.get(p, kind, lambda _: None) for p in files if norm(p) in info}
        unknown = [p for p, hit in known.items() if hit is None]
        detected = classify(unknown, root, info, level, lambda n, tot: progress(lo + span * .5, f"Checking for memes… {n} of {tot}")) if unknown else set()
        for p in unknown:
            known[p] = cache.get(p, kind, lambda p: p in detected)
        memes = {p for p, hit in known.items() if hit}
    finally:
        cache.close()
    t0 = time.time()
    lo2 = lo + span * (0.65 if OPTS.get("meme_filter") != "standard" else 0.30)
    span2 = hi - lo2
    t0 = time.time()
    out, meme_dests, report = SortedPaths(files), set(), []
    out.preview_lines = []
    reserved = set()
    groups = defaultdict(list)
    for i, p in enumerate(files):
        groups[(norm(p.parent), p.stem.casefold())].append((i, p))
    directories = {}
    for n, group in enumerate(groups.values(), 1):
        progress(lo2 + span2 * n / max(1, len(groups)), f"Sorting by date… {n:,} of {len(groups):,}")
        primary = min((p for _, p in group), key=lambda p: (p.suffix.lower() not in RAW_EXT,
                                                           p.suffix.lower() in VIDEO_EXT, str(p)))
        moved = []
        try:
            if any(norm(p) not in info for _, p in group):
                raise OSError("Metadata could not be read; files left in place")
            m = info[norm(primary)][0]
            if not m:
                m = datetime.fromtimestamp(primary.stat().st_mtime).strftime("%Y-%m")
                report.append(f"DATE FALLBACK\t{primary}\tfile modification date: {m}")
            is_meme = primary in memes
            dest_dir = (memes_dir if is_meme else sorted_dir) / m[:4] / m
            if primary.parent not in directories:
                directories[primary.parent] = list(primary.parent.iterdir())
            members = [p for _, p in group]
            # Keep both stem.xmp and filename.ext.xmp conventions, plus common editor sidecars.
            for sidecar in directories[primary.parent]:
                if sidecar.suffix.lower() not in {".xmp", ".aae", ".pp3", ".dop"}:
                    continue
                if any(sidecar.stem.casefold() in (p.stem.casefold(), p.name.casefold()) for p in members):
                    members.append(sidecar)
            stem, suffix = primary.stem, 0
            while True:
                new_stem = stem + (f"_{suffix}" if suffix else "")
                plan = [(p, dest_dir / (new_stem + p.name[len(stem):])) for p in members]
                if all(norm(src) == norm(dst) or (not dst.exists() and norm(dst) not in reserved) for src, dst in plan):
                    break
                suffix += 1
            if preview:
                for src, dst in plan:
                    reserved.add(norm(dst))
                    out.preview_lines.append(f"{src.relative_to(root)} → {dst.relative_to(root)}")
            else:
                dest_dir.mkdir(parents=True, exist_ok=True)
                needed = sum(src.stat().st_size for src, dst in plan
                             if norm(src) != norm(dst) and not same_volume(src, dest_dir))
                if needed and shutil.disk_usage(dest_dir).free < needed + (64 << 20):
                    raise OSError("Not enough free space for this photo and its related files")
            for src, dst in plan:
                if not preview and norm(src) != norm(dst):
                    move_logged(src, dst, root)
                    moved.append((src, dst))
            destinations = dict(plan)
            for i, p in group:
                out[i] = destinations[p]
                if is_meme:
                    meme_dests.add(out[i])
        except (OSError, ValueError) as exc:
            out.failures += len(group)
            report.append(f"FAILED\t{primary}\t{exc}")
            for src, dst in reversed(moved):
                try:
                    move_logged(dst, src, root)
                except (OSError, ValueError) as rollback_error:
                    report.append(f"RESTORE FAILED\t{dst}\t{rollback_error}; use Undo last copy/sort moves")
            for i, p in group:
                out[i] = next((dst for src, dst in moved if src == p and not p.exists() and dst.exists()), p)
    out.preview_lines.extend(report)
    if report and not preview:
        folder = root / INFO_DIR
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "Sort Report.txt").open("a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now():%Y-%m-%d %H:%M} ---\n" + "\n".join(report) + "\n")
    return out, meme_dests


def jpeg_size(path):
    """(width, height) of a JPEG read from its header only (no decoding), or None."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                c = f.read(1)
                while c and c != b"\xff":
                    c = f.read(1)
                while c == b"\xff":
                    c = f.read(1)
                if not c:
                    return None
                m = c[0]
                if m == 0x01 or 0xd0 <= m <= 0xd8:
                    continue                                   # markers without a length
                if m in (0xd9, 0xda):
                    return None                                # reached the picture data without a size
                ln = f.read(2)
                if len(ln) < 2:
                    return None
                if m in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                    d = f.read(5)
                    if len(d) < 5:
                        return None
                    return int.from_bytes(d[3:5], "big"), int.from_bytes(d[1:3], "big")
                f.seek(int.from_bytes(ln, "big") - 2, 1)
    except OSError:
        return None


_REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


def read_for_faces(path):
    """Open a picture for face finding. Big JPEGs are opened at a fraction of their size (much faster) but never
    smaller than MAX_SIDE. Returns (picture, original width, original height) or None if it can not be opened."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    size = jpeg_size(path) if path.suffix.lower() in (".jpg", ".jpeg") else None
    if size:
        long_side = max(size)
        factor = 8 if long_side >= 8 * MAX_SIDE else 4 if long_side >= 4 * MAX_SIDE else 2 if long_side >= 2 * MAX_SIDE else 1
        if factor > 1:
            try:
                img = cv2.imdecode(data, _REDUCED[factor])
            except Exception:
                img = None
            if img is not None:
                rh, rw = img.shape[:2]
                sw, sh = size
                if sw != sh and (sw >= sh) != (rw >= rh):      # the photo was turned (EXIF): sides swap
                    sw, sh = sh, sw
                return img, sw, sh
    try:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    return img, img.shape[1], img.shape[0]


_tls = threading.local()


def _face_models():
    """Each worker thread gets its own detector + recogniser (they can not be shared between threads)."""
    if not hasattr(_tls, "det"):
        _tls.det = cv2.FaceDetectorYN.create(str(MODEL_DIR / YUNET), "", (320, 320), DET_SCORE, 0.3, 5000)
        _tls.rec = cv2.FaceRecognizerSF.create(str(MODEL_DIR / SFACE), "")
    return _tls.det, _tls.rec


def scan_one(p):
    """Faces in one picture: a list of (box, original size, embedding, quality, thumbnail), or None if unreadable."""
    got = read_for_faces(p)
    if got is None:
        return None
    img, ow, oh = got
    detector, recog = _face_models()
    s = min(1.0, MAX_SIDE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, det = detector.detect(img)
    entries = FaceResults()
    for row in (det if det is not None else []):
        x, y, fw, fh = [float(v) for v in row[:4]]
        if fw < MIN_FACE or fh < MIN_FACE:
            continue
        try:
            feat = recog.feature(recog.alignCrop(img, row)).flatten().astype(np.float32)
            if feat.shape != (128,) or not np.isfinite(feat).all() or np.linalg.norm(feat) < 1e-8:
                raise ValueError("Invalid face embedding")
        except Exception:
            entries.failures += 1
            continue
        feat /= (np.linalg.norm(feat) + 1e-9)
        left, top = max(0, x / w), max(0, y / h)
        box = (left, top, max(0, min(1, (x + fw) / w) - left), max(0, min(1, (y + fh) / h) - top))
        if box[2] <= 0 or box[3] <= 0:
            entries.failures += 1
            continue
        entries.append((box, (ow, oh), feat, float(row[14]) * fw, make_thumb(img, x, y, fw, fh)))
    return entries


class FaceResults(list):
    failures = 0


def safe_scan_one(path):
    try:
        return scan_one(path)
    except Exception:
        return None


def valid_face_entries(entries):
    try:
        return isinstance(entries, list) and all(
            len(e) == 5 and len(e[0]) == 4 and len(e[1]) == 2 and
            np.asarray(e[2]).shape == (128,) and np.isfinite(e[2]).all() and
            0.99 <= np.linalg.norm(e[2]) <= 1.01 and
            np.isfinite(e[0]).all() and all(0 <= v <= 1 for v in e[0]) and
            e[0][0] + e[0][2] <= 1.00001 and e[0][1] + e[0][3] <= 1.00001 and
            all(v > 0 for v in e[1]) and np.isfinite(e[3]) and isinstance(e[4], bytes)
            for e in entries)
    except (TypeError, ValueError):
        return False


def find_faces(paths, root, cache, progress, lo, hi, skip=()):
    """Find (or remember from an earlier run) the faces in every picture. New pictures are looked at by a few
    worker threads at once; the result keeps the order of `paths`."""
    started = time.perf_counter()
    hash_seconds, hash_bytes = 0., 0
    total = max(1, len(paths))
    fresh, last_flush = {}, time.time()

    def flush():
        nonlocal fresh, last_flush
        if fresh:
            append_face_cache(root, fresh)
            fresh = {}
        last_flush = time.time()

    found, todo, failures, skipped = {}, [], 0, 0
    for i, p in enumerate(paths):
        if p.suffix.lower() not in IMG_EXT or p in skip:      # videos / HEIC / memes: no faces
            skipped += 1
            continue
        try:
            st = p.stat()
            hash_start = time.perf_counter()
            key = sha256(p)
            hash_seconds += time.perf_counter() - hash_start
            hash_bytes += st.st_size
            after_hash = p.stat()
            if (st.st_size, st.st_mtime_ns, st.st_ctime_ns) != (after_hash.st_size, after_hash.st_mtime_ns, after_hash.st_ctime_ns):
                raise OSError("Photo changed while its identity was checked")
        except OSError:
            failures += 1
            face_report(root, f"Could not read photo: {p}")
            continue
        if key in cache and valid_face_entries(cache[key]):
            found[i] = cache[key]
        else:
            todo.append((i, p, key, st))
    done = len(paths) - len(todo)
    progress(lo + (hi - lo) * done / total, f"Looking for faces… {done} of {len(paths)}")
    cache_hits = len(found)
    scan_start = time.perf_counter()
    ex = None
    try:
        if todo:
            ex = ThreadPoolExecutor(min(FACE_WORKERS, len(todo)))
            for (i, p, key, before), entries in zip(todo, ex.map(safe_scan_one, [t[1] for t in todo])):
                done += 1
                progress(lo + (hi - lo) * done / total, f"Looking for faces… {done} of {len(paths)}")
                if entries is None:
                    failures += 1
                    face_report(root, f"Could not decode or detect faces: {p}")
                    continue
                if getattr(entries, "failures", 0):
                    failures += entries.failures
                    face_report(root, f"{entries.failures} face embeddings failed: {p}")
                try:
                    after = p.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                        raise OSError("Photo changed during detection")
                except OSError:
                    failures += 1
                    face_report(root, f"Photo changed or disappeared during detection: {p}")
                    continue
                if getattr(entries, "failures", 0):
                    found[i] = entries  # retain good faces, retry failed embeddings next run
                    continue
                found[i] = cache[key] = fresh[key] = entries
                if len(fresh) >= 200 or time.time() - last_flush > 60:
                    flush()
    except BaseException:
        if ex is not None:
            ex.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        flush()                                               # also when closed on purpose: nothing found is wasted
    if ex is not None:
        ex.shutdown()
    faces = FaceResults()
    faces.failures = failures
    for i in sorted(found):
        for box, orig, emb, quality, thumb in found[i]:
            faces.append(Face(str(paths[i]), box, orig, emb, quality, thumb))
    face_report(root, f"Scan summary: {len(found)} photos scanned or reused; "
                f"{sum(not entries for entries in found.values())} with no detected faces; "
                f"{skipped} skipped or unsupported; {failures} failures")
    timings = {"time": datetime.now().isoformat(timespec="seconds"), "files": len(paths),
               "cache_hits": cache_hits, "scan_attempts": len(todo), "hash_bytes": hash_bytes,
               "hash_seconds": hash_seconds, "detection_seconds": time.perf_counter() - scan_start,
               "total_seconds": time.perf_counter() - started, "workers": FACE_WORKERS,
               "accuracy": OPTS["accuracy"], "opencv": cv2.__version__}
    with (Path(root) / INFO_DIR / "face-timings.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(timings) + "\n")
    face_report(root, f"Performance: {timings['total_seconds']:.2f}s total; "
                f"{hash_seconds:.2f}s hashing {hash_bytes / 1048576:.1f} MB; "
                f"{timings['detection_seconds']:.2f}s detection/cache handling; "
                f"{cache_hits} reused, {len(todo)} scan attempts")
    return faces


# ---- the two stages
def stage_prepare(root, exe, progress, tasks=("copies", "sort", "faces")):
    """Quick part, done before the person starts naming: find photos, rough copy check, sort, faces.
    tasks: any mix of "copies" (copy check), "sort" (sort by date), "faces" (find faces)."""
    root = Path(os.path.abspath(str(root)))
    do_copies, do_sort, do_faces = "copies" in tasks, "sort" in tasks, "faces" in tasks
    if do_faces:
        ensure_models(progress)
    sorted_dir, memes_dir, dup_dir = root / "Sorted", root / MEMES_DIR, root / "_Duplicates"

    progress(0.01, "Finding your photos…")
    files = walk_media(root)
    files.sort(key=lambda p: (0 if (sorted_dir in p.parents or memes_dir in p.parents) else 1,
                              has_copy_suffix(p.stem), str(p)))
    if not files:
        raise RuntimeError("No photos were found in that folder.")

    if do_copies:
        main, groups = rough_copies(files, progress, 0.02, 0.08 if (do_sort or do_faces) else 1.0)
    else:
        main, groups = files, {}
    memes = set()
    if do_sort:
        dests, memes = sort_files(main, exe, root, progress, 0.08, 0.20 if do_faces else 1.0)
    else:
        dests = list(main)
        if do_faces:
            progress(0.08 if do_copies else 0.05, "Checking which files are memes…")
            memes = meme_set(dests, exe, root, lambda n, tot: progress(0.05, f"Checking for memes… {n} of {tot}"))
    cache, faces = {}, []
    if do_faces:
        cache = load_face_cache(root)
        faces = find_faces(dests, root, cache, progress, 0.20 if do_sort else 0.10, 1.0, skip=memes)
    sort_failures = getattr(dests, "failures", 0)
    return {"faces": faces, "members": cluster(faces), "photos": len(dests) - sort_failures,
        "sort_failures": sort_failures, "face_failures": getattr(faces, "failures", 0),
            "root": str(root), "memes": len(memes), "dups": 0,
            "groups": groups, "current": dict(zip(main, dests)), "cache": cache}


# ---- look-alike copies: the same picture saved again (renamed "(1)", "_1", re-saved, resized)
LOOK_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}   # pictures this check can open
VERIFY_DIR = "_Verify"         # lower-quality look-alikes wait here for a human check; Report.txt says what was kept and why
HASH_DIST = 10                 # how different the tiny "fingerprints" may be to be worth a closer look
NAMED_DIST = 20                # ...when the names say "copy" (photo.jpg / photo (1).jpg / photo_1.jpg)
_SUFFIX = re.compile(r"(\s*\(\d{1,3}\)|\s+-\s+copy(\s*\(\d+\))?|[ _\-]copy([ _\-]?\d+)?)$", re.I)
_rng = np.random.RandomState(20240229)
SAMPLE_PTS = _rng.uniform(0.06, 0.94, (15, 2))   # the same 15 spots (as fractions of the picture) in every image
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def name_base(stem):
    s = re.sub(r"^copy of\s+", "", stem.strip(), flags=re.I)
    for _ in range(3):
        t = _SUFFIX.sub("", s).rstrip()
        if t == s or len(t) < 3:
            break
        s = t
    return s.casefold()


def has_copy_suffix(stem):
    return name_base(stem) != stem.strip().casefold()


def quick_hash(path):
    """Tiny 64-bit fingerprint of how a picture looks (fast: JPEGs are opened at a fraction of their size)."""
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if img is None:
            return None
        small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        return int.from_bytes(np.packbits(bits).tobytes(), "big")
    except Exception:
        return None


def look_features(path):
    """Careful look at one picture: size, exact-pixel fingerprint, a 64x64 grey copy, and 15 spot samples."""
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if min(h, w) < 32:
            return None
        st = os.stat(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pts = []
        for fx, fy in SAMPLE_PTS:
            cx, cy = int(fx * w), int(fy * h)
            patch = img[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3].reshape(-1, 3)
            pts.append(patch.mean(axis=0))
        return {"w": w, "h": h, "size": st.st_size, "mtime": st.st_mtime_ns,
                "pix": hashlib.blake2b(np.ascontiguousarray(img), digest_size=16).digest(),
                "g64": cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16),
                "pts": np.array(pts, dtype=np.float32)}
    except Exception:
        return None


def verify_pair(a, b, named, web):
    """Return why two pictures count as the same picture, or None. named = the file names say 'copy';
    web = a PNG/WebP/BMP (often screenshots/memes, where different pictures can look alike)."""
    same_size = (a["w"], a["h"]) == (b["w"], b["h"])
    if same_size and a["pix"] == b["pix"]:
        return "identical picture (file differs)"
    if web and not named:
        return None
    if abs((a["w"] / a["h"]) / (b["w"] / b["h"]) - 1) > 0.015:       # different shape: not the same picture
        return None
    diff = np.abs(a["g64"] - b["g64"])
    mean = float(diff.mean())
    cell_max = float(diff.reshape(8, 8, 8, 8).mean(axis=(1, 3)).max())  # worst 1/64th of the picture
    spots = np.abs(a["pts"] - b["pts"]).max(axis=1)                   # the 15 spots
    if named:
        lim_mean, lim_cell, lim_spot, miss = 7.0, 22.0, 18.0, 1
    elif same_size:
        lim_mean, lim_cell, lim_spot, miss = 3.5, 12.0, 10.0, 0
    else:
        # Resizing can shift one small spot across a high-contrast edge even
        # when the overall image and every larger cell still match closely.
        lim_mean, lim_cell, lim_spot, miss = 3.0, 10.0, 10.0, 1
    if mean <= lim_mean and cell_max <= lim_cell and int((spots > lim_spot).sum()) <= miss:
        return "re-saved copy" if same_size else "resized copy"
    return None


def near_pairs(hashes, maxd, progress):
    """Bounded candidate pairs whose fingerprints differ in at most maxd bits."""
    h = np.array(hashes, dtype=np.uint64)
    n = len(h)
    block = max(1, 3_000_000 // max(1, n))
    out = []
    for s in range(0, n, block):
        progress(s / n)
        x = h[s:s + block, None] ^ h[None, :]
        d = _POP[x.view(np.uint8)].reshape(len(x), n, 8).sum(axis=2, dtype=np.uint8)
        for r in range(len(x)):                       # each picture keeps at most its 10 closest look-alikes,
            js = np.nonzero(d[r] <= maxd)[0]          # so plain/dark photos can not blow up memory
            js = js[js > r + s]
            if len(js) > 10:
                js = js[np.argsort(d[r, js], kind="stable")[:10]]
            out.extend((r + s, int(b), int(d[r, b])) for b in js)
    return out


class CopyCache:
    """Disposable visual results; exact duplicate SHA-256 checks stay uncached."""
    VERSION = "visual-1"

    def __init__(self, root):
        self.lock = threading.Lock()
        self.db = None
        try:
            folder = Path(root) / INFO_DIR
            folder.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(folder / "copies.sqlite3"), check_same_thread=False, timeout=2)
            self.db.execute("CREATE TABLE IF NOT EXISTS results (path TEXT, kind TEXT, stamp TEXT, value TEXT, "
                            "PRIMARY KEY(path, kind))")
        except (OSError, sqlite3.Error):
            self.close()

    def close(self):
        if self.db is not None:
            try:
                self.db.commit()
            except sqlite3.Error:
                pass
            finally:
                self.db.close()
                self.db = None

    @staticmethod
    def stamp(path):
        st = Path(path).stat()
        # ctime and file identity also detect replacements that retain size/mtime.
        return str((st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino))

    def get(self, path, kind, compute):
        try:
            stamp = self.stamp(path)
        except OSError:
            return None
        key = norm(Path(path).resolve())
        kind = self.VERSION + ":" + kind
        try:
            with self.lock:
                row = self.db.execute("SELECT value FROM results WHERE path=? AND kind=? AND stamp=?",
                                      (key, kind, stamp)).fetchone() if self.db else None
            if row:
                value = json.loads(row[0])
                if ":metadata" in kind:
                    if (isinstance(value, list) and len(value) == 4 and type(value[1]) is bool
                            and (value[0] is None or re.fullmatch(r"(?:19|20)\d\d-(?:0[1-9]|1[0-2])", str(value[0])))
                            and isinstance(value[2], str) and isinstance(value[3], list) and len(value[3]) == 2):
                        return value
                elif ":meme:" in kind:
                    if type(value) is bool:
                        return value
                elif kind.endswith(":hash"):
                    if type(value) is int and 0 <= value < 2**64:
                        return value
                else:
                    value["pix"] = bytes.fromhex(value["pix"])
                    value["g64"] = np.asarray(value["g64"], dtype=np.int16)
                    value["pts"] = np.asarray(value["pts"], dtype=np.float32)
                    if (value["g64"].shape == (64, 64) and value["pts"].shape == (15, 3)
                            and np.isfinite(value["pts"]).all() and len(value["pix"]) == 16
                            and all(type(value[k]) is int for k in ("w", "h", "size", "mtime"))
                            and value["w"] >= 32 and value["h"] >= 32):
                        return value
        except (sqlite3.Error, ValueError, TypeError, KeyError, OverflowError):
            pass
        value = compute(path)
        try:
            if self.stamp(path) != stamp:
                return None  # file changed during decoding; retry on the next run
            if value is not None:
                saved = value
                if isinstance(value, dict):
                    saved = dict(value, pix=value["pix"].hex(), g64=value["g64"].tolist(), pts=value["pts"].tolist())
                with self.lock:
                    if self.db:
                        self.db.execute("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?)",
                                        (key, kind, stamp, json.dumps(saved)))
        except (OSError, sqlite3.Error):
            pass  # a read-only/full drive must not prevent a copy check
        return value


def find_lookalikes(paths, root, progress):
    cache = CopyCache(root)
    try:
        return _find_lookalikes(paths, root, progress, cache)
    finally:
        cache.close()


def _find_lookalikes(paths, root, progress, cache):
    """Set aside pictures that are the same picture as another one (renamed, re-saved or resized).
    The best quality one stays where it is (in the Sorted folders); the lower quality / smaller ones are moved to
    "_Verify" to be checked. Best = largest picture, then largest file, then the one already in Sorted, then the
    one without '(1)' in its name. Returns how many were moved; _Verify/Report.txt lists each one."""
    root = Path(root)
    skip_dir = root / MEMES_DIR
    paths = [p for p in paths if p.suffix.lower() in LOOK_EXT and p.exists() and skip_dir not in p.parents]
    if len(paths) < 2:
        return 0
    progress(0.0, "Comparing pictures…")
    hs = run_parallel(lambda p: cache.get(p, "hash", quick_hash), paths,
                      lambda n, _: progress(0.30 * n / len(paths), f"Fingerprinting pictures… {n} of {len(paths)}"))
    paths = [p for p in paths if hs.get(p) is not None]
    if len(paths) < 2:
        return 0
    cand = {}
    for a, b, d in near_pairs([hs[p] for p in paths], HASH_DIST,
                              lambda f: progress(0.30 + 0.10 * f, "Looking for look-alikes…")):
        cand[(a, b)] = d
    by_name = defaultdict(list)                         # same name apart from "(1)", "_1", "- Copy" …
    for i, p in enumerate(paths):
        by_name[name_base(p.stem)].append(i)
    named_pairs = set()
    for idxs in by_name.values():
        if 1 < len(idxs) <= 60:
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    a, b = idxs[x], idxs[y]
                    if (has_copy_suffix(paths[a].stem) or has_copy_suffix(paths[b].stem)
                            or paths[a].stem.casefold() == paths[b].stem.casefold()):
                        named_pairs.add((a, b))
                        pa, pb = hs[paths[a]], hs[paths[b]]
                        if bin(pa ^ pb).count("1") <= NAMED_DIST:
                            cand.setdefault((a, b), bin(pa ^ pb).count("1"))
    if not cand:
        return 0
    # Every member of a crowded identical-fingerprint group needs a chance to
    # meet the best surviving image, even when the general candidate cap is hit.
    same_hash = defaultdict(list)
    for i, p in enumerate(paths):
        same_hash[hs[p]].append(i)
    nodes = sorted({i for pair in cand for i in pair} |
                   {i for group in same_hash.values() if len(group) > 11 for i in group})
    feats = run_parallel(lambda i: cache.get(paths[i], "features", look_features), nodes,
                         lambda n, _: progress(0.40 + 0.40 * n / len(nodes), f"Comparing look-alikes… {n} of {len(nodes)}"))
    edges = defaultdict(list)
    for (a, b), d in cand.items():
        fa, fb = feats.get(a), feats.get(b)
        if not fa or not fb:
            continue
        named = (a, b) in named_pairs
        web = paths[a].suffix.lower() in WEB_EXT or paths[b].suffix.lower() in WEB_EXT
        why = verify_pair(fa, fb, named, web)
        if why:
            edges[a].append((b, why))
            edges[b].append((a, why))
    sorted_dir = root / "Sorted"
    rank = lambda i: (-(feats[i]["w"] * feats[i]["h"]), -feats[i]["size"],
                      0 if sorted_dir in paths[i].parents else 1,
                      1 if has_copy_suffix(paths[i].stem) else 0, feats[i]["mtime"], str(paths[i]))
    removed = {}                                        # loser -> (kept, why); every loser was checked against its keeper
    for i in sorted(edges, key=rank):
        if i in removed:
            continue
        for j, why in edges[i]:
            if j not in removed and j != i:
                removed[j] = (i, why)
    # The capped candidate search can leave copies behind in large groups of
    # near-identical hashes. Check the survivors directly against each other.
    for group in same_hash.values():
        if len(group) <= 11:
            continue
        survivors = sorted((i for i in group if i in feats and i not in removed), key=rank)
        for pos, i in enumerate(survivors):
            if i in removed:
                continue
            for j in survivors[pos + 1:]:
                if j in removed:
                    continue
                named = (has_copy_suffix(paths[i].stem) or has_copy_suffix(paths[j].stem)
                         or paths[i].stem.casefold() == paths[j].stem.casefold()) and \
                        name_base(paths[i].stem) == name_base(paths[j].stem)
                web = paths[i].suffix.lower() in WEB_EXT or paths[j].suffix.lower() in WEB_EXT
                why = verify_pair(feats[i], feats[j], named, web)
                if why:
                    removed[j] = (i, why)
    if not removed:
        return 0
    out_dir = root / VERIFY_DIR
    report, moved = [], 0
    items = list(removed.items())
    for n, (j, (i, why)) in enumerate(items, 1):
        progress(0.85 + 0.15 * n / len(items), f"Setting look-alikes aside… {n} of {len(items)}")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = unique_path(out_dir / paths[j].name)
            move_logged(paths[j], dest, root)
        except OSError:
            continue
        moved += 1
        report.append(f"{dest.name}\t{why}\tkept: {paths[i]}\tfrom: {paths[j].parent}")
    if report:
        try:
            with open(out_dir / "Report.txt", "a", encoding="utf-8") as f:
                f.write("\n".join(report) + "\n")
        except OSError:
            pass
    return moved


def _exact_pass(root, exe, groups, current, cache, progress, tasks=("copies", "sort", "faces")):
    """Careful part, runs in the background while names are being typed: prove the look-alikes really are
    identical (whole file), set the real copies aside, and sort + scan any look-alike that turned out different."""
    root = Path(root)
    dup_dir = root / "_Duplicates"
    where = lambda p: current.get(p, p)
    todo = []
    for keeper, cands in groups.items():
        todo.append(where(keeper))
        todo += cands
    sizes = {}
    for p in todo:
        try:
            sizes[p] = p.stat().st_size
        except OSError:
            pass
    total, got = max(1, sum(sizes.values())), [0]

    def tick(n, p):
        got[0] += sizes.get(p, 0)
        progress(0.7 * got[0] / total, f"Checking copies… {n} of {len(todo)}")

    hashes = run_parallel(sha256, todo, tick)

    dups, late = [], []
    for keeper, cands in groups.items():
        kh, other = hashes.get(where(keeper)), {}
        for c in cands:
            h = hashes.get(c)
            if h is None:
                continue                                     # could not be read: leave it where it is
            if h == kh or h in other:
                dups.append(c)                               # truly identical: set aside
            else:
                other[h] = c
                late.append(c)                               # looked alike but is a different photo
    moved, failed = 0, 0
    for i, p in enumerate(dups):
        progress(0.7 + 0.05 * i / len(dups), f"Setting copies aside… {i + 1} of {len(dups)}")
        try:
            dup_dir.mkdir(exist_ok=True)
            move_logged(p, unique_path(dup_dir / p.name), root)
            moved += 1
        except OSError:
            failed += 1

    late_faces, late_memes, late_photos, sort_failures = [], 0, 0, 0
    late_now = list(late)                                    # where the look-alikes that stayed are now
    do_sort, do_faces = "sort" in tasks, "faces" in tasks
    if late and (do_sort or do_faces):                       # look-alikes that were different get the other ticked jobs too
        if do_sort:
            dests, memes = sort_files(late, exe, root, progress, 0.75, 0.85)
        else:
            dests, memes = list(late), meme_set(late, exe, root)
        late_now = list(dests)
        if do_faces:
            late_faces = find_faces(dests, root, cache, progress, 0.85, 1.0, skip=memes)
        sort_failures = getattr(dests, "failures", 0)
        late_memes, late_photos = len(memes), len(late) - sort_failures
    return {"dups": moved, "copy_failures": failed, "sort_failures": sort_failures,
            "face_failures": getattr(late_faces, "failures", 0),
            "late_photos": late_photos, "late_memes": late_memes, "late_faces": late_faces,
            "late_now": late_now}


def stage_deep(root, exe, groups, current, cache, progress, tasks=("copies", "sort", "faces")):
    """Background part: prove exact copies (whole file), then find look-alike copies (same picture re-saved,
    renamed or resized) and set them aside."""
    res = _exact_pass(root, exe, groups, current, cache, lambda f, t: progress(f * 0.4, t), tasks)
    survivors = [Path(p) for p in current.values()] + [Path(p) for p in res.pop("late_now")]
    res["near"] = find_lookalikes(survivors, root, lambda f, t: progress(0.4 + 0.6 * f, t))
    return res


# ---- broken / damaged files
BROKEN_DIR = "_Broken"
# exiftool warnings that really mean "damaged" (ordinary small warnings are ignored)
BAD_WARNING = re.compile(r"truncat|premature|corrupt|not a valid|unexpected end|JPEG format error", re.I)
IGNORE_ERROR = re.compile(r"unsupported|unknown file type|not supported", re.I)


def structure_problem(path, size, ext):
    """Quick look at the start and end of a file. Returns a short reason, or None if it looks fine."""
    if size == 0:
        return "empty file (0 bytes)"
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            f.seek(max(0, size - 1024))
            tail = f.read()
    except OSError:
        return "could not be read"
    if ext in (".jpg", ".jpeg"):
        if head[:2] != b"\xff\xd8":
            return "not a valid JPEG"
        if b"\xff\xd9" not in tail:
            try:
                with open(path, "rb") as stream:
                    previous = b""
                    while True:
                        block = stream.read(1024 * 1024)
                        if not block:
                            return "JPEG is cut off (incomplete file)"
                        if b"\xff\xd9" in previous + block:
                            break
                        previous = block[-1:]
            except OSError:
                return "could not be read"
    elif ext == ".png":
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return "not a valid PNG"
        if b"IEND" not in tail:
            return "PNG is cut off (incomplete file)"
    elif ext == ".gif":
        if head[:3] != b"GIF":
            return "not a valid GIF"
        if b";" not in tail:
            return "GIF is cut off (incomplete file)"
    elif ext == ".webp":
        if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
            return "not a valid WebP"
        if size < int.from_bytes(head[4:8], "little") + 8:
            return "WebP is cut off (incomplete file)"
    return None


def check_file(p):
    """Return why a file looks broken, or None. Never raises."""
    try:
        ext = p.suffix.lower()
        why = structure_problem(p, p.stat().st_size, ext)
        if why:
            return why
        if ext in IMG_EXT and cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_4) is None:
            return "cannot be opened as a picture"
    except Exception:
        return "could not be read"
    return None


def find_ffprobe():
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe.exe")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("ffprobe")


def check_video_file(path, ffprobe):
    """Return (damage reason, review note); a timeout is left for manual review."""
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
             "-of", "json", str(path)], capture_output=True, timeout=60, creationflags=NO_WINDOW)
        if result.returncode:
            return "video could not be read by ffprobe", None
        if not json.loads(result.stdout).get("streams"):
            return "no video stream was found", None
        return None, None
    except subprocess.TimeoutExpired:
        return None, f"REVIEW\t{path}\tffprobe timed out after 60 seconds"
    except (OSError, ValueError, TypeError):
        return None, f"REVIEW\t{path}\tffprobe could not complete its check"


def stage_broken(root, exe, progress):
    """Check every photo/video and move the damaged ones into _Broken (with a Report.txt saying why)."""
    root = Path(root)
    progress(0.01, "Finding your photos…")
    files = walk_media(root)
    if not files:
        raise RuntimeError("No photos were found in that folder.")
    by_norm = {norm(p): p for p in files}
    exif_bad = {}
    progress(0.02, "Reading file details…")
    for d in exif_json(exe, ["-Error", "-Warning"], files,
                       lambda n, t: progress(0.02 + 0.38 * n / t, f"Reading file details… {n} of {t}")):
        p = by_norm.get(norm(d.get("SourceFile", "")))
        if p is None:
            continue
        err, warn = " ".join(_as_list(d.get("Error"))), " ".join(_as_list(d.get("Warning")))
        if err and not IGNORE_ERROR.search(err):
            exif_bad[p] = err
        elif warn and BAD_WARNING.search(warn):
            exif_bad[p] = warn
    ffprobe = find_ffprobe()
    def check_media(p):
        problem = check_file(p)
        if problem or p.suffix.lower() not in VIDEO_EXT or not ffprobe:
            return problem, None
        return check_video_file(p, ffprobe)
    found = run_parallel(check_media, files, lambda n, _: progress(0.4 + 0.5 * n / len(files),
                                                                   f"Checking media… {n} of {len(files)}"))
    bad = {p: found[p][0] or exif_bad[p] for p in files if (p in found and found[p][0]) or p in exif_bad}
    folder = root / BROKEN_DIR
    report = [note for _, note in found.values() if note]
    if not ffprobe and any(p.suffix.lower() in VIDEO_EXT for p in files):
        report.append("REVIEW\tVideo stream checks skipped: ffprobe.exe was not found")
    moved = 0
    for i, (p, why) in enumerate(bad.items()):
        progress(0.9 + 0.1 * i / max(1, len(bad)), f"Setting broken files aside… {i + 1} of {len(bad)}")
        folder.mkdir(exist_ok=True)
        try:
            dest = unique_path(folder / p.name)
            move_logged(p, dest, root)
            report.append(f"{dest.name}\t{why}\tfrom: {p.parent}")
            moved += 1
        except OSError:
            pass
    if report:
        try:
            folder.mkdir(exist_ok=True)
            with open(folder / "Report.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {datetime.now():%Y-%m-%d %H:%M} ---\n" + "\n".join(report) + "\n")
        except OSError:
            pass
    return {"checked": len(files), "found": moved, "folder": folder}


# ---- finding photos of chosen people (names are read back out of the photos)
# ---- compress videos
COMPRESSED_DIR = "_Compressed"        # compressed copies (when the originals are kept)
UNCOMPRESSED_DIR = "_Uncompressed"    # the originals (when they are replaced)
COMPRESS_LOG = "compressed.log"       # in the notes folder: videos already done, so a second run skips them


def human_size(n):
    n = float(n)
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} bytes" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024


def load_compress_log(root):
    done = set()
    try:
        with open(Path(root) / INFO_DIR / COMPRESS_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    r, size, mt = json.loads(line)
                    done.add((r, size, mt))
                except Exception:
                    continue
    except OSError:
        pass
    return done


def add_compress_log(root, rel, size, mtime_ns):
    d = Path(root) / INFO_DIR
    try:
        d.mkdir(exist_ok=True)
        with open(d / COMPRESS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps([norm(rel), size, mtime_ns]) + "\n")
    except OSError:
        pass


def video_out_kind(src):
    """(extension, ffmpeg container) of the compressed file. mp4/m4v/mov keep their kind, the rest become mp4."""
    return (".mov", "mov") if src.suffix.lower() == ".mov" else (".mp4", "mp4")


def ffmpeg_command(ffmpeg, src, dest, fmt, encoder, crf):
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a?", "-sn", "-dn",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", encoder, "-crf", str(crf), "-preset", "fast"]
    if encoder == "libx265":
        cmd += ["-tag:v", "hvc1", "-x265-params", "log-level=error"]
    else:
        cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "160k", "-map_metadata", "0", "-movflags", "+faststart+use_metadata_tags",
            "-f", fmt, "-progress", "pipe:1", "-nostats", str(dest)]
    return cmd


def run_ffmpeg(cmd, on_frac):
    """Run ffmpeg, calling on_frac(0..1) as it goes. Returns (ok, last error line). A raise from on_frac
    (the app is closing) stops ffmpeg first."""
    log = tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False)
    log.close()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=open(log.name, "wb"), stdin=subprocess.DEVNULL,
                            creationflags=NO_WINDOW)
    duration = None
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            key, _, val = line.partition("=")
            if key not in ("out_time_us", "out_time_ms") or not val.lstrip("-").isdigit():
                continue
            if duration is None:
                try:
                    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", open(log.name, "rb").read().decode("utf-8", "replace"))
                    if m:
                        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                except OSError:
                    pass
            if duration:
                on_frac(max(0.0, min(1.0, int(val) / 1e6 / duration)))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    try:
        tail = [l for l in open(log.name, "rb").read().decode("utf-8", "replace").splitlines() if l.strip()]
    except OSError:
        tail = []
    try:
        os.unlink(log.name)
    except OSError:
        pass
    return proc.returncode == 0, (tail[-1].strip() if tail else "")


def stage_compress(root, ffmpeg, progress, quality, codec, min_mb, replace):
    """Make videos smaller. Originals are never deleted:
    replace off: compressed copies go to _Compressed (same folder layout); the originals are not touched.
    replace on:  the compressed video takes the original's place and the original is moved to _Uncompressed."""
    root = Path(os.path.abspath(str(root)))
    crf = next(r[3] for r in VIDEO_QUALITY if r[0] == quality)[0 if codec == "h265" else 1]
    encoder = next(r[3] for r in VIDEO_CODECS if r[0] == codec)
    progress(0.01, "Finding your videos…")
    vids = [p for p in walk_media(root) if p.suffix.lower() in VIDEO_EXT]
    if not vids:
        raise RuntimeError("No videos were found in that folder.")
    done_before = load_compress_log(root)
    todo, skipped = [], 0
    for p in vids:
        try:
            st = p.stat()
        except OSError:
            continue
        if (norm(p.relative_to(root)), st.st_size, st.st_mtime_ns) in done_before or st.st_size < min_mb * 1048576:
            skipped += 1
            continue
        todo.append((p, st))
    res = {"done": 0, "before": 0, "after": 0, "kept": 0, "skipped": skipped, "failed": 0, "replace": replace,
           "folder": root / (UNCOMPRESSED_DIR if replace else COMPRESSED_DIR), "checked": len(vids)}
    total = max(1, sum(st.st_size for _, st in todo))
    got, report = 0, []
    for n, (src, st) in enumerate(todo, 1):
        rel = src.relative_to(root)
        base = got
        label_ = f"Compressing videos… {n} of {len(todo)}  ({src.name})"

        def on_frac(f, base=base, size=st.st_size, label_=label_):
            progress(0.02 + 0.98 * (base + f * size) / total, f"{label_}  {int(f * 100)}%")

        on_frac(0.0)
        got += st.st_size
        ext, fmt = video_out_kind(src)
        if replace:
            part = src.with_name(src.stem + ".compressing")
        else:
            out_dir = root / COMPRESSED_DIR / rel.parent
            part = out_dir / (src.stem + ".compressing")
        if shutil.disk_usage(str(part.parent if part.parent.exists() else root)).free < st.st_size // 2 + (512 << 20):
            raise RuntimeError("There is not enough free space on the drive to carry on compressing.")
        try:
            part.parent.mkdir(parents=True, exist_ok=True)
            ok, err = run_ffmpeg(ffmpeg_command(ffmpeg, src, part, fmt, encoder, crf), on_frac)
        except Cancelled:
            try:
                part.unlink()
            except OSError:
                pass
            raise
        except OSError as e:
            ok, err = False, str(e)
        new_size = part.stat().st_size if ok and part.exists() else 0
        if not ok or new_size == 0:
            res["failed"] += 1
            report.append(f"FAILED\t{rel}\t{err[:200]}")
            part.unlink(missing_ok=True)
            continue
        if new_size >= st.st_size * 0.95:                     # not worth it: keep the original, do not try again
            res["kept"] += 1
            report.append(f"KEPT (not smaller)\t{rel}\t{human_size(st.st_size)} -> {human_size(new_size)}")
            part.unlink(missing_ok=True)
            add_compress_log(root, rel, st.st_size, st.st_mtime_ns)
            continue
        try:
            if replace:
                old_dest = unique_path(root / UNCOMPRESSED_DIR / rel)
                old_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(old_dest))          # original out of the way first ...
                final = src.with_name(src.stem + ext)
                if final.exists():
                    final = unique_path(final)
                try:
                    os.replace(str(part), str(final))         # ... then the compressed one takes its place
                except OSError:
                    shutil.move(str(old_dest), str(src))      # could not: put the original back
                    raise
            else:
                final = unique_path(part.with_name(src.stem + ext))
                os.replace(str(part), str(final))
            os.utime(str(final), ns=(st.st_atime_ns, st.st_mtime_ns))   # keep the original's date
        except OSError as e:
            res["failed"] += 1
            report.append(f"FAILED\t{rel}\t{str(e)[:200]}")
            part.unlink(missing_ok=True)
            continue
        res["done"] += 1
        res["before"] += st.st_size
        res["after"] += new_size
        report.append(f"OK\t{rel}\t{human_size(st.st_size)} -> {human_size(new_size)}")
        if replace:                                           # the new file in its place is "done" for next time
            add_compress_log(root, final.relative_to(root), new_size, final.stat().st_mtime_ns)
        else:
            add_compress_log(root, rel, st.st_size, st.st_mtime_ns)
    if report:
        try:
            (root / INFO_DIR).mkdir(exist_ok=True)
            with open(root / INFO_DIR / "Compress Report.txt", "a", encoding="utf-8") as f:
                f.write(f"--- {datetime.now():%Y-%m-%d %H:%M} ---\n" + "\n".join(report) + "\n")
        except OSError:
            pass
    return res


def walk_media(root):
    """All photos/videos under root, leaving out _Duplicates, _Broken, _Verify, _Compressed, _Uncompressed and the notes folder."""
    root = Path(root)
    skip = {root / "_Duplicates", root / BROKEN_DIR, root / VERIFY_DIR, root / COMPRESSED_DIR, root / UNCOMPRESSED_DIR}
    files = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if Path(dp, d) not in skip and d != INFO_DIR]
        for f in fn:
            if Path(f).suffix.lower() in MEDIA_EXT:
                files.append(Path(dp) / f)
    return files


def _as_list(v):
    if v is None:
        return []
    return [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]


def read_people(exe, files, progress, lo, hi):
    """Return {path: set of names} read from the photos (the names this app saved earlier)."""
    by_norm = {norm(p): p for p in files}
    out = {p: set() for p in files}
    progress(lo, "Reading names…")
    for d in exif_json(exe, ["-XMP-iptcExt:PersonInImage", "-XMP-mwg-rs:RegionName"], files,
                       lambda n, t: progress(lo + (hi - lo) * n / t, f"Reading names… {n} of {t}")):
        p = by_norm.get(norm(d.get("SourceFile", "")))
        if p is not None:
            out[p].update(_as_list(d.get("PersonInImage")))
            out[p].update(_as_list(d.get("RegionName")))
    return out


def stage_people(root, exe, progress):
    progress(0.01, "Finding your photos…")
    files = [p for p in walk_media(root) if p.suffix.lower() in PEOPLE_EXT]
    if not files:
        raise RuntimeError("No photos were found in that folder.")
    index = read_people(exe, files, progress, 0.03, 1.0)
    counts = Counter(n for names in index.values() for n in names)
    return {"index": index, "counts": counts}


def pick_photos(index, chosen, match_all, only_chosen):
    """Photos with any (or all) of the chosen people; optionally only when nobody else is named in them."""
    chosen = set(chosen)
    out = []
    for p, names in index.items():
        if not names:
            continue
        hit = chosen <= names if match_all else bool(chosen & names)
        if hit and (not only_chosen or names <= chosen):
            out.append(p)
    return sorted(out, key=str)


def make_grid_thumbs(paths, progress):
    """Small previews for the picture grid. Returns (path, jpeg bytes or None) for each (None = cannot preview)."""
    out = []
    for i, p in enumerate(paths):
        progress(i / max(1, len(paths)), f"Making previews… {i + 1} of {len(paths)}")
        data = None
        if p.suffix.lower() in IMG_EXT:
            try:
                img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
                if img is not None:
                    h, w = img.shape[:2]
                    s = 300 / max(h, w)
                    if s < 1:
                        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                    data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            except Exception:
                data = None
        out.append((p, data))
    return out


def people_folder_name(chosen):
    name = " & ".join(sorted(chosen, key=str.casefold))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name).strip(" .")
    return name[:120] or "People"


def file_photos(paths, root, chosen, progress):
    """Copy photos into <root>/<names>, next to Sorted. The originals are never moved. Later scans treat this folder like any other."""
    folder = Path(root) / people_folder_name(chosen)
    folder.mkdir(exist_ok=True)
    done = skipped = 0
    for i, p in enumerate(paths):
        progress(i / max(1, len(paths)), f"Copying… {i + 1} of {len(paths)}")
        dest = folder / p.name
        try:
            if dest.exists() and dest.stat().st_size == p.stat().st_size:
                skipped += 1                                  # already copied on an earlier run
                continue
            dest = unique_path(dest)
            shutil.copy2(str(p), str(dest))
            done += 1
        except OSError:
            skipped += 1
    return {"folder": folder, "done": done, "skipped": skipped}


def cluster(faces):
    n = len(faces)
    if not n:
        return []
    order = sorted(range(n), key=lambda i: -faces[i].quality)
    C = np.zeros((n, 128), dtype=np.float32)
    S = np.zeros((n, 128), dtype=np.float32)
    members, k = [], 0
    for i in order:
        e = faces[i].emb
        if k:
            sims = C[:k] @ e
            j = int(sims.argmax())
            if sims[j] >= CLUSTER_T:
                members[j].append(i)
                S[j] += e
                C[j] = S[j] / (np.linalg.norm(S[j]) + 1e-9)
                continue
        C[k] = e
        S[k] = e
        members.append([i])
        k += 1
    members.sort(key=len, reverse=True)
    return members


def pick_samples(faces, idxs, n=None):
    n = n or SAMPLES
    pool = sorted(idxs, key=lambda i: -faces[i].quality)[:40]
    chosen = [pool[0]]
    while len(chosen) < min(n, len(pool)):
        best, best_s = None, 2.0
        for i in pool:
            if i in chosen:
                continue
            s = max(float(faces[i].emb @ faces[c].emb) for c in chosen)
            if s < best_s:
                best, best_s = i, s
        chosen.append(best)
    return chosen


def esc(s):
    for ch in "\\,[]{}|":
        s = s.replace(ch, "\\" + ch)
    return s


FACE_REMOVE_TAGS = ("XMP-mwg-rs:RegionInfo", "XMP-MP:RegionInfoMP", "XMP-acdsee-rs:RegionInfoACDSee",
                    "XMP-iptcExt:PersonInImage", "XMP-iptcExt:PersonInImageWDetails")
FACE_REMOVE_KEYWORDS = ("XMP-dc:Subject", "XMP-digiKam:TagsList", "XMP-lr:HierarchicalSubject", "IPTC:Keywords")


def face_removal_values(record):
    names, keywords = set(), {}
    def collect(value):
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ("Name", "PersonDisplayName", "PersonName"):
                    collect(item)
                elif isinstance(item, (dict, list)):
                    collect(item)
    for tag in FACE_REMOVE_TAGS:
        collect(record.get(tag))
    for tag in FACE_REMOVE_KEYWORDS:
        values = _as_list(record.get(tag))
        for value in values:
            if value.casefold().startswith(("people/", "people|")):
                names.add(value[7:])
        keywords[tag] = values
    remove = {tag: [v for v in values if v in names or v.casefold().startswith(("people/", "people|"))]
              for tag, values in keywords.items()}
    affected = any(record.get(tag) for tag in FACE_REMOVE_TAGS) or any(remove.values()) or any(
        key.startswith("XMP-apple-fi:") for key in record)
    return affected, remove


def prepare_face_removal(root, exe, progress):
    root = Path(root).resolve()
    files, notes, stamps = [], [], {}
    for directory, dirs, names in os.walk(root):
        if INFO_DIR in dirs:
            notes.append(str(Path(directory) / INFO_DIR))
        dirs[:] = [d for d in dirs if d != INFO_DIR and not Path(directory, d).is_symlink()]
        for name in names:
            path = Path(directory, name)
            if path.suffix.lower() not in PEOPLE_EXT | {".xmp"} or path.is_symlink():
                continue
            if not path.resolve().is_relative_to(root):
                continue
            files.append(path)
            stamps[norm(path)] = CopyCache.stamp(path)
    progress(0, "Checking images for face metadata…")
    records = exif_json(exe, ["-G1", "-a", "-struct"] + ["-" + tag for tag in FACE_REMOVE_TAGS + FACE_REMOVE_KEYWORDS]
                        + ["-XMP-apple-fi:all"], files,
                        lambda n, t: progress(n / max(1, t), f"Checking face metadata… {n} of {t}"))
    by_path = {norm(p): p for p in files}
    entries, unreadable, seen = [], [], set()
    for record in records:
        key = norm(record.get("SourceFile", ""))
        if key not in by_path:
            continue
        seen.add(key)
        if record.get("ExifTool:Error") or record.get("Error"):
            unreadable.append(str(by_path[key]))
            continue
        affected, remove = face_removal_values(record)
        if affected:
            entries.append({"path": str(by_path[key]), "stamp": stamps[key], "remove": remove})
    unreadable.extend(str(p) for key, p in by_path.items() if key not in seen)
    return {"root": str(root), "entries": entries, "unreadable": unreadable, "checked": len(files), "notes": notes}


def remove_face_data(plan, exe, progress, forget=True):
    root = Path(plan["root"]).resolve()
    backup_root = root / INFO_DIR / "face-removal-backups" / uuid.uuid4().hex
    result = {"removed": 0, "failed": len(plan["unreadable"]), "backups": str(backup_root), "forgotten": 0}
    report = [f"UNREADABLE\t{path}" for path in plan["unreadable"]]
    entries = plan["entries"]
    timings = {}
    # 1) check each file is still the one previewed and build its edit (no exiftool run yet)
    started = time.perf_counter()
    pending = []
    for i, entry in enumerate(entries):
        progress(i / max(1, len(entries)), f"Preparing face data removal… {i+1} of {len(entries)}")
        path = Path(entry["path"])
        try:
            if not path.resolve().is_relative_to(root) or CopyCache.stamp(path) != entry["stamp"]:
                raise OSError("Image changed since preview; left unchanged")
            # Make and verify a copy before allowing ExifTool to edit the image.
            backup = backup_root / "images" / path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            partial = backup.with_name(backup.name + ".partial")
            try:
                shutil.copy2(path, partial)
                if sha256(path) != sha256(partial) or CopyCache.stamp(path) != entry["stamp"]:
                    raise OSError("Image changed while being backed up; left unchanged")
                os.replace(partial, backup)
            finally:
                partial.unlink(missing_ok=True)
            args = ["-P", "-overwrite_original"] + ["-" + tag + "=" for tag in FACE_REMOVE_TAGS] + ["-XMP-apple-fi:all="]
            for tag, values in entry["remove"].items():
                for value in values:
                    if "\n" in value or "\r" in value:
                        raise ValueError("Metadata value contains line breaks; manual cleanup required")
                    args.append(f"-{tag}-={value}")
            pending.append((entry, path, args))
        except (OSError, ValueError, RuntimeError) as exc:
            result["failed"] += 1
            report.append(f"FAILED\t{path}\t{exc}")
    timings["prepare"] = time.perf_counter() - started
    # 2) one exiftool run per chunk of files: all edits go into one argument file, with "-execute" between files.
    #    The exit code is not trusted per file; step 3 checks each file, so a failed edit is reported there.
    started = time.perf_counter()
    for start in range(0, len(pending), EXIF_CHUNK):
        chunk = pending[start:start + EXIF_CHUNK]
        progress(start / max(1, len(pending)), f"Removing face data… {start + len(chunk)} of {len(pending)}")
        lines = []
        for _, path, args in chunk:
            lines += args + [str(path), "-execute"]
        try:
            run_exif(exe, lines, tolerate_file_errors=True)
        except RuntimeError as exc:
            report.append(f"EXIFTOOL WARNING\t{exc}")
    timings["edit"] = time.perf_counter() - started
    # 3) one batched read verifies every file that was sent for editing
    started = time.perf_counter()
    progress(0.9, "Checking removed face data…")
    by_key = {norm(path): (entry, path) for entry, path, _ in pending}
    records = exif_json(exe, ["-G1", "-a", "-struct"] + ["-" + tag for tag in FACE_REMOVE_TAGS + FACE_REMOVE_KEYWORDS]
                        + ["-XMP-apple-fi:all"], [path for _, path, _ in pending])
    seen = set()
    for record in records:
        key = norm(record.get("SourceFile", ""))
        if key not in by_key:
            continue
        seen.add(key)
        entry, path = by_key[key]
        if (record.get("ExifTool:Error") or record.get("Error") or face_removal_values(record)[0] or
                any(value in _as_list(record.get(tag)) for tag, values in entry["remove"].items() for value in values)):
            result["failed"] += 1
            report.append(f"FAILED\t{path}\tFace metadata remains after editing")
        else:
            result["removed"] += 1
            report.append(f"REMOVED\t{path}")
    for key, (entry, path) in by_key.items():
        if key not in seen:
            result["failed"] += 1
            report.append(f"FAILED\t{path}\tCould not verify the edit")
    timings["verify"] = time.perf_counter() - started
    report.append("TIMING\t" + ", ".join(f"{name} {secs:.1f}s" for name, secs in timings.items())
                  + f", files {len(entries)}")
    if forget:
        for directory in plan["notes"]:
            directory = Path(directory)
            if not directory.resolve().is_relative_to(root):
                continue
            state = list(directory.glob("faces*.jsonl*")) + list(directory.glob("faces*.cache*"))
            state += [directory / "people.json", directory / "face-calibration.json"]
            for path in state:
                if path.is_file() and not path.is_symlink():
                    try:
                        dest = backup_root / "remembered-data" / path.relative_to(root)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        path.rename(dest)
                        result["forgotten"] += 1
                    except OSError as exc:
                        result["failed"] += 1
                        report.append(f"COULD NOT FORGET\t{path}\t{exc}")
    folder = root / INFO_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "Face Removal Report.txt").open("a", encoding="utf-8") as f:
        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M} ---\n" + "\n".join(report) + "\n")
    progress(1, "Face data removal finished")
    return result


def people_signature(refs):
    content = {name: [np.asarray(e, dtype=np.float32).tolist() for e in samples]
               for name, samples in sorted(refs.items())}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def prepare_face_rescan(root, exe, progress):
    data = stage_prepare(root, exe, progress, ("faces",))
    data["rename_all"] = True
    return data


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_people(root):
    try:
        value = json.loads((Path(root) / INFO_DIR / "people.json").read_text(encoding="utf-8"))
        if value["version"] != 1 or value["model"] != MODEL_SHA256[SFACE]:
            return {}
        result = {}
        for name, samples in value["people"].items():
            if not isinstance(name, str) or not name.strip() or len(name) > 100 or any(ord(c) < 32 for c in name):
                continue
            valid = []
            for sample in samples[:32]:
                e = np.asarray(sample, dtype=np.float32)
                if e.shape == (128,) and np.isfinite(e).all() and .99 <= np.linalg.norm(e) <= 1.01:
                    valid.append(e)
            if valid:
                result[name] = valid
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def remember_person(root, name, embedding):
    """Only call after a human confirms the identity, never for inferred labels."""
    refs = load_people(root)
    name = next((n for n in refs if n.casefold() == name.casefold()), name.strip())
    if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
        raise ValueError("Use a name of 1–100 characters with no control characters")
    e = np.asarray(embedding, dtype=np.float32)
    if e.shape != (128,) or not np.isfinite(e).all() or not .99 <= np.linalg.norm(e) <= 1.01:
        raise ValueError("Invalid approved face sample")
    # A correction removes the exact old sample from the previously assigned person.
    for other in list(refs):
        refs[other] = [old for old in refs[other] if not np.allclose(old, e, atol=1e-6)]
        if not refs[other]:
            del refs[other]
    refs.setdefault(name, []).append(e)
    refs[name] = refs[name][-32:]
    atomic_json(Path(root) / INFO_DIR / "people.json",
                {"version": 1, "model": MODEL_SHA256[SFACE],
                 "people": {n: [v.tolist() for v in vectors] for n, vectors in refs.items()}})
    return refs


def score_face(embedding, refs):
    return sorted(((max(float(np.dot(e, embedding)) for e in samples), name)
                   for name, samples in refs.items() if samples), reverse=True)


def predicted_name(embedding, refs, threshold):
    scores = score_face(embedding, refs)
    if not scores or scores[0][0] < threshold or (len(scores) > 1 and scores[0][0] - scores[1][0] < .05):
        return None
    return scores[0][1]


def calibrate_people(refs):
    """Leave each approved sample out of its own match; never use self similarity."""
    if len(refs) < 2 or any(len(samples) < 3 for samples in refs.values()):
        raise ValueError("Confirm at least three different face samples for each of at least two people first.")
    genuine, impostor = [], []
    for name, samples in refs.items():
        for i, e in enumerate(samples):
            genuine.append(max(float(e @ other) for j, other in enumerate(samples) if j != i))
            impostor.append(max(float(e @ other) for n, vectors in refs.items() if n != name for other in vectors))
    threshold = max(MATCH_T, max(impostor) + .05)
    if threshold > .95:
        raise ValueError("Approved samples overlap too strongly. Review the names and add clearer examples.")
    return {"version": 1, "model": MODEL_SHA256[SFACE], "signature": people_signature(refs),
            "threshold": threshold, "samples": len(genuine), "people": len(refs),
            "same_person_min": min(genuine), "different_person_max": max(impostor),
            "sample_accept_rate": sum(s >= threshold for s in genuine) / len(genuine)}


def matching_threshold(root, refs):
    if root and OPTS["use_calibration"]:
        try:
            result = json.loads((Path(root) / INFO_DIR / "face-calibration.json").read_text(encoding="utf-8"))
            threshold = float(result["threshold"])
            if (result["model"] == MODEL_SHA256[SFACE] and result["signature"] == people_signature(refs)
                    and np.isfinite(threshold) and 0 <= threshold <= .95):
                return max(MATCH_T, threshold)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return MATCH_T


def stage_match(data, refs, cluster_names, exe, progress):
    faces, members = data["faces"], data["members"]
    if not refs:
        return {"tagged": 0, "people": []}
    threshold = data.get("threshold", matching_threshold(data.get("root"), refs))
    per_image, uncertain = {}, 0
    for i, f in enumerate(faces):
        name = data.get("confirmed", {}).get(i) if i in data.get("confirmed", {}) else predicted_name(f.emb, refs, threshold)
        if not name:
            uncertain += 1
        if name:
            per_image.setdefault(f.path, []).append((name, f))

    paths = [p for p in per_image if os.path.exists(p)]      # a look-alike may have been set aside meanwhile
    root = Path(data.get("root") or os.path.commonpath([str(Path(p).parent) for p in paths])) if paths else None
    tagged, failed, people = 0, 0, set()
    for index, path in enumerate(paths):
        progress(index / max(1, len(paths)), "Saving names into your photos…")
        try:
            items = per_image[path]
            ow, oh = items[0][1].orig
            current = exif_json(exe, ["-struct", "-XMP-mwg-rs:RegionInfo", "-XMP-iptcExt:PersonInImage",
                                      "-XMP-dc:Subject", "-XMP-digiKam:TagsList", "-XMP-lr:HierarchicalSubject"], [path])
            if len(current) != 1 or current[0].get("Error"):
                raise RuntimeError("Existing annotations could not be read")
            current = current[0]
            existing = current.get("RegionInfo", {})
            old_regions = existing.get("RegionList", []) if isinstance(existing, dict) else []
            if isinstance(old_regions, dict):
                old_regions = [old_regions]
            dims = existing.get("AppliedToDimensions", {}) if isinstance(existing, dict) else {}
            if old_regions and (dims.get("W"), dims.get("H")) != (ow, oh):
                raise RuntimeError("Existing face regions use different dimensions; annotations left unchanged")
            regions = []
            accepted = []
            for nm, f in items:
                x, y, w, h = f.box
                overlaps = []
                for old in old_regions:
                    area = old.get("Area", {})
                    if area.get("Unit") != "normalized":
                        raise RuntimeError("Existing face region units require manual review")
                    ox, oy, bw, bh = (float(area[k]) for k in ("X", "Y", "W", "H"))
                    intersection = max(0, min(x+w, ox+bw/2)-max(x, ox-bw/2)) * max(0, min(y+h, oy+bh/2)-max(y, oy-bh/2))
                    if intersection / max(1e-9, min(w*h, bw*bh)) > 0.5:
                        overlaps.append(old)
                if overlaps:
                    if all(old.get("Name") == nm for old in overlaps):
                        accepted.append((nm, f))
                    else:
                        face_report(root, f"Kept existing face annotation instead of predicted {nm}: {path}")
                    continue
                accepted.append((nm, f))
                regions.append("{Area={X=%.5f,Y=%.5f,W=%.5f,H=%.5f,Unit=normalized},Name=%s,Type=Face}"
                               % (x + w / 2, y + h / 2, w, h, esc(nm)))
            lines = ["-charset", "utf8", "-charset", "filename=utf8", "-P"]
            for nm in sorted({n for n, _ in accepted}):
                safe = nm.replace("|", " ").replace("/", " ")
                for tag, key, value in (("XMP-iptcExt:PersonInImage", "PersonInImage", nm),
                                        ("XMP-dc:Subject", "Subject", nm),
                                        ("XMP-digiKam:TagsList", "TagsList", f"People/{safe}"),
                                        ("XMP-lr:HierarchicalSubject", "HierarchicalSubject", f"People|{safe}")):
                    if value not in _as_list(current.get(key)):
                        lines.append(f"-{tag}+={value}")
            if regions:
                if not old_regions:
                    lines.append(f"-XMP-mwg-rs:RegionAppliedToDimensions={{W={ow},H={oh},Unit=pixel}}")
                lines.extend("-XMP-mwg-rs:RegionList+=" + region for region in regions)
            if len(lines) == 5:
                continue
            backup = unique_path(root / INFO_DIR / "name-backups" / Path(path).relative_to(root))
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)  # backup exists before any metadata write
            original = Path(path + "_original")
            had_original = original.exists()
            run_exif(exe, lines + [path])
            if original.exists() and not had_original:
                original.rename(unique_path(backup.with_name(backup.name + "_exiftool_original")))
            tagged += 1
            people.update(n for n, _ in accepted)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            failed += 1
            face_report(root, f"Could not save names: {path}: {exc}")
    return {"tagged": tagged, "people": sorted(people), "failed": failed, "uncertain": uncertain}


# ------------------------------------------------------------------ UI
STYLE = """
QWidget { font-family: "PH Sans", "Segoe UI", Arial; color: #EDE9E1; }
#root, QDialog, QMessageBox { background: #0A0A0A; }
#content { background: #121212; }
#title { font-family: "PH Serif", "Georgia", serif; font-size: 46px; color: #EDE9E1; }
#welcomeTitle { font-family: "PH Serif", "Georgia", serif; font-size: 36px; color: #EDE9E1; }
#subtitle { font-size: 15px; color: #A8A39A; }
#caption { font-size: 12px; color: #85817B; }
#version { font-size: 12px; color: #85817B; }
#status { font-family: "PH Sans Medium", "PH Sans", "Segoe UI"; font-size: 16px; color: #EDE9E1; }
#hint { font-size: 13px; color: #85817B; }
#card { background: #181818; border: 1px solid #2A2A2A; border-radius: 14px; }
#rule { background: #9C7C49; border: none; }
QMessageBox QLabel { color: #EDE9E1; font-size: 14px; }
QPushButton { background: #181818; color: #EDE9E1; border: 1px solid #2A2A2A; border-radius: 10px;
              padding: 9px 18px; font-family: "PH Sans Medium", "PH Sans", "Segoe UI"; font-size: 13px; }
QPushButton:hover { background: #1F1F1F; }
QPushButton:pressed { background: #101010; }
QPushButton:disabled { color: #6F6A63; }
QPushButton#primary { background: #B08D57; color: #0A0A0A; border: none; border-radius: 12px;
                      padding: 14px 40px; font-family: "PH Sans Semibold", "PH Sans", "Segoe UI"; font-size: 15px; }
QPushButton#primary:hover { background: #C49A5C; }
QPushButton#primary:pressed { background: #8E6E3B; }
QPushButton#primary:disabled { background: #1F1F1F; color: #6F6A63; }
QPushButton#link { background: transparent; border: none; color: #A8A39A; font-size: 14px; padding: 8px 14px; }
QPushButton#link:hover { color: #EDE9E1; }
QPushButton#secondary { background: #181818; color: #EDE9E1; border: 1px solid #2A2A2A; border-radius: 12px;
                        padding: 13px 28px; font-family: "PH Sans Semibold", "PH Sans", "Segoe UI"; font-size: 14px; }
QPushButton#secondary:hover { background: #1F1F1F; border: 1px solid #B08D57; }
QLineEdit { background: #121212; color: #EDE9E1; border: 1px solid #2A2A2A; border-radius: 12px;
            padding: 12px 16px; font-size: 16px; selection-background-color: #B08D57; selection-color: #0A0A0A; }
QLineEdit:focus { border: 1px solid #B08D57; }
QCheckBox { font-size: 15px; spacing: 12px; padding: 0; min-height: 28px; background: transparent; }
QCheckBox#allbox { font-family: "PH Sans Semibold", "PH Sans", "Segoe UI"; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; background: #121212; border: 1px solid #3A3A3A; }
QCheckBox::indicator:hover { border: 1px solid #9C7C49; }
QCheckBox::indicator:checked { background: #B08D57; border: 1px solid #B08D57; }
QCheckBox::indicator:checked:hover { background: #C49A5C; border: 1px solid #C49A5C; }
QCheckBox::indicator:disabled { background: #101010; border: 1px solid #232323; }
QCheckBox::indicator:checked:disabled { background: #6B5633; border: 1px solid #6B5633; }
QListWidget { background: #121212; color: #EDE9E1; border: 1px solid #2A2A2A; border-radius: 12px; font-size: 15px; padding: 6px; }
QListWidget#grid { background: transparent; border: none; font-size: 12px; }
QComboBox { background: #121212; color: #EDE9E1; border: 1px solid #2A2A2A; border-radius: 10px;
            padding: 9px 14px; font-size: 14px; min-width: 300px; }
QComboBox:hover, QComboBox:focus { border: 1px solid #B08D57; }
QComboBox#optcombo { min-width: 0px; font-size: 13px; padding: 7px 12px; }
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView { background: #181818; color: #EDE9E1; border: 1px solid #2A2A2A;
                              selection-background-color: #B08D57; selection-color: #0A0A0A; outline: none; font-size: 14px; }
QSpinBox { background: #121212; color: #EDE9E1; border: 1px solid #2A2A2A; border-radius: 8px;
           padding: 5px 8px; font-size: 14px; min-width: 56px; }
QSpinBox:focus { border: 1px solid #B08D57; }
#optpage { background: #121212; }
QPushButton#seg { background: transparent; border: 1px solid #2A2A2A; color: #A8A39A; border-radius: 10px;
                  padding: 8px 16px; font-family: "PH Sans Medium", "PH Sans", "Segoe UI"; font-size: 13px; }
QPushButton#seg:hover { color: #EDE9E1; border: 1px solid #3A3A3A; }
QPushButton#seg:checked { background: #181818; color: #EDE9E1; border: 1px solid #B08D57; }
#sidebar { background: #101010; border-right: 1px solid #2A2A2A; }
#brand { font-family: "PH Serif", "Georgia", serif; font-size: 26px; color: #EDE9E1; }
QPushButton#nav { background: transparent; border: none; color: #A8A39A; text-align: left;
                  padding: 10px 14px; font-family: "PH Sans Medium", "PH Sans", "Segoe UI"; font-size: 14px; border-radius: 10px; }
QPushButton#nav:hover { background: #1F1F1F; color: #EDE9E1; }
QPushButton#nav:checked { background: #181818; color: #EDE9E1; border-left: 2px solid #B08D57; }
QPushButton#nav:disabled { color: #6F6A63; }
#opttitle { font-family: "PH Heading", "Georgia", serif; font-size: 22px; }
#opttitle2 { font-family: "PH Heading", "Georgia", serif; font-size: 16px; }
#optscroll, #optinner { background: #121212; border: none; }
#optname { font-family: "PH Sans Semibold", "PH Sans", "Segoe UI"; font-size: 14px; }
#opthint { font-size: 12px; color: #85817B; }
QProgressBar { background: #1F1F1F; border: none; border-radius: 4px; }
QProgressBar::chunk { background: #B08D57; border-radius: 4px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #2A2A2A; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


def numbered_photo(path, faces, width, max_height=None, selected=None):
    """The whole picture, scaled to fit `width` and `max_height` (keeping its shape), with each face boxed and numbered.
    faces: [(number, (left, top, w, h))] with the box as fractions of the picture. selected: the number of the face
    being named; its box is drawn bright and thicker. Returns None if it can not be read."""
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    s = min(1.0, width / w0, (max_height / h0) if max_height else 1.0)
    img = cv2.resize(img, (max(1, int(w0 * s)), max(1, int(h0 * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    gold = (74, 162, 201)                                    # BGR
    bright = (0, 255, 255)
    for number, (left, top, bw, bh) in faces:
        x1, y1 = int(left * w), int(top * h)
        x2, y2 = int((left + bw) * w), int((top + bh) * h)
        on = number == selected
        colour = bright if on else gold
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, max(2, w // 300) + (3 if on else 0))
        cv2.putText(img, str(number), (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, w / 800),
                    colour, max(2, w // 350) + (1 if on else 0), cv2.LINE_AA)
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class FocusWatcher(QObject):
    """Calls `on_focus` whenever the watched name box gets the keyboard, so the matching face is drawn in the preview."""
    def __init__(self, on_focus, parent=None):
        super().__init__(parent)
        self.on_focus = on_focus

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            self.on_focus()
        return False


def rounded_pixmap(data, size, radius):
    pm = QPixmap.fromImage(QImage.fromData(data)).scaled(size, size, Qt.KeepAspectRatioByExpanding,
                                                          Qt.SmoothTransformation)
    out = QPixmap(size, size)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, size, size, radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pm)
    p.end()
    return out


def camera_pixmap(size):
    """A light camera on a gold rounded square, drawn at any size."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 100.0, size / 100.0)
    blue, white = QColor("#B08D57"), QColor("#EDE9E1")
    p.setPen(Qt.NoPen)
    p.setBrush(blue)
    p.drawRoundedRect(2, 2, 96, 96, 22, 22)
    p.setBrush(white)
    p.drawRoundedRect(16, 29, 26, 12, 4, 4)           # top bump (viewfinder)
    p.drawRoundedRect(12, 35, 76, 48, 10, 10)         # camera body
    p.setBrush(blue)
    p.drawEllipse(31, 40, 38, 38)                     # lens ring
    p.setBrush(white)
    p.drawEllipse(36, 45, 28, 28)
    p.setBrush(blue)
    p.drawEllipse(41, 50, 18, 18)                     # lens glass
    p.setBrush(white)
    p.drawEllipse(45, 54, 6, 6)                       # glint
    p.setBrush(blue)
    p.drawEllipse(74, 42, 6, 6)                       # flash dot
    p.end()
    return pm


def app_icon():
    branded = QIcon(str(RES_DIR / "assets" / "nnps_icon.png"))
    if not branded.isNull():
        return branded
    ic = QIcon()
    for sz in (16, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(camera_pixmap(sz))
    return ic


class Worker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = False

    def run(self):
        def prog(f, t):
            if self.cancel:
                raise Cancelled()
            self.progress.emit(float(f), t)
        try:
            self.done.emit(self.fn(prog))
        except Cancelled:
            pass
        except Exception as e:
            self.failed.emit(str(e) if isinstance(e, RuntimeError) else traceback.format_exc())


def label(text, name=None, align=Qt.AlignCenter, wrap=True):
    l = QLabel(text)
    if name:
        l.setObjectName(name)
    l.setAlignment(align)
    l.setWordWrap(wrap)
    return l


def keep_awake(on):
    """Stop Windows going to sleep by itself while No-Nonsense Photo Sorter is busy."""
    if sys.platform != "win32":
        return
    try:
        f = ctypes.windll.kernel32.SetThreadExecutionState
        f.argtypes, f.restype = [ctypes.c_uint], ctypes.c_uint
        f(0x80000001 if on else 0x80000000)          # ES_CONTINUOUS (+ ES_SYSTEM_REQUIRED)
    except Exception:
        pass


def shutdown_note(hwnd, text):
    """Windows shows this text on its "this app is preventing shutdown" screen. text=None removes it."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import wintypes
        u = ctypes.windll.user32
        u.ShutdownBlockReasonCreate.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        u.ShutdownBlockReasonDestroy.argtypes = [wintypes.HWND]
        if text:
            u.ShutdownBlockReasonCreate(hwnd, text)
        else:
            u.ShutdownBlockReasonDestroy(hwnd)
    except Exception:
        pass


CLOSE_WARNING = {
    "removing_faces": ("No-Nonsense Photo Sorter is removing face data from images.",
                       "Some images may already be cleaned and others may still contain face data. "
                       "Run removal again to finish."),
    "scan": ("No-Nonsense Photo Sorter is still getting your photos ready.",
             "• It will stop where it is. Nothing is lost or deleted.\n"
             "• Photos already tidied stay where they are.\n"
             "• Next time, choose the same folder and it carries on. Faces it already found are remembered."),
    "naming": ("No-Nonsense Photo Sorter is waiting for you to name the people.",
               "• Names typed so far will NOT be saved into the photos (that happens at the very end).\n"
               "• {copies}\n"
               "• Nothing is deleted. Next time, choose the same folder and it carries on; "
               "you would only need to type the names again."),
    "waiting": ("No-Nonsense Photo Sorter is finishing its check for copies, then it saves the names.",
                "• Names typed so far will NOT be saved into the photos.\n"
                "• {copies}\n"
                "• Nothing is deleted. Next time, choose the same folder and it carries on."),
    "saving": ("No-Nonsense Photo Sorter is saving the names into your photos right now.",
               "• Some photos will have their names and some will not.\n"
               "• Nothing is damaged or deleted.\n"
               "• Next time, choose the same folder and type the names again to finish."),
    "filing": ("No-Nonsense Photo Sorter is copying photos into the person's folder right now.",
               "• Some photos will already be in the folder and some will not.\n"
               "• Nothing is damaged or deleted.\n"
               "• Run it again to finish; photos already copied are skipped."),
    "compressing": ("No-Nonsense Photo Sorter is compressing videos right now.",
                    "• The video being compressed at this moment is stopped and thrown away. Your original is not touched.\n"
                    "• Videos already finished stay finished.\n"
                    "• Nothing is deleted. Next time, choose the same folder and it carries on."),
}
PHASE_NAME = {"scan": "getting the photos ready", "naming": "waiting for names",
              "removing_faces": "removing face data",
              "waiting": "checking for copies", "saving": "saving names into the photos",
              "filing": "copying photos into a person's folder", "compressing": "compressing videos"}
SHUTDOWN_REASON = "No-Nonsense Photo Sorter is still working on your photos. Closing it now stops it part-way."


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setWindowIcon(app_icon())
        self.resize(820, 720)
        self.setMinimumSize(720, 620)
        apply_options(load_config().get("options"))
        self.exe = None
        self.root_folder = None
        self.worker = None
        self.deep = None
        self.deep_finished, self.deep_frac, self.deep_error, self.waiting_deep = True, 1.0, None, False
        self.deep_result = {"dups": 0, "near": 0, "late_photos": 0, "late_memes": 0, "late_faces": []}
        self.tasks = set(COMBINABLE)
        self._ticking = False
        self.phase = "welcome"
        self.status = None
        self.heartbeat = QTimer(self)
        self.heartbeat.timeout.connect(lambda: self.status and self.phase in BUSY and self.status.write())
        self.heartbeat.start(20000)
        self.stack = QStackedWidget()
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        left = QWidget()
        left.setObjectName("content")
        lay = QVBoxLayout(left)
        lay.setContentsMargins(32, 24, 32, 24)
        lay.addWidget(self.stack)
        self.build_nav()
        outer.addWidget(self.sidebar)
        outer.addWidget(left, 1)
        self.opt_open = False
        self.build_welcome()
        self.build_work()
        self.build_naming()
        self.build_done()
        self.build_people()
        self.build_results()
        self.build_options()
        self.stack.addWidget(self.opt_panel)
        for _b in self.opt_panel.findChildren(QPushButton):      # section tabs: never narrower than their text
            if _b.objectName() == "seg":
                _b.ensurePolished()
                _b.setMinimumWidth(_b.sizeHint().width())
        self.stack.currentChanged.connect(lambda *_: self.sync_options())
        self.thumb, self.thumb_token = None, 0
        self.show_last_run()

    def build_nav(self):
        n = QFrame()
        n.setObjectName("sidebar")
        n.setFixedWidth(204)
        nv = QVBoxLayout(n)
        nv.setContentsMargins(18, 30, 18, 22)
        nv.setSpacing(4)
        brand = QHBoxLayout()
        mark = QLabel()
        mark.setPixmap(QPixmap(str(RES_DIR / "assets" / "nnps_icon.png")).scaled(
            48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        brand.addWidget(mark)
        brand.addWidget(label(APP_SHORT_NAME, "brand", Qt.AlignLeft))
        brand.addStretch(1)
        nv.addLayout(brand)
        nv.addSpacing(24)
        self.nav_home = QPushButton("Sort photos")
        self.nav_opts = QPushButton("Options")
        for b, idx in ((self.nav_home, 0), (self.nav_opts, 6)):
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=idx: self.nav_to(i))
            nv.addWidget(b)
        nv.addStretch(1)
        nv.addWidget(label(f"Version {APP_VERSION}", "version", Qt.AlignLeft))
        self.sidebar = n

    # ---- pages
    def build_welcome(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addStretch(2)
        mark = QLabel()
        mark.setPixmap(QPixmap(str(RES_DIR / "assets" / "nnps_icon.png")).scaled(
            96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        v.addWidget(mark, alignment=Qt.AlignCenter)
        v.addWidget(label("No-Nonsense\nPhoto Sorter", "welcomeTitle"))
        rule = QFrame()
        rule.setObjectName("rule")
        rule.setFixedSize(64, 1)
        v.addSpacing(6)
        v.addWidget(rule, alignment=Qt.AlignCenter)
        v.addSpacing(8)
        v.addWidget(label("Tidy your photos and videos, set aside copies,\nand add the names of the people in them.", "subtitle"))
        v.addSpacing(14)
        v.addWidget(label("What should I do?", "caption"))
        v.addSpacing(2)
        saved = load_config().get("tasks")
        if not saved:
            saved = OLD_MODES.get(load_config().get("mode"), list(COMBINABLE))
        saved = [k for k in saved if k in TASK_LABEL] or list(COMBINABLE)
        box_holder = QFrame()
        box_holder.setObjectName("card")
        bv = QVBoxLayout(box_holder)
        bv.setContentsMargins(26, 10, 26, 10)
        bv.setSpacing(2)
        self.all_box = QCheckBox("Everything (copies, sort, names)")
        self.all_box.setObjectName("allbox")
        self.all_box.setToolTip("Ticks Find copies, Sort by date and Name faces together.")
        self.all_box.setFixedHeight(28)
        self.all_box.toggled.connect(lambda on: self.on_task_toggled("all", on))
        bv.addWidget(self.all_box)
        self.task_boxes = {}
        for key, text, hint in TASKS:
            cb = QCheckBox(text)
            cb.setToolTip(hint)
            cb.setFixedHeight(28)
            cb.toggled.connect(lambda on, k=key: self.on_task_toggled(k, on))
            self.task_boxes[key] = cb
            bv.addWidget(cb)
            if key == COMBINABLE[-1]:
                bv.addSpacing(6)             # small gap before the stand-alone jobs
        v.addWidget(box_holder, alignment=Qt.AlignCenter)
        self._ticking = True
        for k, cb in self.task_boxes.items():
            cb.setChecked(k in saved)
        self.all_box.setChecked(all(self.task_boxes[k].isChecked() for k in COMBINABLE))
        self._ticking = False
        v.addSpacing(4)
        self.mode_hint = label("", "hint")
        self.mode_hint.setMinimumHeight(34)            # room for two lines so the page never jumps
        self.mode_hint.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        v.addWidget(self.mode_hint)
        v.addSpacing(10)
        b = QPushButton("Choose Photo Folder")
        b.setObjectName("primary")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self.choose_folder)
        self.choose_btn = b
        v.addWidget(b, alignment=Qt.AlignCenter)
        self.undo_btn = QPushButton("Undo last recorded moves")
        self.undo_btn.setObjectName("link")
        self.undo_btn.clicked.connect(self.undo_moves)
        v.addWidget(self.undo_btn, alignment=Qt.AlignCenter)
        self.update_task_hint()
        v.addSpacing(6)
        self.last_note = label("", "hint")
        v.addWidget(self.last_note)
        v.addStretch(3)
        v.addWidget(label("Copies are moved to _Duplicates. Back up your photos before editing metadata.", "caption"))
        v.addSpacing(6)
        v.addWidget(label(f"Version {APP_VERSION}", "version"))
        sa = QScrollArea()                             # short window: scroll instead of squashing the rows
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        sa.viewport().setObjectName("welcomevp")
        sa.viewport().setStyleSheet("#welcomevp { background: #121212; }")
        sa.setWidget(pg)
        self.stack.addWidget(sa)

    def build_work(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addStretch(1)
        self.work_title = label("Working on it…", "title")
        v.addWidget(self.work_title)
        v.addSpacing(24)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        v.addWidget(self.bar)
        v.addSpacing(16)
        self.status_label = label("", "status")
        v.addWidget(self.status_label)
        v.addSpacing(8)
        v.addWidget(label("This can take a while. You can leave it running.\n"
                          "A note called Status.txt in your photo folder says whether it finished.", "caption"))
        v.addStretch(2)
        self.stack.addWidget(pg)

    def build_naming(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addWidget(label("Who is this?", "title"))
        v.addSpacing(12)
        self.face = QLabel()
        self.face.setAlignment(Qt.AlignCenter)
        v.addWidget(self.face, alignment=Qt.AlignCenter)
        v.addSpacing(16)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Type a name")
        self.name_edit.setAlignment(Qt.AlignCenter)
        self.completer_model = QStringListModel()
        comp = QCompleter(self.completer_model)
        comp.setCaseSensitivity(Qt.CaseInsensitive)
        self.name_edit.setCompleter(comp)
        self.name_edit.textChanged.connect(lambda t: self.next_btn.setEnabled(bool(t.strip())))
        self.name_edit.returnPressed.connect(self.on_next)
        v.addWidget(self.name_edit)
        v.addSpacing(6)
        self.name_hint = label("", "hint")
        v.addWidget(self.name_hint)
        v.addSpacing(12)
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("primary")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.on_next)
        v.addWidget(self.next_btn, alignment=Qt.AlignCenter)
        v.addStretch(1)
        row = QHBoxLayout()
        row.addStretch()
        for text, fn in (("Skip this picture", self.skip_picture), ("Skip this person", self.skip_person),
                         ("Skip all", self.skip_all)):
            b = QPushButton(text)
            b.setObjectName("link")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        self.counter = label("", "caption")
        v.addWidget(self.counter)
        self.bg = label("", "caption")
        v.addWidget(self.bg)
        self.stack.addWidget(pg)

    def build_done(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addStretch(1)
        mark = label("✓", "title")
        mark.setStyleSheet("font-size: 72px; color: #6F8A5E;")
        v.addWidget(mark)
        self.done_title = label("All done", "title")
        v.addWidget(self.done_title)
        v.addSpacing(8)
        self.done_text = label("", "subtitle")
        v.addWidget(self.done_text)
        v.addStretch(2)
        self.stack.addWidget(pg)

    def build_people(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        v.addWidget(label("Whose photos?", "title"))
        v.addSpacing(6)
        v.addWidget(label("Tick the people you want to see.", "subtitle"))
        v.addSpacing(12)
        self.people_list = QListWidget()
        v.addWidget(self.people_list, 1)
        v.addSpacing(10)
        self.match_box = QComboBox()
        self.match_box.addItem("Photos with any of them", "any")
        self.match_box.addItem("Photos with all of them together", "all")
        v.addWidget(self.match_box, alignment=Qt.AlignCenter)
        v.addSpacing(6)
        self.only_box = QCheckBox("Only photos where nobody else is in it")
        v.addWidget(self.only_box, alignment=Qt.AlignCenter)
        v.addSpacing(14)
        b = QPushButton("Show photos")
        b.setObjectName("primary")
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self.show_people_photos)
        v.addWidget(b, alignment=Qt.AlignCenter)
        back = QPushButton("Back")
        back.setObjectName("link")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(self.go_home)
        v.addWidget(back, alignment=Qt.AlignCenter)
        self.stack.addWidget(pg)

    def build_results(self):
        pg = QWidget()
        v = QVBoxLayout(pg)
        self.res_title = label("", "title")
        v.addWidget(self.res_title)
        self.res_note = label("", "hint")
        v.addWidget(self.res_note)
        v.addSpacing(6)
        self.grid = QListWidget()
        self.grid.setObjectName("grid")
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(150, 150))
        self.grid.setGridSize(QSize(170, 170))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setWordWrap(True)
        self.grid.itemDoubleClicked.connect(self.open_photo)
        card = QFrame()
        card.setObjectName("card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(12, 12, 12, 12)
        cv.addWidget(self.grid)
        v.addWidget(card, 1)
        self.folder_hint = label("", "caption")
        v.addWidget(self.folder_hint)
        v.addSpacing(6)
        row = QHBoxLayout()
        row.addStretch()
        cp = QPushButton("Copy to folder")
        cp.setObjectName("primary")
        cp.setCursor(Qt.PointingHandCursor)
        cp.clicked.connect(self.file_matches)
        row.addWidget(cp)
        row.addStretch()
        v.addLayout(row)
        back = QPushButton("Back to the list of people")
        back.setObjectName("link")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(lambda: self.stack.setCurrentIndex(4))
        v.addWidget(back, alignment=Qt.AlignCenter)
        self.stack.addWidget(pg)

    # ---- flow
    def go_home(self):
        self.stack.setCurrentIndex(0)
        self.show_last_run()

    def on_broken_done(self, res):
        self.set_phase("welcome")
        text = (f"Checked {res['checked']} files.\n"
                + (f"{res['found']} broken files moved into “{BROKEN_DIR}”.\nReport.txt in that folder says why."
                   if res["found"] else "No broken files were found."))
        if self.status:
            self.status.update("done", text.replace("\n", " "), force=True)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("No-Nonsense Photo Sorter")
        box.setText(text)
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole) if res["found"] else None
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()
        if open_btn is not None and (box.clickedButton() is open_btn or OPTS["open_folder"]):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(res["folder"])))
        self.go_home()

    def on_compress_done(self, res):
        self.set_phase("welcome")
        lines = []
        if res["done"]:
            lines.append(f"{res['done']} video{'s' if res['done'] != 1 else ''} compressed: "
                         f"{human_size(res['before'])} → {human_size(res['after'])} "
                         f"(saved {human_size(res['before'] - res['after'])}).")
            if res["replace"]:
                lines.append(f"The originals are in “{UNCOMPRESSED_DIR}”. Delete that folder when you are happy "
                             "with the result to free the space. No-Nonsense Photo Sorter never deletes anything itself.")
            else:
                lines.append(f"The compressed copies are in “{COMPRESSED_DIR}”. Your originals were not touched.")
        else:
            lines.append("No videos needed compressing.")
        if res["kept"]:
            lines.append(f"{res['kept']} left as they were (compressing did not make them smaller).")
        if res["skipped"]:
            lines.append(f"{res['skipped']} skipped (under the size limit, or done before).")
        if res["failed"]:
            lines.append(f"{res['failed']} could not be compressed and were left alone. "
                         f"“Compress Report.txt” in “{INFO_DIR}” says why.")
        text = "\n".join(lines)
        if self.status:
            self.status.update("done", " ".join(lines), force=True)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("No-Nonsense Photo Sorter")
        box.setText(text)
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole) if res["done"] else None
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()
        if open_btn is not None and (box.clickedButton() is open_btn or OPTS["open_folder"]):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(res["folder"] if res["folder"].exists() else self.root_folder)))
        self.go_home()

    def on_people_indexed(self, res):
        self.set_phase("welcome")
        if self.status:
            self.status.update("done", "Looked up the people named in the photos.", force=True)
        self.people_index = res["index"]
        if not res["counts"]:
            QMessageBox.information(self, "No names found",
                                    "No named people were found in these photos yet.\n\n"
                                    "Run No-Nonsense Photo Sorter with “Only name faces” (or “Everything”) first.")
            return self.go_home()
        self.people_list.clear()
        for name, n in sorted(res["counts"].items(), key=lambda kv: (-kv[1], kv[0].casefold())):
            it = QListWidgetItem(f"{name}   ({n})")
            it.setData(Qt.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.people_list.addItem(it)
        self.stack.setCurrentIndex(4)

    def show_people_photos(self):
        chosen = [self.people_list.item(i).data(Qt.UserRole) for i in range(self.people_list.count())
                  if self.people_list.item(i).checkState() == Qt.Checked]
        if not chosen:
            QMessageBox.information(self, "No-Nonsense Photo Sorter", "Tick at least one person first.")
            return
        self.matches = pick_photos(self.people_index, chosen, self.match_box.currentData() == "all",
                                   self.only_box.isChecked())
        if not self.matches:
            QMessageBox.information(self, "No-Nonsense Photo Sorter", "No photos match that choice.")
            return
        self.chosen = chosen
        n, shown = len(self.matches), self.matches[:SHOW_LIMIT]
        self.res_title.setText(f"{n} photo{'s' if n != 1 else ''}")
        self.res_note.setText("Making previews…" + (f"  (showing the first {SHOW_LIMIT}; copying uses all {n})"
                                                    if n > SHOW_LIMIT else "  Double-click a photo to open it."))
        self.folder_hint.setText(f"Folder: “{people_folder_name(chosen)}”, next to Sorted")
        self.grid.clear()
        self.stack.setCurrentIndex(5)
        if self.thumb is not None and self.thumb.isRunning():     # stop the previous preview maker first
            self.thumb.cancel = True
            self.thumb.wait(5000)
        self.thumb_token += 1
        token = self.thumb_token
        self.thumb = Worker(lambda prog: (token, make_grid_thumbs(shown, prog)))
        self.thumb.done.connect(self.fill_grid)
        self.thumb.progress.connect(lambda f, t: self.thumb_token == token and self.res_note.setText(t))
        self.thumb.start()

    def fill_grid(self, res):
        token, items = res
        if token != self.thumb_token:
            return
        self.res_note.setText(f"Double-click a photo to open it." + (
            f"  Showing the first {SHOW_LIMIT} of {len(self.matches)}." if len(self.matches) > SHOW_LIMIT else ""))
        for p, data in items:
            if data:
                pm = QPixmap.fromImage(QImage.fromData(data))
            else:
                pm = QPixmap(150, 150)
                pm.fill(Qt.lightGray)
                qp = QPainter(pm)
                qp.drawText(pm.rect(), Qt.AlignCenter, p.suffix.upper().lstrip(".") + "\n(no preview)")
                qp.end()
            it = QListWidgetItem(QIcon(pm), "")
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(str(p))
            self.grid.addItem(it)

    def open_photo(self, item):
        QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.UserRole)))

    def file_matches(self, *_):
        paths, root, chosen = list(self.matches), self.root_folder, list(self.chosen)
        self.work_title.setText("Copying photos…")
        self.run_worker(lambda prog: file_photos(paths, root, chosen, prog), self.on_filed, "filing")

    def on_filed(self, res):
        self.set_phase("welcome")
        if self.status:
            self.status.update("done", f"{res['done']} photos filed into “{res['folder'].name}”.", force=True)
        text = f"{res['done']} photos copied into “{res['folder'].name}”."
        if res["skipped"]:
            text += f"\n{res['skipped']} skipped (already there, or could not be read)."
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("No-Nonsense Photo Sorter")
        box.setText(text)
        open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is open_btn or OPTS["open_folder"]:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(res["folder"])))
        self.stack.setCurrentIndex(5)

    def build_options(self):
        p = QFrame()
        p.setObjectName("optpage")
        self.opt_panel = p
        pl = QVBoxLayout(p)
        pl.setContentsMargins(40, 30, 40, 24)
        pl.setSpacing(0)
        pl.addWidget(label("Options", "opttitle", Qt.AlignLeft))
        pl.addSpacing(16)
        seg = QHBoxLayout()
        seg.setSpacing(8)
        self.opt_tabs = []
        self.opt_stack = QStackedWidget()
        for i, title in enumerate(("Sorting Options", "Face/Name Options", "Video Options")):
            tb = QPushButton(title)
            tb.setObjectName("seg")
            tb.setCheckable(True)
            tb.setCursor(Qt.PointingHandCursor)
            tb.clicked.connect(lambda _=False, j=i: self.opt_stack.setCurrentIndex(j))
            seg.addWidget(tb)
            self.opt_tabs.append(tb)
        seg.addStretch(1)
        self.opt_stack.currentChanged.connect(
            lambda i: [tb.setChecked(j == i) for j, tb in enumerate(self.opt_tabs)])
        pl.addLayout(seg)
        pl.addSpacing(18)
        pl.addWidget(self.opt_stack, 1)
        pl.addSpacing(12)
        rb = QPushButton("Reset options")
        rb.setObjectName("link")
        rb.setCursor(Qt.PointingHandCursor)
        rb.clicked.connect(self.reset_options)
        pl.addWidget(rb, alignment=Qt.AlignLeft)
        self.opt_hints = {}

        def section():
            sc = QScrollArea()
            sc.setObjectName("optscroll")
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.viewport().setStyleSheet("background: #121212;")
            inner = QWidget()
            inner.setObjectName("optinner")
            sc.setWidget(inner)
            self.opt_stack.addWidget(sc)
            col = QVBoxLayout(inner)
            col.setContentsMargins(4, 4, 12, 20)
            col.setSpacing(3)
            return col

        def combo(v, key, title, table, gap=12):
            v.addSpacing(gap)
            v.addWidget(label(title, "optname", Qt.AlignLeft))
            cb = QComboBox()
            cb.setObjectName("optcombo")
            for k, text, _, _ in table:
                cb.addItem(text, k)
            cb.setCurrentIndex(max(0, cb.findData(OPTS[key])))
            cb.currentIndexChanged.connect(self.options_changed)
            v.addWidget(cb)
            hint = label("", "opthint", Qt.AlignLeft)
            v.addWidget(hint)
            self.opt_hints[key] = (hint, table)
            return cb

        def spin(v, title, lo, hi, value):
            v.addSpacing(12)
            row = QHBoxLayout()
            row.addWidget(label(title, "optname", Qt.AlignLeft), 1)
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(value)
            sb.valueChanged.connect(self.options_changed)
            row.addWidget(sb)
            v.addLayout(row)
            return sb

        # --- Sorting Options
        v = section()
        self.opt_speed = combo(v, "speed", "Working speed", SPEEDS, 0)
        self.opt_meme = combo(v, "meme_filter", "Meme and caption filter", MEME_FILTERS)
        v.addSpacing(14)
        self.opt_awake = QCheckBox("Keep PC awake while working")
        self.opt_awake.setChecked(OPTS["keep_awake"])
        self.opt_awake.toggled.connect(self.options_changed)
        v.addWidget(self.opt_awake)
        self.opt_open_cb = QCheckBox("Open folder when finished")
        self.opt_open_cb.setChecked(OPTS["open_folder"])
        self.opt_open_cb.toggled.connect(self.options_changed)
        v.addWidget(self.opt_open_cb)
        self.opt_sort_preview = QCheckBox("Preview sorting before moving")
        self.opt_sort_preview.setChecked(OPTS["sort_preview"])
        self.opt_sort_preview.toggled.connect(self.options_changed)
        v.addWidget(self.opt_sort_preview)
        v.addStretch(1)

        # --- Face/Name Options
        v = section()
        self.opt_accuracy = combo(v, "accuracy", "Face detection accuracy", ACCURACY, 0)
        self.opt_matching = combo(v, "matching", "Same-person matching", MATCHING)
        self.opt_min = spin(v, "Ask about people with at least this many photos", 2, 20, OPTS["min_photos"])
        self.opt_samples = spin(v, "Pictures shown per person", 3, 12, OPTS["samples"])
        v.addSpacing(14)
        self.face_option_boxes = {}
        for key, title in (("remember_people", "Remember people I confirm in this photo folder"),
                           ("review_faces", "Review uncertain faces before saving"),
                           ("use_calibration", "Use calibrated face matching when available")):
            cb = QCheckBox(title)
            cb.setChecked(OPTS[key])
            cb.toggled.connect(self.options_changed)
            self.face_option_boxes[key] = cb
            v.addWidget(cb)
        calibrate = QPushButton("Calibrate face matching…")
        calibrate.clicked.connect(self.calibrate_faces)
        v.addWidget(calibrate)
        wipe = QPushButton("Remove names from photos…")
        wipe.clicked.connect(self.choose_face_removal)
        v.addWidget(wipe)
        v.addSpacing(22)
        v.addWidget(label("Remembered faces", "opttitle2", Qt.AlignLeft))
        v.addWidget(label("Ran No-Nonsense Photo Sorter in several folders and then joined the folders? Join their remembered "
                          "faces too, so nothing is looked at twice.", "opthint", Qt.AlignLeft))
        v.addSpacing(6)
        mb = QPushButton("Merge face caches…")
        mb.setObjectName("secondary")
        mb.setCursor(Qt.PointingHandCursor)
        mb.clicked.connect(self.merge_caches)
        v.addWidget(mb, alignment=Qt.AlignLeft)
        v.addSpacing(16)
        v.addWidget(label("Changing face accuracy means photos are looked at again the first time. "
                          "Earlier results for each setting are kept.", "opthint", Qt.AlignLeft))
        v.addStretch(1)

        # --- Video Options
        v = section()
        self.opt_vquality = combo(v, "video_quality", "Quality", VIDEO_QUALITY, 0)
        self.opt_vcodec = combo(v, "video_codec", "Format", VIDEO_CODECS)
        self.opt_vmin = spin(v, "Skip videos smaller than (MB)", 0, 1000, OPTS["video_min_mb"])
        v.addSpacing(14)
        self.opt_vreplace = QCheckBox("Replace the original videos")
        self.opt_vreplace.setChecked(OPTS["video_replace"])
        self.opt_vreplace.toggled.connect(self.options_changed)
        v.addWidget(self.opt_vreplace)
        self.opt_vreplace_hint = label("", "opthint", Qt.AlignLeft)
        v.addWidget(self.opt_vreplace_hint)
        v.addStretch(1)

        self.opt_tabs[0].setChecked(True)
        self.opt_stack.setCurrentIndex(0)
        self.update_option_hints()

    def merge_caches(self):
        start = self.root_folder or load_config().get("last_folder") or "C:/"
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose the faces cache files to merge (hold Ctrl to pick several)", str(start),
            "Remembered faces (*.jsonl)")
        if len(files) < 1:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose the joined photo folder to save the result in",
                                                  str(self.root_folder or start))
        if not folder:
            return
        dlg = QProgressDialog("Merging…", None, 0, 100, self)
        dlg.setWindowTitle("No-Nonsense Photo Sorter")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        w = Worker(lambda prog: merge_face_caches(files, folder, prog))
        w.progress.connect(lambda f, t: (dlg.setLabelText(t), dlg.setValue(int(f * 100))))

        def finished(res):
            dlg.close()
            lines = [f"{name}: {total} photos remembered ({new} new)" for name, total, new in res]
            QMessageBox.information(self, "Merged",
                                    "Done. The faces from all those files are now remembered together.\n\n"
                                    + "\n".join(lines) + "\n\nPhotos in the folder that are not in the cache "
                                    "are simply looked at the next time you run No-Nonsense Photo Sorter there.")

        def failed(msg):
            dlg.close()
            QMessageBox.warning(self, "Could not merge", msg)

        w.done.connect(finished)
        w.failed.connect(failed)
        self._merge_worker = w
        w.start()

    def update_option_hints(self):
        for key, (hint, table) in self.opt_hints.items():
            hint.setText(next(h for k, _, h, _ in table if k == OPTS[key]))
        self.opt_vreplace_hint.setText(
            f"On: the compressed video takes the original's place and the original is moved to “{UNCOMPRESSED_DIR}”. "
            "Delete that folder yourself to free the space." if OPTS["video_replace"] else
            f"Off: compressed copies go to “{COMPRESSED_DIR}” and your originals are not touched.")

    def options_changed(self, *_):
        apply_options({"accuracy": self.opt_accuracy.currentData(), "matching": self.opt_matching.currentData(),
                       "speed": self.opt_speed.currentData(), "min_photos": self.opt_min.value(),
                       "samples": self.opt_samples.value(), "keep_awake": self.opt_awake.isChecked(),
                       "sort_preview": self.opt_sort_preview.isChecked(),
                       **{k: cb.isChecked() for k, cb in self.face_option_boxes.items()},
                       "open_folder": self.opt_open_cb.isChecked(), "video_quality": self.opt_vquality.currentData(),
                       "video_codec": self.opt_vcodec.currentData(), "video_min_mb": self.opt_vmin.value(),
                       "video_replace": self.opt_vreplace.isChecked(),
                       "meme_filter": self.opt_meme.currentData()})
        cfg = load_config()
        cfg["options"] = dict(OPTS)
        save_config(cfg)
        self.update_option_hints()

    def reset_options(self):
        for key, cb in self.face_option_boxes.items():
            cb.blockSignals(True)
            cb.setChecked(DEFAULT_OPTS[key])
            cb.blockSignals(False)
        self.opt_sort_preview.blockSignals(True)
        self.opt_sort_preview.setChecked(DEFAULT_OPTS["sort_preview"])
        self.opt_sort_preview.blockSignals(False)
        widgets = (self.opt_accuracy, self.opt_matching, self.opt_speed, self.opt_min, self.opt_samples,
                   self.opt_awake, self.opt_open_cb, self.opt_vquality, self.opt_vcodec, self.opt_vmin,
                   self.opt_vreplace, self.opt_meme)
        for w in widgets:
            w.blockSignals(True)
        for cb, key in ((self.opt_accuracy, "accuracy"), (self.opt_matching, "matching"), (self.opt_speed, "speed"),
                        (self.opt_vquality, "video_quality"), (self.opt_vcodec, "video_codec"),
                        (self.opt_meme, "meme_filter")):
            cb.setCurrentIndex(max(0, cb.findData(DEFAULT_OPTS[key])))
        self.opt_min.setValue(DEFAULT_OPTS["min_photos"])
        self.opt_samples.setValue(DEFAULT_OPTS["samples"])
        self.opt_awake.setChecked(DEFAULT_OPTS["keep_awake"])
        self.opt_open_cb.setChecked(DEFAULT_OPTS["open_folder"])
        self.opt_vmin.setValue(DEFAULT_OPTS["video_min_mb"])
        self.opt_vreplace.setChecked(DEFAULT_OPTS["video_replace"])
        for w in widgets:
            w.blockSignals(False)
        self.options_changed()

    def toggle_options(self):
        self.opt_open = not self.opt_open
        self.sync_options()

    def sync_options(self):
        """Keeps the sidebar's highlighted button in step with the page on screen."""
        idx = self.stack.currentIndex()
        self.nav_home.setChecked(idx == 0)
        self.nav_opts.setChecked(idx == 6)

    def nav_to(self, index):
        if self.phase in BUSY:
            return
        if index == 0:
            self.go_home()
        else:
            self.stack.setCurrentIndex(index)

    def selected_tasks(self):
        return [k for k, _, _ in TASKS if self.task_boxes[k].isChecked()]

    def on_task_toggled(self, key, on):
        if self._ticking:
            return
        self._ticking = True
        try:
            if key == "all":
                for k in COMBINABLE:
                    self.task_boxes[k].setChecked(on)
                if on:
                    for k in STANDALONE:
                        self.task_boxes[k].setChecked(False)
            elif on:
                others = STANDALONE if key in COMBINABLE else [k for k in self.task_boxes if k != key]
                for k in others:                 # stand-alone jobs and the copies/sort/faces mix do not run together
                    self.task_boxes[k].setChecked(False)
            self.all_box.setChecked(all(self.task_boxes[k].isChecked() for k in COMBINABLE))
        finally:
            self._ticking = False
        self.update_task_hint()

    def update_task_hint(self):
        picked = self.selected_tasks()
        self.choose_btn.setEnabled(bool(picked))
        if not picked:
            self.mode_hint.setText("Tick at least one job to carry on.")
        elif len(picked) == 1:
            self.mode_hint.setText(next(h for k, _, h in TASKS if k == picked[0]))
        else:
            self.mode_hint.setText("I will: " + ", then ".join(TASK_LABEL[k].lower() for k in picked) + ".")

    def choose_folder(self):
        picked = self.selected_tasks()
        if picked == ["names_off"]:
            return self.choose_face_removal()
        self.exe = None if picked == ["copies"] else find_exiftool()
        if picked != ["copies"] and not self.exe:
            QMessageBox.information(self, "One quick thing",
                                    "Please show me where exiftool.exe is (it is inside the digiKam folder).")
            path, _ = QFileDialog.getOpenFileName(self, "Find exiftool.exe", "C:/Program Files", "exiftool.exe")
            if not path:
                return
            self.exe = path
            cfg = load_config()
            cfg["exiftool"] = path
            save_config(cfg)
        self.ffmpeg = None
        if "compress" in self.selected_tasks():
            self.ffmpeg = find_ffmpeg()
            if not self.ffmpeg:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Information)
                box.setWindowTitle("One quick thing")
                box.setText("Compressing videos needs a free tool called ffmpeg, and I could not find it.")
                box.setInformativeText("Download the “release essentials” zip, unzip it, and either show me ffmpeg.exe "
                                       "(it is in the bin folder) or copy ffmpeg.exe next to No-Nonsense Photo Sorter.")
                show = box.addButton("Show me ffmpeg.exe", QMessageBox.AcceptRole)
                get = box.addButton("Get ffmpeg", QMessageBox.ActionRole)
                box.addButton("Cancel", QMessageBox.RejectRole)
                box.exec()
                if box.clickedButton() is get:
                    QDesktopServices.openUrl(QUrl("https://www.gyan.dev/ffmpeg/builds/"))
                    return
                if box.clickedButton() is not show:
                    return
                path, _ = QFileDialog.getOpenFileName(self, "Find ffmpeg.exe", "C:/", "ffmpeg.exe")
                if not path:
                    return
                self.ffmpeg = path
                cfg = load_config()
                cfg["ffmpeg"] = path
                save_config(cfg)
        folder = QFileDialog.getExistingDirectory(self, "Choose your photo folder")
        if not folder:
            return
        self.root_folder = os.path.abspath(folder)
        folder = self.root_folder
        cfg = load_config()
        cfg["last_folder"] = folder
        if not picked:
            return
        self.tasks = set(picked)
        start_move_run(folder)
        cfg["tasks"] = picked
        cfg.pop("mode", None)
        save_config(cfg)
        self.status = Status(folder)
        self.status.update("working", "starting", force=True)
        self.deep, self.deep_finished, self.deep_error, self.waiting_deep = None, True, None, False
        self.deep_result = {"dups": 0, "near": 0, "late_photos": 0, "late_memes": 0, "late_faces": []}
        self.work_title.setText("Checking for copies…" if self.tasks == {"copies"} else "Working on it…")
        tasks = set(self.tasks)
        if "people" in tasks:
            self.work_title.setText("Reading the names in your photos…")
            self.run_worker(lambda prog: stage_people(folder, self.exe, prog), self.on_people_indexed, "scan")
            return
        if "compress" in tasks:
            self.work_title.setText("Compressing videos…")
            ff, o = self.ffmpeg, dict(OPTS)
            self.run_worker(lambda prog: stage_compress(folder, ff, prog, o["video_quality"], o["video_codec"],
                                                        o["video_min_mb"], o["video_replace"]),
                            self.on_compress_done, "compressing")
            return
        if "broken" in tasks:
            self.work_title.setText("Checking for broken files…")
            self.run_worker(lambda prog: stage_broken(folder, self.exe, prog), self.on_broken_done, "scan")
            return
        if "sort" in tasks and OPTS["sort_preview"]:
            self.run_worker(lambda prog: sort_files(walk_media(Path(folder)), self.exe, Path(folder), prog, 0, 1, preview=True),
                            self.on_sort_preview, "scan")
        else:
            self.run_worker(lambda prog: stage_prepare(folder, self.exe, prog, tasks), self.on_prepared, "scan")

    def on_sort_preview(self, result):
        self.set_phase("welcome")
        box = QMessageBox(self)
        box.setWindowTitle("Preview sorting")
        box.setText(f"Preview for {len(result)} media files. {result.failures} could not be planned.")
        box.setInformativeText("Expand Details to see destinations, related files and date fallbacks. "
                               "Copy checks may set duplicate files aside instead. Start sorting when ready.")
        box.setDetailedText("\n".join(result.preview_lines) or "No files found.")
        start = box.addButton("Start sorting", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is start:
            folder, tasks = self.root_folder, set(self.tasks)
            self.run_worker(lambda prog: stage_prepare(folder, self.exe, prog, tasks), self.on_prepared, "scan")
        else:
            self.stack.setCurrentIndex(0)
            self.show_last_run()

    def hwnd(self):
        try:
            return int(self.winId())
        except Exception:
            return 0

    def set_phase(self, phase):
        self.phase = phase
        busy = phase in BUSY
        self.nav_home.setEnabled(not busy)
        self.nav_opts.setEnabled(not busy)
        keep_awake(busy and OPTS["keep_awake"])
        shutdown_note(self.hwnd(), SHUTDOWN_REASON if busy else None)
        if self.status and busy:
            state = {"naming": "naming", "saving": "saving"}.get(phase, "working")
            self.status.update(state, "" if state != "working" else PHASE_NAME.get(phase, ""), force=True)

    def run_worker(self, fn, on_done, phase="scan"):
        CANCEL_EVENT.clear()
        self.set_phase(phase)
        self.set_progress(0, "Starting…")
        self.stack.setCurrentIndex(1)
        self.worker = Worker(fn)
        self.worker.progress.connect(self.set_progress)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def set_progress(self, frac, text):
        self.bar.setValue(int(max(0.0, min(1.0, frac)) * 1000))
        self.status_label.setText(text)
        if self.status and self.phase in ("scan", "waiting", "saving", "filing", "compressing", "removing_faces") and text:
            self.status.update(detail=text)

    def on_failed(self, msg):
        if self.status:
            self.status.update("error", "See the message on screen.", force=True)
        self.set_phase("welcome")
        keep_awake(False)
        QMessageBox.warning(self, "Something went wrong", msg)
        self.stack.setCurrentIndex(0)
        self.show_last_run()

    def show_last_run(self):
        folder = load_config().get("last_folder")
        self.undo_btn.setEnabled(bool(folder and pending_moves(folder)))
        st = read_status(folder) if folder else None
        if not st:
            self.last_note.setText("")
            return
        if st.get("state") == "done":
            self.last_note.setText(f"Last time ({st.get('updated', '')}) this finished.")
        else:
            self.last_note.setText(f"Last time did not finish ({st.get('updated', '')}). "
                                   "Choose the same folder to carry on.")

    def undo_moves(self):
        folder = load_config().get("last_folder")
        if not folder:
            return
        res = undo_last_run(folder)
        self.show_last_run()
        QMessageBox.information(self, "Undo last recorded moves",
                                f"Restored {res['restored']} files. Skipped {res['skipped']} files "
                                "that were missing, already restored, or blocked by another file. "
                                "Names saved inside photos are not changed.")

    def on_prepared(self, data):
        self.data = data
        self.stats = (data["photos"], data["dups"])
        self.refs = load_people(self.root_folder) if OPTS["remember_people"] else {}
        self.cluster_names, self.spelling = {}, {name.casefold(): name for name in self.refs}
        self.data["confirmed"] = {}
        self.last_name = ""
        threshold = matching_threshold(self.root_folder, self.refs)
        self.ask = [ci for ci, m in enumerate(data["members"]) if data.get("rename_all") or
                    (len(m) >= MIN_CLUSTER and not all(predicted_name(data["faces"][i].emb, self.refs, threshold)
                            for i in pick_samples(data["faces"], m)))]
        self.completer_model.setStringList(sorted(self.refs))
        self.pos = -1
        if data["groups"] or "copies" in self.tasks:
            self.deep_finished, self.deep_frac = False, 0.0
            root, exe, tasks = self.root_folder, self.exe, set(self.tasks)
            self.deep = Worker(lambda prog: stage_deep(root, exe, data["groups"], data["current"],
                                                       data["cache"], prog, tasks))
            self.deep.progress.connect(self.on_deep_progress)
            self.deep.done.connect(self.on_deep_done)
            self.deep.failed.connect(self.on_deep_failed)
            self.deep.start()
        self.next_person()

    def on_deep_progress(self, frac, text):
        self.deep_frac = frac
        if self.phase == "naming":
            self.bg.setText(f"Checking for copies in the background… {int(frac * 100)}%")
        elif self.phase == "waiting":
            self.set_progress(frac, "Finishing the check for copies…")

    def on_deep_done(self, res):
        self.deep_result, self.deep_finished = res, True
        self.bg.setText("")
        if self.waiting_deep:
            self.waiting_deep = False
            self.begin_save()

    def on_deep_failed(self, msg):
        self.deep_error, self.deep_finished = msg, True      # copies simply stay where they are
        self.bg.setText("")
        if self.waiting_deep:
            self.waiting_deep = False
            self.begin_save()

    def next_person(self):
        self.pos += 1
        if self.pos >= len(self.ask):
            return self.start_match()
        self.cur = self.ask[self.pos]
        self.samples = pick_samples(self.data["faces"], self.data["members"][self.cur])
        self.votes = []
        self.si = 0
        self.last_name = ""
        self.show_sample()
        self.set_phase("naming")
        self.stack.setCurrentIndex(2)
        QApplication.alert(self, 0)

    def show_sample(self):
        f = self.data["faces"][self.samples[self.si]]
        self.face.setPixmap(rounded_pixmap(f.thumb, 280, 56))
        self.name_edit.setText(self.last_name)
        self.name_edit.selectAll()
        self.name_hint.setText("Same person? Just press Next." if self.last_name else " ")
        self.counter.setText(f"Person {self.pos + 1} of {len(self.ask)}  ·  Photo {self.si + 1} of {len(self.samples)}")
        self.name_edit.setFocus()

    def on_next(self):
        name = self.name_edit.text().strip()
        if not name:
            return
        name = self.spelling.setdefault(name.casefold(), name)
        if not self.confirm_face(self.samples[self.si], name):
            return
        self.votes.append(name)
        self.last_name = name
        self.completer_model.setStringList(sorted(self.refs.keys()))
        self.si += 1
        if self.si >= len(self.samples):
            self.finish_person()
        else:
            self.show_sample()

    def finish_person(self):
        if self.votes:
            self.cluster_names[self.cur] = Counter(self.votes).most_common(1)[0][0]
        self.next_person()

    def skip_picture(self):
        """This picture is not the person being named: leave it unnamed and move on to the next picture."""
        self.data["confirmed"][self.samples[self.si]] = None      # None = unnamed, even if a suggestion exists
        self.si += 1
        if self.si >= len(self.samples):
            self.finish_person()
        else:
            self.show_sample()

    def skip_person(self):
        self.finish_person()

    def skip_all(self):
        if self.votes:
            self.cluster_names[self.cur] = Counter(self.votes).most_common(1)[0][0]
        self.start_match()

    def start_match(self):
        if self.waiting_deep:
            return
        if not self.deep_finished:
            self.waiting_deep = True
            self.work_title.setText("Checking for copies…" if self.tasks == {"copies"} else "Almost done…")
            self.set_phase("waiting")
            self.stack.setCurrentIndex(1)
            self.set_progress(self.deep_frac, "Finishing the check for copies…")
            return
        self.begin_save()

    def begin_save(self):
        self.data["faces"] = self.data["faces"] + self.deep_result["late_faces"]
        if OPTS["review_faces"]:
            self.review_uncertain_faces()
        self.data["threshold"] = matching_threshold(self.root_folder, self.refs)
        if not self.refs:                                    # nobody was named: nothing to save into the photos
            return self.on_matched({"tagged": 0, "people": []})
        self.work_title.setText("Saving names…")
        self.run_worker(lambda prog: stage_match(self.data, self.refs, self.cluster_names, self.exe, prog),
                        self.on_matched, "saving")

    def confirm_face(self, index, name):
        name = name.strip()
        if not name or len(name) > 100 or any(ord(c) < 32 for c in name):
            QMessageBox.warning(self, "Person name", "Use a name of 1–100 characters with no control characters.")
            return False
        name = self.spelling.setdefault(name.casefold(), name)
        embedding = self.data["faces"][index].emb
        if OPTS["remember_people"]:
            try:
                self.refs = remember_person(self.root_folder, name, embedding)
            except (OSError, ValueError) as exc:
                face_report(self.root_folder, f"Could not remember approved person: {exc}")
                self.refs.setdefault(name, []).append(embedding)
        else:
            self.refs.setdefault(name, []).append(embedding)
        self.data["confirmed"][index] = name
        return True

    def review_uncertain_faces(self):
        """Uncertain faces are reviewed one photo at a time, so the whole group is on screen together."""
        self.set_phase("naming")
        photos = {}                                          # photo path -> its unnamed faces, in order
        for index, face in enumerate(self.data["faces"]):
            if index in self.data["confirmed"] or not Path(face.path).exists():
                continue
            photos.setdefault(face.path, []).append(index)
        threshold = matching_threshold(self.root_folder, self.refs)
        self.naming_left = sum(1 for indices in photos.values() for i in indices
                               if not predicted_name(self.data["faces"][i].emb, self.refs, threshold))
        for number, (path, indices) in enumerate(photos.items(), 1):
            if self.review_photo(path, indices, number, len(photos)) is False:    # "Skip remaining"
                break

    def review_photo(self, path, indices, number=1, of=1):
        """One dialog per photo: the whole picture with every unnamed face numbered, and a name box per face.
        Returns False if the user chose to skip the rest."""
        faces = self.data["faces"]
        threshold = matching_threshold(self.root_folder, self.refs)
        todo = [i for i in indices if not predicted_name(faces[i].emb, self.refs, threshold)]
        if not todo:
            return True
        dlg = QDialog(self)
        dlg.setWindowTitle("Who is who?")
        # Keep the whole dialog on screen: the picture gets what is left after the name rows and buttons.
        screen = QApplication.primaryScreen()
        avail_h = screen.availableGeometry().height() if screen else 900
        dlg.setMaximumHeight(int(avail_h * 0.96))
        picture_h = max(240, int(avail_h * 0.96) - 190 - 64 * len(todo))
        layout = QVBoxLayout(dlg)
        left = max(0, self.naming_left)
        layout.addWidget(label(f"{Path(path).name}  ·  photo {number} of {of}  ·  {len(todo)} unnamed "
                               f"{'face' if len(todo) == 1 else 'faces'}  ·  {left} faces left to name in total",
                               "caption"))
        boxes = [(n, faces[i].box) for n, i in enumerate(todo, 1)]
        picture = numbered_photo(path, boxes, 720, picture_h)
        shown = QLabel()
        shown.setAlignment(Qt.AlignCenter)
        if picture is not None:
            shown.setPixmap(picture)
        else:
            shown.setText("This photo can not be shown here. The faces are listed below.")

        def show_face(selected=None):
            pic = numbered_photo(path, boxes, 720, picture_h, selected)
            if pic is not None:
                shown.setPixmap(pic)
        layout.addWidget(shown, alignment=Qt.AlignCenter)
        combos = {}
        for n, index in enumerate(todo, 1):
            face = faces[index]
            row = QHBoxLayout()
            row.addWidget(label(str(n), "title"))
            thumb = QLabel()
            thumb.setPixmap(rounded_pixmap(face.thumb, 56, 10))
            row.addWidget(thumb)
            scores = score_face(face.emb, self.refs)
            hint = ", ".join(f"{name} ({s:.2f})" for s, name in scores[:2]) or "No remembered people yet"
            row.addWidget(label("Suggestions: " + hint, "hint", Qt.AlignLeft))
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems([name for _, name in scores])
            combo.setCurrentText("")                         # blank: a wrong guess is worse than no name
            combos[index] = combo
            watch = FocusWatcher(lambda n=n: show_face(n), dlg)     # typing in this name box draws its face
            combo.installEventFilter(watch)
            combo.lineEdit().installEventFilter(watch)
            row.addWidget(combo, 1)
            layout.addLayout(row)
        buttons = QHBoxLayout()
        save = QPushButton("Save names")
        leave = QPushButton("Leave this photo unnamed")
        stop = QPushButton("Skip remaining")
        buttons.addWidget(save); buttons.addWidget(leave); buttons.addWidget(stop)
        layout.addLayout(buttons)
        chosen = {}

        def accept():
            for index in todo:
                name = combos[index].currentText().strip()
                if name and (len(name) > 100 or any(ord(c) < 32 for c in name)):
                    QMessageBox.warning(dlg, "Person name", "Use a name of 1–100 characters with no control characters.")
                    return
                chosen[index] = name
            dlg.done(1)

        save.clicked.connect(accept)
        leave.clicked.connect(lambda: dlg.done(2))
        stop.clicked.connect(dlg.reject)
        result = dlg.exec()
        dlg.deleteLater()
        if result in (1, 2):
            self.naming_left -= len(todo)
        if result == 1:
            for index in todo:
                if chosen[index]:
                    self.confirm_face(index, chosen[index])
                else:
                    self.data["confirmed"][index] = None
        elif result == 2:
            for index in todo:
                self.data["confirmed"][index] = None
        else:
            return False
        return True

    def choose_face_removal(self):
        exe = find_exiftool()
        if not exe:
            exe, _ = QFileDialog.getOpenFileName(self, "Find exiftool.exe", "C:/Program Files", "exiftool.exe")
        if not exe:
            return
        root = QFileDialog.getExistingDirectory(self, "Choose the folder to remove face data from",
                                                str(self.root_folder or load_config().get("last_folder") or "C:/"))
        if not root:
            return
        self.face_removal_exe = exe
        self.root_folder = root
        self.status = Status(root)
        self.work_title.setText("Checking images for face data…")
        self.run_worker(lambda prog: prepare_face_removal(root, exe, prog), self.on_face_removal_preview, "scan")

    def on_face_removal_preview(self, plan):
        self.set_phase("welcome")
        box = QMessageBox(self)
        box.setWindowTitle("Remove names from photos")
        box.setText(f"Remove face names, region annotations and matching people tags from {len(plan['entries'])} files?")
        box.setInformativeText(f"Folder: {plan['root']}\nIncludes images and XMP sidecars in subfolders, including duplicates. "
                               f"{len(plan['unreadable'])} files could not be checked.\n\n"
                               "Original files are copied to _Photo Helper/face-removal-backups before editing. "
                               "Other applications may keep their own face databases.")
        forget = QCheckBox("Reset face memory so I can name everyone again")
        forget.setToolTip("Clears active remembered names, calibration and detection caches. Images can be scanned and named again.")
        forget.setChecked(True)
        box.setCheckBox(forget)
        box.setDetailedText("\n".join(entry["path"] for entry in plan["entries"]) or "No matching face metadata found.")
        remove = box.addButton("Remove names", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() is not remove:
            if self.status:
                self.status.update("closed", "Face data removal cancelled; no images changed.", force=True)
            self.stack.setCurrentIndex(0)
            return
        self.work_title.setText("Removing face data…")
        clear = forget.isChecked()
        self.run_worker(lambda prog: remove_face_data(plan, self.face_removal_exe, prog, clear),
                        self.on_face_removal_done, "removing_faces")

    def on_face_removal_done(self, result):
        message = (f"Cleaned {result['removed']} files. {result['failed']} failures. "
                   f"Forgot {result['forgotten']} remembered-data files.\n\n"
                   "These images can be scanned and named again.\n\n"
                   + (f"Old face memory files were moved to: {result['backups']}\n" if result["forgotten"] else "")
                   + "Report: _Photo Helper/Face Removal Report.txt")
        self.set_phase("welcome")
        if self.status:
            self.status.update("done" if not result["failed"] else "error", message, force=True)
        self.stack.setCurrentIndex(0)
        box = QMessageBox(self)
        box.setWindowTitle("Face data reset finished")
        box.setText(message)
        scan = box.addButton("Scan faces again", QMessageBox.AcceptRole)
        box.addButton("Done", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is scan:
            self.start_face_rescan()

    def start_face_rescan(self):
        root = self.root_folder
        self.exe = self.face_removal_exe
        self.tasks = {"faces"}
        self.deep, self.deep_finished, self.deep_error, self.waiting_deep = None, True, None, False
        self.deep_result = {"dups": 0, "near": 0, "late_photos": 0, "late_memes": 0, "late_faces": []}
        self.refs, self.cluster_names, self.spelling = {}, {}, {}
        self.status = Status(root)
        cfg = load_config()
        cfg.update(last_folder=root, tasks=["faces"])
        save_config(cfg)
        self.work_title.setText("Scanning faces again…")
        self.run_worker(lambda prog: prepare_face_rescan(root, self.exe, prog), self.on_prepared, "scan")

    def calibrate_faces(self):
        root = self.root_folder or load_config().get("last_folder")
        if not root:
            root = QFileDialog.getExistingDirectory(self, "Choose the folder with your approved people")
        if not root:
            return
        try:
            result = calibrate_people(load_people(root))
        except ValueError as exc:
            QMessageBox.information(self, "Face calibration", str(exc))
            return
        box = QMessageBox(self)
        box.setWindowTitle("Face calibration")
        box.setText(f"Suggested threshold: {result['threshold']:.2f} from {result['samples']} approved samples of {result['people']} people.")
        box.setInformativeText(f"{result['sample_accept_rate']:.0%} of these samples match another sample of the same person. "
                               "This is a check of your approved examples, not an independent accuracy test. "
                               "Calibration never lowers your selected matching threshold and expires when approved samples change.")
        apply = box.addButton("Apply calibration", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is apply:
            try:
                atomic_json(Path(root) / INFO_DIR / "face-calibration.json", result)
            except OSError as exc:
                QMessageBox.warning(self, "Could not save calibration", str(exc))

    def on_matched(self, res):
        dr = self.deep_result
        lines = []
        if "sort" in self.tasks:
            memes = self.data.get("memes", 0) + dr["late_memes"]
            photos = self.stats[0] + dr["late_photos"] - memes
            lines.append(f"{photos} photos and videos sorted by date")
            failed = self.data.get("sort_failures", 0) + dr.get("sort_failures", 0)
            if failed:
                lines.append(f"{failed} files could not be sorted; see _Photo Helper/Sort Report.txt")
            if memes:
                lines.append(f"{memes} memes and GIFs sorted separately")
        if "copies" in self.tasks:
            lines.append(f"{dr['dups']} exact copies set aside")
            if dr.get("copy_failures"):
                lines.append(f"{dr['copy_failures']} exact copies could not be moved; they remain in place")
            if dr.get("near"):
                lines.append(f"{dr['near']} look-alike copies set aside (lower quality ones are in “_Verify”)")
            if self.deep_error:
                lines.append("The check for copies could not finish, so some copies may still be mixed in.")
        if "faces" in self.tasks:
            failures = self.data.get("face_failures", 0) + dr.get("face_failures", 0) + res.get("failed", 0)
            if failures:
                lines.append(f"{failures} face scan or save failures; see _Photo Helper/Face Report.txt")
            if res.get("uncertain"):
                lines.append(f"{res['uncertain']} uncertain faces left unnamed")
            if res["tagged"]:
                lines.append(f"{len(res['people'])} people named in {res['tagged']} photos")
            else:
                lines.append("No names were saved into the photos")
        text = "\n".join(lines)
        self.set_phase("done")
        if self.status:
            self.status.update("done", "; ".join(lines) + ".", force=True)
        self.done_text.setText(text)
        self.stack.setCurrentIndex(3)
        self.set_progress(1, "")
        QApplication.alert(self, 0)
        dlg = QDialog(self)
        dlg.setWindowTitle("No-Nonsense Photo Sorter")
        dlg.setMinimumWidth(380)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(36, 32, 36, 28)
        v.addWidget(label("All done ✓", "title"))
        v.addSpacing(8)
        v.addWidget(label(text, "subtitle"))
        v.addSpacing(20)
        b = QPushButton("Close")
        b.setObjectName("primary")
        b.clicked.connect(dlg.accept)
        v.addWidget(b, alignment=Qt.AlignCenter)
        dlg.exec()
        if OPTS["open_folder"] and self.root_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root_folder)))
        QApplication.quit()

    # ---- closing
    def closeEvent(self, ev):
        if self.phase in BUSY:
            head, body = CLOSE_WARNING[self.phase]
            copies = ("The check for copies is not finished, so some copies will stay where they are."
                      if not self.deep_finished else "The check for copies is already finished.")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Close No-Nonsense Photo Sorter?")
            box.setText(head)
            box.setInformativeText("If you close it now:\n\n" + body.format(copies=copies))
            keep = box.addButton("Keep working", QMessageBox.AcceptRole)
            box.addButton("Close anyway", QMessageBox.DestructiveRole)
            box.setDefaultButton(keep)
            box.exec()
            if box.clickedButton() is keep:
                ev.ignore()
                return
            phase = self.phase
            self.phase = "closing"
            CANCEL_EVENT.set()
            for w in (self.worker, self.deep):
                if w is not None and w.isRunning():
                    w.cancel = True
            for w in (self.worker, self.deep):
                if w is not None:
                    w.wait(15000)
            if self.status:
                self.status.update("closed", PHASE_NAME.get(phase, ""), force=True)
        if self.thumb is not None and self.thumb.isRunning():     # preview maker: stop it before leaving
            self.thumb.cancel = True
            self.thumb.wait(5000)
        keep_awake(False)
        shutdown_note(self.hwnd(), None)
        ev.accept()


def load_fonts():
    """Load the bundled Inter / Cinzel / Cormorant Garamond faces (SIL Open Font License) so the app looks the
    same on every PC, offline. Falls back to Segoe UI if a file is missing."""
    try:
        d = RES_DIR / "fonts"
        for f in sorted(d.glob("*.ttf")):
            QFontDatabase.addApplicationFont(str(f))
    except Exception:
        pass


def tick_css():
    """Dark tick for the gold checked boxes. Qt style sheets can only show a tick from a picture file,
    so draw one now and point the style sheet at it."""
    try:
        d = Path(tempfile.gettempdir()) / "PhotoHelperUI"
        d.mkdir(exist_ok=True)
        f = d / "tick.png"
        big = QPixmap(72, 72)
        big.fill(Qt.transparent)
        qp = QPainter(big)
        qp.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#0A0A0A"), 9)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        qp.setPen(pen)
        qp.drawPolyline([QPoint(18, 38), QPoint(31, 51), QPoint(54, 21)])
        qp.end()
        pm = big.scaled(18, 18, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)   # exactly the indicator size
        pm.save(str(f), "PNG")
        return ("QCheckBox::indicator:checked { image: url(%s); width: 18px; height: 18px; }\n"
                "QCheckBox::indicator:checked:disabled { image: url(%s); width: 18px; height: 18px; }\n" % (f.as_posix(), f.as_posix()))
    except Exception:
        return ""


def main():
    if sys.platform == "win32":                 # lets Windows show our camera in the taskbar, not Python's icon
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PhotoHelper.App")
        except Exception:
            pass
    app = QApplication(sys.argv)
    load_fonts()
    app.setStyleSheet(STYLE + tick_css())
    app.setWindowIcon(app_icon())

    def hook(et, ev, tb):
        msg = "".join(traceback.format_exception(et, ev, tb))
        try:
            (APP_DIR / "error.log").write_text(msg, encoding="utf-8")
        except Exception:
            pass
        QMessageBox.critical(None, "No-Nonsense Photo Sorter", "Something went wrong.\n\n" + msg[-600:])

    sys.excepthook = hook
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        import release_checks
        sys.exit(release_checks.run(sys.modules[__name__], sys.argv[2]))
    main()
