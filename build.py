"""Build a versioned portable release and verify the packaged executable."""
import json
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
import photo_helper as ph

HERE = Path(__file__).resolve().parent
TOOL_SHA256 = {
    "ffmpeg.exe": "557e97af30a1c9b6dfbd08a1d2a04894279dd15a77d23b1723e560cc7f4fc90e",
    "ffprobe.exe": "11cb2352bac147bb8906f567af9af1da3779eca73e7a71a49a81c7d277e289cd",
    "exiftool.exe": "44b336d7596c3ec86577a5ff6d46e8388a56eb69a15f09987d4ac6cc3abbf75e",
}


def make_version_file(build):
    """Write the Windows version resource from APP_VERSION so Properties > Details shows it."""
    v = ph.APP_VERSION
    parts = [int(x) for x in v.split(".")[:3]]
    t = tuple(parts + [0] * (4 - len(parts)))
    text = (
        "VSVersionInfo(\n"
        f"  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),\n"
        "  kids=[StringFileInfo([StringTable('040904B0', [\n"
        "    StringStruct('FileDescription', 'No-Nonsense Photo Sorter'), StringStruct('FileVersion', '%s'),\n"
        "    StringStruct('InternalName', 'No-Nonsense Photo Sorter'), StringStruct('OriginalFilename', 'No-Nonsense Photo Sorter.exe'),\n"
        "    StringStruct('ProductName', 'No-Nonsense Photo Sorter'), StringStruct('ProductVersion', '%s')])]),\n"
        "    VarFileInfo([VarStruct('Translation', [1033, 1200])])])\n" % (v, v))
    f = Path(build) / "version_info.txt"
    f.write_text(text, encoding="utf-8")
    return f


def clean_build_folder(path):
    path = Path(path).resolve()
    root = (HERE / "build_files").resolve()
    if path == root or not path.is_relative_to(root):
        raise RuntimeError(f"Refusing to clean outside the build directory: {path}")
    if path.exists():
        shutil.rmtree(path)


def verify_tool(path):
    path = Path(path)
    expected = TOOL_SHA256[path.name.lower()]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Unexpected {path.name} build. See THIRD_PARTY_NOTICES.txt for the pinned version and hash.")
    return path


def get_ffmpeg():
    for directory in (HERE / "bin", HERE / "ffmpeg" / "bin"):
        ffmpeg, ffprobe = directory / "ffmpeg.exe", directory / "ffprobe.exe"
        if ffmpeg.is_file() and ffprobe.is_file():
            return verify_tool(ffmpeg), verify_tool(ffprobe)
    raise RuntimeError("Place the pinned ffmpeg.exe and ffprobe.exe in bin/ before building. See README.md.")


def get_icon(build):
    icon = HERE / "assets" / "nnps.ico"
    if not icon.is_file():
        raise RuntimeError("assets/nnps.ico is required to build the app")
    return icon


def main():
    build = HERE / "build_files"
    build.mkdir(exist_ok=True)
    ph.MODEL_DIR = build / "models"
    ph.ensure_models(lambda f, t: print(t, flush=True))
    exe = ph.find_exiftool()
    if not exe:
        raise RuntimeError("Could not find exiftool.exe")
    verify_tool(exe)
    source, exif = Path(exe).parent, build / "exiftool"
    clean_build_folder(exif)
    exif.mkdir()
    shutil.copy2(exe, exif / "exiftool.exe")
    if (source / "exiftool_files").is_dir():
        shutil.copytree(source / "exiftool_files", exif / "exiftool_files")
    else:
        for dll in source.glob("*.dll"):
            shutil.copy2(dll, exif / dll.name)
    ffmpeg, ffprobe = get_ffmpeg()
    tools = build / "ffmpeg"
    clean_build_folder(tools)
    tools.mkdir()
    shutil.copy2(ffmpeg, tools / "ffmpeg.exe")
    shutil.copy2(ffprobe, tools / "ffprobe.exe")
    dist = HERE.parent
    out = dist / "No-Nonsense Photo Sorter"
    if out.resolve() != out.absolute() or not out.resolve().is_relative_to(HERE.parent.resolve()):
        raise RuntimeError("Output must stay inside this workspace without redirects")
    print(f"Building No-Nonsense Photo Sorter {ph.APP_VERSION}...", flush=True)
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
                           "--name", "No-Nonsense Photo Sorter", "--contents-directory", "bin", "--icon", str(get_icon(build)),
                           "--version-file", str(make_version_file(build)),
                           "--add-data", f"{build / 'models'};models", "--add-data", f"{exif};exiftool",
                           "--add-data", f"{tools};ffmpeg", "--add-data", f"{HERE / 'fonts'};fonts",
                           "--add-data", f"{HERE / 'assets'};assets", "--distpath", str(dist),
                           "--workpath", str(build / "work"), "--specpath", str(build), str(HERE / "photo_helper.py")])
    import PySide6
    qt = Path(PySide6.__file__).parent
    # Python 3.11's top-level VC runtime is older than Qt requires.
    for runtime in qt.glob("*140*.dll"):
        shutil.copy2(runtime, out / "bin" / runtime.name)
    # Qt uses the Windows ICU adapter. A collected ICU 78 library has the same
    # filename but different exports and shadows the working System32 adapter.
    incompatible_icu = out / "bin" / "icuuc.dll"
    incompatible_icu.unlink(missing_ok=True)
    for name in ("FACE_WORKFLOW.md", "FACE_PERFORMANCE.md", "LICENSE", "THIRD_PARTY_NOTICES.txt"):
        shutil.copy2(HERE / name, out / name)
    print("Testing the packaged executable...", flush=True)
    report = out / "build-smoke.json"
    checked = subprocess.run([str(out / "No-Nonsense Photo Sorter.exe"), "--smoke-test", str(report)],
                             capture_output=True, timeout=90, creationflags=ph.NO_WINDOW)
    if checked.returncode or not report.exists() or not json.loads(report.read_text())["passed"]:
        raise RuntimeError("Packaged checks failed: " + checked.stderr.decode("utf-8", "replace"))
    print(f"DONE: verified No-Nonsense Photo Sorter {ph.APP_VERSION}\n{out}", flush=True)


if __name__ == "__main__":
    main()
