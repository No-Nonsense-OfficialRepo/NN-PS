"""Isolated checks invoked by the packaged app's --smoke-test option."""
import json
import os
from pathlib import Path
import tempfile
import traceback


def run(ph, destination):
    output = Path(destination).resolve()
    result = {"version": ph.APP_VERSION, "checks": [], "passed": False}
    try:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        with tempfile.TemporaryDirectory(prefix="photo-helper-check-", dir=output.parent) as directory:
            root = Path(directory)
            ph.CONFIG = root / "config.json"
            app = ph.QApplication.instance() or ph.QApplication([])
            window = ph.Main()
            window.face_option_boxes["review_faces"].setChecked(True)
            assert ph.load_config()["options"]["review_faces"] is True
            window.reset_options()
            assert not ph.load_config()["options"]["review_faces"]
            result["checks"].append("Qt window and saved Options")
            ph.ensure_models(lambda *_: None)
            path = root / "photo.jpg"
            assert ph.cv2.imwrite(str(path), ph.np.zeros((100, 100, 3), ph.np.uint8))
            assert ph.scan_one(path) == []
            result["checks"].append("verified models and real blank-image detection")
            exe = ph.find_exiftool()
            assert exe and Path(exe).is_relative_to(ph.RES_DIR), "Bundled ExifTool missing"
            ffmpeg = ph.find_ffmpeg()
            assert ffmpeg and Path(ffmpeg).is_relative_to(ph.RES_DIR), "Bundled FFmpeg missing"
            ff = ph.subprocess.run([ffmpeg, "-version"], capture_output=True, creationflags=ph.NO_WINDOW)
            assert ff.returncode == 0
            ffprobe = ph.find_ffprobe()
            assert ffprobe and Path(ffprobe).is_relative_to(ph.RES_DIR), "Bundled FFprobe missing"
            probe = ph.subprocess.run([ffprobe, "-version"], capture_output=True, creationflags=ph.NO_WINDOW)
            assert probe.returncode == 0
            result["checks"].append("bundled FFmpeg and FFprobe start")
            ph.run_exif(exe, ["-Make=Test Camera", "-DateTimeOriginal=2020:05:01 12:00:00", str(path)])
            duplicate = root / "photo (1).jpg"
            ph.shutil.copy2(path, duplicate)
            ph.start_move_run(root)
            prepared = ph.stage_prepare(root, exe, lambda *_: None, ("copies", "sort"))
            deep = ph.stage_deep(root, exe, prepared["groups"], prepared["current"], prepared["cache"], lambda *_: None, ("copies", "sort"))
            path = root / "Sorted" / "2020" / "2020-05" / "photo.jpg"
            assert path.exists() and deep["dups"] == 1
            result["checks"].append("bundled ExifTool, sorting and exact copies")
            embedding = ph.np.eye(128, dtype=ph.np.float32)[0]
            face = ph.Face(str(path), (.1,.1,.2,.2), (100,100), embedding, 1., b"thumbnail")
            data = {"root": str(root), "faces": [face], "members": [[0]], "confirmed": {0: "Old spelling"}}
            refs = ph.remember_person(root, "Old spelling", embedding)
            assert ph.stage_match(data, refs, {}, exe, lambda *_: None)["tagged"] == 1
            plan = ph.prepare_face_removal(root, exe, lambda *_: None)
            cleared = ph.remove_face_data(plan, exe, lambda *_: None)
            assert cleared["removed"] == 1 and cleared["failed"] == 0
            assert not ph.load_people(root)
            assert ph.prepare_face_rescan(root, exe, lambda *_: None)["rename_all"]
            data["confirmed"] = {0: "Correct spelling"}
            refs = ph.remember_person(root, "Correct spelling", embedding)
            assert ph.stage_match(data, refs, {}, exe, lambda *_: None)["tagged"] == 1
            names = ph.exif_json(exe, ["-PersonInImage"], [path])[0]["PersonInImage"]
            assert names == "Correct spelling"
            result["checks"].append("face save, backup, wipe, rescan and corrected name")
            window.phase = "welcome"
            window.close()
            app.processEvents()
            result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["passed"] else 1
