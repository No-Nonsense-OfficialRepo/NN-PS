from pathlib import Path

import cv2
import numpy as np


def annotated(ph, path):
    exe = str(Path(ph.APP_DIR) / "Exiftool" / "exiftool.exe")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((100, 100, 3), np.uint8))
    ph.run_exif(exe, ["-XMP-mwg-rs:RegionInfo={AppliedToDimensions={W=100,H=100,Unit=pixel},RegionList=[{Area={X=0.2,Y=0.2,W=0.2,H=0.2,Unit=normalized},Name=Alice,Type=Face}]}",
                       "-XMP-iptcExt:PersonInImage=Alice", "-XMP-dc:Subject=Alice", "-XMP-dc:Subject+=Holiday",
                       "-XMP-digiKam:TagsList=People/Alice", "-XMP-digiKam:TagsList+=Albums/Holiday",
                       "-XMP-lr:HierarchicalSubject=People|Alice", "-XMP-lr:HierarchicalSubject+=Places|Beach",
                       "-IPTC:Keywords=Alice", "-IPTC:Keywords+=Beach", "-EXIF:DateTimeOriginal=2020:05:01 12:00:00", str(path)])
    return exe


def test_removal_preserves_other_metadata_pixels_and_backups(ph, tmp_path):
    path = tmp_path / "_Duplicates" / "photo.jpg"
    exe = annotated(ph, path)
    before = path.read_bytes()
    pixels = cv2.imread(str(path))
    notes = tmp_path / ph.INFO_DIR
    notes.mkdir()
    (notes / "people.json").write_text('{"remembered":true}')
    (notes / "faces.jsonl").write_text("cache")
    plan = ph.prepare_face_removal(tmp_path, exe, lambda *_: None)
    assert len(plan["entries"]) == 1
    assert path.read_bytes() == before  # preview does not edit images
    result = ph.remove_face_data(plan, exe, lambda *_: None)
    assert result["removed"] == 1 and result["failed"] == 0
    assert result["forgotten"] == 2
    assert not (notes / "people.json").exists()
    assert any(p.read_bytes() == before for p in Path(result["backups"]).rglob("photo.jpg"))
    current = ph.exif_json(exe, ["-G1", "-a", "-XMP:all", "-IPTC:Keywords", "-EXIF:DateTimeOriginal"], [path])[0]
    assert not ph.face_removal_values(current)[0]
    assert current["XMP-dc:Subject"] == "Holiday"
    assert current["XMP-digiKam:TagsList"] == "Albums/Holiday"
    assert current["XMP-lr:HierarchicalSubject"] == "Places|Beach"
    assert current["IPTC:Keywords"] == "Beach"
    assert current["ExifIFD:DateTimeOriginal"] == "2020:05:01 12:00:00"
    np.testing.assert_array_equal(cv2.imread(str(path)), pixels)
    assert ph.prepare_face_removal(tmp_path, exe, lambda *_: None)["entries"] == []


def test_changed_file_is_not_edited_and_forget_is_optional(ph, tmp_path):
    path = tmp_path / "photo.jpg"
    exe = annotated(ph, path)
    notes = tmp_path / ph.INFO_DIR
    notes.mkdir()
    memory = notes / "people.json"
    memory.write_text("remembered")
    plan = ph.prepare_face_removal(tmp_path, exe, lambda *_: None)
    path.write_bytes(path.read_bytes() + b"changed")
    changed = path.read_bytes()
    result = ph.remove_face_data(plan, exe, lambda *_: None, forget=False)
    assert result["failed"] == 1 and result["removed"] == 0
    assert memory.exists() and path.read_bytes() == changed


def test_file_failure_does_not_stop_later_images(ph, tmp_path, monkeypatch):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    exe = annotated(ph, a)
    annotated(ph, b)
    plan = ph.prepare_face_removal(tmp_path, exe, lambda *_: None)
    real = ph.run_exif
    def fail_one(exe, args, **kwargs):
        if "-execute" in args and str(a) in args:
            # The editor batches files; simulate ExifTool skipping the first edit.
            commands, start = [], 0
            for end, arg in enumerate(args):
                if arg == "-execute":
                    command = args[start:end + 1]
                    if str(a) not in command:
                        commands.extend(command)
                    start = end + 1
            args = commands
        return real(exe, args, **kwargs)
    monkeypatch.setattr(ph, "run_exif", fail_one)
    result = ph.remove_face_data(plan, exe, lambda *_: None, forget=False)
    assert result["failed"] == 1 and result["removed"] == 1


def test_sidecars_are_included_but_backup_folders_are_excluded(ph, tmp_path):
    exe = str(Path(ph.APP_DIR) / "Exiftool" / "exiftool.exe")
    sidecar = tmp_path / "photo.xmp"
    sidecar.write_text('''<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:iptcExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"><iptcExt:PersonInImage><rdf:Bag><rdf:li>Alice</rdf:li></rdf:Bag></iptcExt:PersonInImage></rdf:Description></rdf:RDF></x:xmpmeta>''')
    backups = tmp_path / ph.INFO_DIR / "name-backups"
    backups.mkdir(parents=True)
    hidden = backups / "photo.xmp"
    hidden.write_bytes(sidecar.read_bytes())
    plan = ph.prepare_face_removal(tmp_path, exe, lambda *_: None)
    assert plan["checked"] == 1 and len(plan["entries"]) == 1
    result = ph.remove_face_data(plan, exe, lambda *_: None, forget=False)
    assert result["removed"] == 1 and result["failed"] == 0
    assert "Alice" in hidden.read_text()


def test_other_standard_face_namespaces_are_removed(ph, tmp_path):
    path = tmp_path / "photo.jpg"
    exe = annotated(ph, path)
    ph.run_exif(exe, ["-XMP-MP:RegionPersonDisplayName=Bob",
                      "-XMP-acdsee-rs:RegionInfoACDSee={RegionList=[{Name=Carol,Type=Face}]}",
                      "-XMP-iptcExt:PersonInImageWDetails={PersonName=Dave,PersonId=person123}",
                      "-XMP-apple-fi:FaceID=123", "-XMP-dc:Subject+=Bob", "-XMP-dc:Subject+=Carol",
                      "-XMP-dc:Subject+=Dave", str(path)])
    plan = ph.prepare_face_removal(tmp_path, exe, lambda *_: None)
    assert len(plan["entries"]) == 1
    assert {"Alice", "Bob", "Carol", "Dave"} <= set(plan["entries"][0]["remove"]["XMP-dc:Subject"])
    result = ph.remove_face_data(plan, exe, lambda *_: None, forget=False)
    assert result["removed"] == 1 and result["failed"] == 0
    record = ph.exif_json(exe, ["-G1", "-XMP:all"], [path])[0]
    assert record["XMP-dc:Subject"] == "Holiday"


def test_reset_can_rescan_and_write_a_corrected_name(ph, tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    exe = annotated(ph, path)
    e = np.eye(128, dtype=np.float32)[0]
    ph.remember_person(tmp_path, "Alice", e)
    ph.append_face_cache(tmp_path, {ph.sha256(path): [((.1,.1,.2,.2), (100,100), e, 1., b"thumb")]})
    plan = ph.prepare_face_removal(tmp_path, exe, lambda *_: None)
    assert ph.remove_face_data(plan, exe, lambda *_: None)["failed"] == 0
    assert ph.load_people(tmp_path) == {} and ph.load_face_cache(tmp_path) == {}
    calls = []
    monkeypatch.setattr(ph, "ensure_models", lambda *_: None)
    monkeypatch.setattr(ph, "meme_set", lambda *a, **k: set())
    def scan(p):
        calls.append(p)
        return [((.1,.1,.2,.2), (100,100), e, 1., b"thumb")]
    monkeypatch.setattr(ph, "scan_one", scan)
    data = ph.prepare_face_rescan(tmp_path, exe, lambda *_: None)
    assert calls == [path] and len(data["faces"]) == 1 and data["rename_all"]
    data["confirmed"] = {0: "Alicia"}
    refs = ph.remember_person(tmp_path, "Alicia", e)
    result = ph.stage_match(data, refs, {}, exe, lambda *_: None)
    assert result["tagged"] == 1 and result["failed"] == 0
    current = ph.exif_json(exe, ["-PersonInImage", "-RegionName", "-Subject"], [path])[0]
    assert current["PersonInImage"] == "Alicia" and current["RegionName"] == "Alicia"
    assert "Alice" not in ph._as_list(current["Subject"])
