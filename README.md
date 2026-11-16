# No-Nonsense Photo Sorter

No-Nonsense Photo Sorter (NNPS) is a Windows desktop app for organizing a local photo and video folder. It can find exact copies and likely look-alikes, sort media by date, detect and name faces, gather photos of selected people, check for damaged pictures, compress videos, and remove saved face names. Review the results and keep a backup of your photo library before running jobs that move or edit files.

## Run from source

1. Install Python 3.11 on Windows.
2. Create a virtual environment and install `requirements.txt`:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Supply `Exiftool/exiftool.exe` and its accompanying `exiftool_files` directory, or install ExifTool separately so it is on `PATH`.
4. Supply `bin/ffmpeg.exe` and `bin/ffprobe.exe` for video tasks, or install FFmpeg separately so it is on `PATH`.
5. Start the app with `.\.venv\Scripts\python.exe photo_helper.py`.

The face models are downloaded from a pinned OpenCV Zoo revision on first use if they are absent. Their SHA-256 hashes are checked by the app. The app otherwise processes local files. Paths to external tools and app options are stored in a local `config.json`. Face examples, reports, move history, and backups are kept in `_Photo Helper` within the chosen photo folder. That folder name is retained so existing libraries remain readable after the product rename.

## Build a portable app

Place the pinned ExifTool 13.60 Windows package in `Exiftool/` and the pinned FFmpeg 2026-10-29 essentials build's `ffmpeg.exe` and `ffprobe.exe` in `bin/`. See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) for hashes and sources. Run `Build NNPS.bat`. The script installs the pinned direct Python dependencies and runs `build.py`, which creates the portable `../No-Nonsense Photo Sorter` folder and runs packaged smoke checks. A Python installation is required to build, but not to run the packaged app.

The build script verifies the local tool hashes and the app verifies its pinned model downloads. Bundled binary tools and face models are deliberately excluded from this source repository. Review their licenses and redistribution terms before attaching a portable build to a release.

## Undo and backups

The app records copy, sort, and broken-file moves in `_Photo Helper/moves.jsonl`. Use **Undo last recorded moves** in the app to reverse them. Metadata edits and video replacement are not covered by that undo log. Replaced video originals are retained in `_Uncompressed`. Face naming backups are in `_Photo Helper/name-backups`. Face removal copies each affected original to `_Photo Helper/face-removal-backups` before editing it. Keep a separate backup of the photo folder too, and check the status and report files after a run.

## Development

Tests live in `tests/` and use `pytest`. In a development environment, install `requirements-dev.txt`, then run `python -m pytest tests -q`. Some integration tests call the local ExifTool binary; supply the pinned build described above before running the full suite.

## License

The app source is distributed under [GPL-3.0](LICENSE). Bundled font notices are in `fonts/`. ExifTool, FFmpeg, and OpenCV models are third-party components with their own terms.
