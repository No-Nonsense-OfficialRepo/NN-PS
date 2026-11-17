# Changelog

All notable changes to No-Nonsense Photo Sorter. Add one line per change under "Unreleased";
`bump_version.py` files them under the new version when you release.

## Unreleased

- Matched NNLL's button hover behavior with a bottom-only fading trace, stronger brass glow, and animated label emphasis; removed the full outline and clipped full-width scaling.
- Replaced the Sort Photos/Options title-bar tabs with one Options popup; kept the main screen in place and enlarged the title-bar NN logo.
- Reworked the main layout to match the compact NNLL and NNVLA shell; added shared control icons, fading brass button traces, two-column task selection, and a welcome page that fits at the default size.
- Matched NNLL and NNVLA with a frameless, draggable NNPS title bar, custom window controls, resize grip, and brass outlined chamfered buttons.
- Renamed the product to No-Nonsense Photo Sorter (NNPS) and added the supplied marble NN monogram to the app and executable.

- Added GitHub source ignore rules, setup documentation, and pinned direct build dependencies.
- Restricted configured FFmpeg and ExifTool paths to existing executables with the expected names.
- Added verified originals for face removal, plus FFprobe video checks and journaled broken-file moves.
- Made ExifTool cancellation stop its child process during shutdown.
- Pinned the FFmpeg, FFprobe, and ExifTool build inputs by SHA-256; portable builds now include FFprobe and third-party notices.
- Clarified the scope of move undo and archived local-only audit notes from the source selection.

## 0.7.1 - 2026-11-11
- Remove names no longer copies and hash-verifies every image into a backup folder; ExifTool edits in place (-overwrite_original).
- Fixed the ExifTool "could not complete the metadata operation" error: reading file details no longer fails the whole run when ExifTool exits with code 1 for files with errors or IPTCDigest warnings.
- Interface rework: No-Nonsense palette (Obsidian / Marble / Classical Gold), left sidebar, Options page with Sorting, Face/Name and Video sections.
- Remove names is now its own task; "Everything" renamed to "Everything (copies, sort, names)".
- Typography kit applied: Cormorant Garamond, Cinzel and Inter bundled (SIL Open Font License) and used offline.
- Fixed checkboxes resizing on click and overlapping rows; main screen fits the default window.
- Better text contrast for hints, captions and the version label.
- The .exe now carries its version in Properties > Details.
- Added CHANGELOG.md and bump_version.py (Bump Version.bat).

## 0.7.0
- Starting point for this changelog.
