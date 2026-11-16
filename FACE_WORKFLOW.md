# Remembered people and face review

Options includes:

- **Remember people I confirm in this photo folder** (on by default). Only names you type or explicitly accept become approved examples. Automatic matches are never added as examples. Examples stay in `_Photo Helper/people.json` inside that library and are reused on later runs. Turning this off disables both remembering and reuse; it does not erase existing examples.
- **Review uncertain faces before saving** (off by default). After scanning and initial naming, uncertain faces show a crop and suggested names. Select or type a name and choose **Accept name**, choose **Leave unnamed**, or **Skip remaining**. Closing the review window skips the rest. Similarity numbers are not probabilities. Rejected faces stay unnamed during that run.
- **Use calibrated face matching when available** (on by default). Calibration can make matching stricter; it never weakens the selected Strict/Normal/Loose threshold.

To calibrate, first confirm at least three different samples for each of at least two people. Click **Calibrate face matching…** in Options and inspect the result before applying it. Each sample is matched against the other examples, excluding itself. The result measures agreement within the approved examples, not accuracy on unseen photos. Include different ages, angles and lighting when confirming examples. Adding or correcting examples invalidates the previous calibration; run calibration again to update it.

Calibration is stored in `_Photo Helper/face-calibration.json` and applies only to the matching approved examples and recognition model. Face annotation backups and existing manual region protection continue to apply.

Every completed face scan appends measurements to `_Photo Helper/face-timings.jsonl` and a readable summary to `Face Report.txt`. These include total time, hashing time and bytes read, cache hits and scan attempts. Compare runs on the same library and settings to measure repeat-scan performance. Full photo hashing remains enabled to prevent cache collisions.

See [FACE_PERFORMANCE.md](FACE_PERFORMANCE.md) for the synthetic benchmark and its limits.

## Remove face data

Choose **Remove face data from images…** in Options, select a folder, then inspect the affected files before confirming. This scans subfolders, including `_Duplicates` and `_Verify`, plus XMP sidecars. It removes standard MWG, Microsoft, ACDSee and Apple face/region annotations, IPTC person fields, `People/` and `People|` tags, and keyword values matching the names found in those face annotations. Unrelated keywords, capture dates and image pixels are preserved. Arbitrary names in descriptions, proprietary maker-note data and other applications' private databases are outside this cleanup.

The confirmation has **Reset face memory so I can name everyone again**, selected by default. It archives remembered names, calibration and detection caches so the next run starts fresh. This does not block images or prevent new face metadata from being written. Images that changed since the preview are skipped. Failures are listed in `_Photo Helper/Face Removal Report.txt`.

After removal, choose **Scan faces again** to rescan the same folder and enter corrected names. This rescan asks about every detected person group, even a group with just one face. The normal **Name faces** job also remains available for later runs. Keep the reset checkbox selected when you want to discard old spellings and start over.

Original images and archived remembered face data are kept under `_Photo Helper/face-removal-backups`. The normal move Undo button does not restore image metadata; use the original image copies if you need to restore it. Back up the photo folder separately before processing. These backups still contain the removed information; remove them yourself after verifying the cleaned images if permanent erasure is required.
