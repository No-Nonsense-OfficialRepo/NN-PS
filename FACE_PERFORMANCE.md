# Synthetic face scan benchmark

12 generated JPEGs at 1600 x 1200; actual bundled OpenCV models; balanced accuracy. No real faces or user photos.

| Scan | Total seconds | Hash seconds | Scan attempts | Cache hits |
|---|---:|---:|---:|---:|
| First | 0.417 | 0.070 | 12 | 0 |
| Repeat | 0.007 | 0.007 | 0 | 12 |

The timing covers find_faces, excluding model verification, cache loading, sorting, and interactive naming. Synthetic images are not a measure of recognition accuracy or real-library performance. Full content hashing remains enabled on repeat scans.
