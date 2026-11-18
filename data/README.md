# OBSCURA: Multi-Target Atmospheric Prediction Around Industrial Sites

For each anonymized site-window, predict **five** quantities jointly:
satellite methane, nitrogen dioxide, carbon monoxide, absorbing
aerosol, and fire/flare thermal intensity, from a Sentinel-2 image
chip and weather/terrain/surface features.

## Targets

All targets are measured over a fixed 10 km window centred on the
site (Sentinel-5P, ~1.1 km grid). Gas enhancements are the window mean
minus the median of a 30-75 km background ring. Every target is released as a **z-score**
computed with train-set statistics that are not published; units are
therefore hidden and every target is on a comparable scale.

**Missing targets are blank.** A satellite gas product is often
unavailable (cloud, quality filtering). A blank target is not scored
for that row. Predict all five columns for every test row anyway.

## Metric

Masked mean column-wise RMSE (lower is better): for each target take
the RMSE over the test rows where that target was measured, then
average the five RMSEs. Predicting the train mean for every target
scores close to 1.0 on rows that look like the training data. On
held-out facilities it scores higher, because their targets are
shifted relative to the training mean; use it as the baseline to beat.

## Files

- `train.csv` (3218 rows): `sample_id`, `site_key`, five targets, features.
- `test.csv` (1221 rows): `sample_id` and features only.
- `sample_submission.csv`: `sample_id` plus the five target columns.
- `images/<sample_id>.tif`: 384x384 px, uint16, 14 bands, no georeferencing and no metadata.
  Band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, SCL, CLDPRB.
  B1-B12 are Sentinel-2 L2A surface reflectance (x1e-4 = reflectance);
  SCL is the scene classification; CLDPRB the s2cloudless cloud
  probability (0-100). Chips are NOT cloud-masked.
  Each site's frame was curated to its facility, so physical extent
  differs between sites and is not disclosed.

## Column dictionary

| column | unit | description |
|---|---|---|
| sample_id | - | Anonymized sample identifier (matches images/<sample_id>.tif) |
| site_key | - | train.csv only. Anonymized site group: use it for GroupKFold. Test sites never appear in train. |
| t_ch4 | z-score | XCH4 enhancement: Sentinel-5P site mean minus regional background median |
| t_no2 | z-score | Tropospheric NO2 column enhancement (site minus background) |
| t_co | z-score | CO column enhancement (site minus background) |
| t_aai | z-score | Absorbing aerosol index over the site window |
| t_firms | z-score | Thermal-anomaly (fire/flare) intensity; no detection is a real, low value |
| month_sin / month_cos | - | Cyclic month-of-year encoding (month only; no day) |
| temperature | degC | ERA5-Land 2 m air temperature, daily mean (quantized 0.5) |
| humidity | % | Relative humidity from ERA5-Land dewpoint, daily mean (quantized 2) |
| wind_speed | m/s | ERA5-Land 10 m wind speed near satellite overpass (mean of hourly speeds, quantized 0.5) |
| wind_direction | deg | Meteorological wind direction near overpass (quantized 10) |
| precipitation | m/day | ERA5-Land total precipitation, daily sum (quantized 0.0005) |
| solar_rad | MJ/m2/day | ERA5-Land downward surface solar radiation (quantized 0.5) |
| elevation | m | Copernicus GLO-30 mean elevation over the chip (quantized 50) |
| slope | deg | Mean terrain slope over the chip (quantized 1) |
| s2_cloud_frac | 0-1 | Fraction of chip pixels with cloud probability >= 40 |
| ndvi / evi / savi / ndre / ndwi / mndwi ... | - | Spectral indices computed from the cloud-masked chip (chip means) |

## Splits and rules

- Train, public test and private test share nothing: they contain different
  facilities (far enough apart that their satellite background areas do not
  overlap) and they come from different, non-overlapping stretches of time.
  The private test is the latest.
- Some test rows are scored on the public leaderboard, the rest only at
  the end (private).
- Do not try to identify sites, locations or dates, and do not use
  outside satellite, weather or methane data for any row. Violations
  invalidate a submission.

## Known limitations

- Enhancements are small relative to retrieval noise; negative z-scores
  are normal. Treat targets as noisy.
- Targets describe coarse column quantities, not facility emission rates.
- Weather features can be missing near coasts.

## Attribution

- Contains modified Copernicus Sentinel data (Sentinel-2, Sentinel-5P),
  2022-2026, processed by ESA.
- FIRMS (NASA); ERA5-Land (C3S/ECMWF); Copernicus DEM GLO-30 (ESA).
- Data access via Google Earth Engine.

Dataset pipeline version 3.0.0.
