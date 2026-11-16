# An empirical cosmic-ray library for H2RG detectors

A library of cosmic-ray events measured on the ground with Teledyne
HAWAII-2RG (H2RG) infrared detectors. Each event is a 31×31 pixel stamp in
electrons, with its charge, size and shape, plus the measured event rate
and the fraction of the library that is noise.

The library was built to inject realistic cosmic rays into simulated data
for an H2RG detector (the GIRMOS data simulator). An early version was used
to test CR rejection during UTR fitting, the results are published in an 
[SPIE proceedings paper](https://doi.org/10.1117/12.3105241). To our
knowledge, no empirical cosmic-ray library from ground-based data exists for
HxRG detectors; existing libraries and models come from space missions. For
silicon CCDs at optical wavelengths, see Groom (2002, Exp. Astron. 14, 45-55).

The events come from dark frames of **GSAOI**, the Gemini South Adaptive
Optics Imager, on the 8.1 m Gemini South telescope at Cerro Pachón, Chile
(~2,700 m). GSAOI images at 0.9–2.5 µm on a mosaic of four 2048×2048 H2RG
arrays (18 µm pixels, 2.5 µm cutoff).

<p align="center">
  <img src="figures/fig_example_stamps.png" width="80%"
       alt="Nine example cosmic-ray stamps from the library">
</p>

## Release contents

The library files are in this repository, compressed (.fits.gz); astropy opens them directly.

| File | Charge cut | Events | Noise (`CONTAM`) | Rate (events cm⁻² min⁻¹) |
|---|---|---|---|---|
| `cr_library_h2rg_1pct.fits` | 800 e⁻ | 39,029 | 1.0% | 0.91 |
| `cr_library_h2rg_10pct.fits` | 350 e⁻ | 47,627 | 9.5% | 1.11 |

The two files contain the same events above their own cut. Use `1pct` for
clean cosmic rays. Use `10pct` to include fainter events, mixed with some
noise, for example to test rejection algorithms near the noise level.

Each file has three extensions:

- **`STAMPS`**: an (N, 31, 31) cube in electrons, sorted by charge
  (brightest first). Pixels outside the event are 0.
- **`METADATA`**: one row per stamp. `CHARGE` (e⁻, the whole event),
  `NPIX` (pixels in the event), `NCORE` (pixels above 5σ), `ELONGATION`
  (of the core pixels; NaN below 6 core pixels), `CHIP`, `EPOCH` (row of
  `EPOCHS`), `FRAME`, `CX`, `CY`.
- **`EPOCHS`**: the dark sequences used: date, observation ID, `EXPTIME`,
  `LNRS`, `COADDS`, number of darks.

Main header keywords: `CR_RATE` (events cm⁻² s⁻¹), `CHG_CUT` (e⁻),
`CONTAM` and `CONTAMER` (noise fraction and its error), `TOT_EXP` (s),
`SRCH_ARE` (cm²), `CHIPS`, `LNRS`, `STAMPSZ`. Multiply `CR_RATE` by the
detector area (cm²) and the exposure time (s) to get the expected number of
events in a frame.

## Input data

All GSAOI dark frames with exposure times longer than 50 s available in
the Gemini Observatory Archive as of November 2026. The sequences used are
listed in the `EPOCHS` extension of each file.

## Running the extraction

`extract_cr_library.py` needs Python 3 with `numpy`, `scipy`, `astropy` and
`matplotlib`.

1. Put the darks in a folder `Darks/` next to the script, and the GSAOI bad
   pixel mask `bpm_20121104_gsaoi_gsaoi_11_full_4amp.fits` (from the Gemini archive)
   next to it.

2. Adjust the settings at the top of the script if needed: `LIBRARIES`
   (read modes, chips, coadds and minimum exposure of each library),
   `CUT_TARGETS` (noise fractions for the cuts) and the detection settings.
3. Run `python extract_cr_library.py`.

The script prints the sequences it found and, for each library, the
contamination at several cuts. It writes `cr_library_<library>_<cut>.fits`
and a diagnostic plot `charge_distribution_<library>.png` per library. As
distributed it also builds a `comparison` library (all chips and
sequences), used for the figure below. Processing all the darks takes about
5 minutes in a 16 GB RAM MacOS .

## Method

1. **Group the darks into sequences**: same night, observation, exposure
   time, read mode (`LNRS`) and coadds. The first dark of each sequence is
   dropped.
2. **Subtract a reference**: for each sequence and chip, the median of its
   darks. The noise σ of each frame is measured from the median absolute
   deviation.
3. **Detect events**: an event starts at a pixel above 5σ and grows through
   touching pixels (corners included) above 3σ, so its faint edge (charge
   spread by diffusion and inter-pixel capacitance) is part of it.
4. **Measure the noise**: the same detection runs on the negated frame.
   Cosmic rays only add charge, so these negative events are noise only.
5. **Flag** events whose pixels fire repeatedly (see below).
6. **Build the library**: join the sequences, choose the charge cuts from
   the contamination (next section), and save the stamps.

### Applied to every library

| Step | Setting | Why |
|---|---|---|
| Bad pixel mask, plus a corner of chip 1 | `BPM_FILE`, `EXTRA_BAD` | In the corner, noise reaches cosmic-ray charges: as many negative as positive events above the cut. |
| First dark of each sequence dropped | `SKIP_FIRST_FRAMES = 1` | Detector settling after reset: it held ~80% of the large noise events. |
| Coadd correction | — | GSAOI *averages* its coadds, so charge and exposure are multiplied by `COADDS`. |
| Recurrence filter | `FILTER_RECURRING` | An event whose core pixels pass 5σ, up or down, in more than one frame is an unstable pixel, not a cosmic ray. Costs ~1% of real events. |
| Persistence rescue | `RESCUE_PERSISTENCE` | Keeps the first event of a bright hit whose afterglow made its pixels recur. |
| Edge margin | `STAMP_HALFSIZE = 15` | The 31×31 stamp must fit on the detector. |
| Size limit | `MAX_PIXELS = 50` core pixels | Larger events do not fit in a stamp (~2% of events). |

There is **no shape filter**. Rejecting round events (meant for clusters of
hot pixels) removed ~10–13% of the events, almost all of them compact
cosmic-ray hits, and no noise. Clusters of hot pixels recur, so the
recurrence filter already removes them.

### Chosen per library

| Setting | Options | This release |
|---|---|---|
| `lnrs` | read modes (number of Fowler read pairs) | all (2, 8, 16) |
| `chips` | 1–4 | 1, 2, 3 |
| `coadds` | coadd values | all |
| `min_exptime` | minimum exposure time (s) | 60 |
| `CUT_TARGETS` | noise fractions for the cuts | 1%, 10% |

## Contamination and charge cuts

The noise in the positive events is estimated from the negative ones, which
go through the same detection and filters. Above a charge *c*, the
contamination is

<p align="center">
  CONTAM(<i>c</i>) = N<sub>neg</sub>(≥ <i>c</i>) / N<sub>pos</sub>(≥ <i>c</i>)
</p>


and each cut is the lowest charge from which `CONTAM` stays below its
target. One cut is used for all the sequences, so the charge distribution
of the library is not biased towards any read mode or exposure time.

The comparison below shows why chip 4 and the 50 s darks are left out:

<p align="center">
  <img src="figures/fig_library_comparison.png" width="80%"
       alt="Cosmic-ray rate vs charge for the library and for subsets">
</p>

- **Read modes and exposure times agree** above ~2 ke⁻. Below it, short
  exposures with many reads (50 s, `LNRS` 16) have up to 2.5× more events:
  with Fowler sampling, a cosmic ray that lands during a read group is
  recorded with only part of its charge.
- **Chip 4 differs**: more noise at low charge, a ~20–40% higher rate and
  a peak at ~1.7× higher charge, so it is left out.

Rates are per unit area, time and decade (dex) of charge.

## Limitations

- Below ~1–2 ke⁻ the library still holds some partially recorded events and
  noise (10–30% above the cleanest subset).
- Stamps include the GSAOI read noise of their pixels.
- Events with more than 50 core pixels (~2%) are not included, and a few
  of the largest stamps are clipped at 31×31.
- The noise estimate cannot see artefacts that only add charge.
- The rate is measured at ~2,700 m and will differ at other altitudes.

## Authors

Gaia Gaspar & Marcin Sawicki
