"""
Extract cosmic ray events from GSAOI dark frames to build a CR library.

An empirical cosmic ray library for Teledyne H2RG near-infrared detectors,
built from GSAOI darks. The event stamps and the charge distribution are
measured rather than simulated, so the library can be injected into data
from any H2RG-based instrument.

Developed for the GIRMOS project.

Authors: Gaia Gaspar & Marcin Sawicki
Contact: gaiagaspar@gmail.com

If you use this library, please cite the link to this GitHube repository.



# ==============================================================================

Steps:
  1. Group the darks into epochs: one sequence each (night, observation,
     EXPTIME, LNRS).  Steps 2-5 run separately for each epoch.
  2. Median-combine per chip to build a clean reference
  3. Subtract reference; events start at pixels > DETECTION_SIGMA and
     grow through touching pixels > GROW_SIGMA, excluding masked pixels
  4. Do the same on the negated residual (noise only, see CUT_TARGETS)
  5. Extract postage stamps (electrons) of events above the lowest cut
  6. Each library in LIBRARIES joins the epochs with its read modes, on
     its chips, and is saved once per cut with its measured contamination

# ==============================================================================

Input data: GSAOI darks with the blocking filter, from the Gemini South
Archive, in DARK_DIR.  The epochs used are printed and stored in the
EPOCHS table of each library.
  - Bad pixel mask: bpm_20121104_gsaoi_gsaoi_11_full_4amp.fits.  Flagged
      pixels are removed before events are labelled, so a CR survives if
      at least one of its pixels is unflagged, but its NPIX and CHARGE
      then count only the unflagged ones.  A flagged pixel in the middle
      of a track splits it into two separate events.
  - Extra masked region (EXTRA_BAD): a noisy corner of chip 1 where,
      above the cuts, positive and negative events are equally common:
      its noise reaches cosmic-ray charges.

# ==============================================================================

Produces, for each library and cut, cr_library_<library>_<cut key>.fits,
with three HDUs: STAMPS (images), METADATA (one row per event) and EPOCHS
(the dark sequences used).  The libraries drop recurring events (see
FILTER_RECURRING); there is no shape filter (see ELONG_NPIX_MIN).  No uncut
library is saved: below the cuts the events are almost all noise.

Plus one diagnostic plot per library: charge_distribution_<library>.png.

# ==============================================================================

Usage:
  python extract_cr_library.py
"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy import ndimage
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────
DARK_DIR = Path(__file__).parent / "Darks"
BPM_FILE = (Path(__file__).parent /
            "bpm_20121104_gsaoi_gsaoi_11_full_4amp.fits")
OUTPUT_DIR = Path(__file__).parent

# Libraries to build: name -> read modes (LNRS values, None for all),
# chips (1-4) and, optionally, coadds (COADDS values) and min_exptime (s).
# Every library is saved once per cut, as cr_library_<name>_<cut key>.fits.
#   h2rg:       the library.  Chip 4 is left out (higher noise, more and
#               brighter events than chips 1-3).  min_exptime drops the 50 s
#               darks: a cosmic ray landing during a Fowler read group shows
#               only part of its charge, and with short exposures and many
#               reads (50 s, LNRS 16) these doubled the rate below 1 ke-.
#   comparison: every chip and dark set, not for release; 
LIBRARIES = {
    "h2rg": {"lnrs": None, "chips": (1, 2, 3), "min_exptime": 60},
    "comparison": {"lnrs": None, "chips": (1, 2, 3, 4)},
}
MIN_DARKS = 5                # skip epochs with fewer darks (poor median)
# The first dark of each sequence behaves differently (detector settling
# after reset/idle; DRAGONS also drops it for NIRI): it held ~80% of the
# large negative events and a matching excess of positive ones.  These
# many darks are dropped from the start of every sequence.
SKIP_FIRST_FRAMES = 1
SAVE_EPOCH_LIBRARIES = False  # also save one library per epoch

N_CHIPS = 4
# stamps are (2*h+1) x (2*h+1) = 31x31
STAMP_HALFSIZE = 15
DETECTION_SIGMA = 5.0        # an event starts at a pixel above this
# ...and grows through touching pixels above this above GROW_SIGMA, so the 
# faint edge of an event (charge spread by diffusion and IPC) stays part of it.  
GROW_SIGMA = 3.0
MAX_PIXELS = 50              # discard events with more core pixels (not CRs)
PIXEL_SIZE_CM = 18.0e-4      # 18 μm in cm
DETECTOR_NPIX = 2048
DETECTOR_AREA_CM2 = (DETECTOR_NPIX * PIXEL_SIZE_CM) ** 2
# Events closer than STAMP_HALFSIZE to the edge are discarded (the stamp
# would fall outside the array), so the area actually searched for CRs is
# smaller than the full detector.  Masked pixels inside it are not searched
# either; CR_RATE uses SEARCHED_AREA_CM2, which removes both (computed
# after loading the BPM).
EFFECTIVE_NPIX = DETECTOR_NPIX - 2 * STAMP_HALFSIZE
EFFECTIVE_AREA_CM2 = (EFFECTIVE_NPIX * PIXEL_SIZE_CM) ** 2

# Charge cuts, set by a contamination target.  Cosmic rays only add charge,
# while the noise is symmetric, so the same detection run on the negated
# residual finds noise only.  Above a cut, (negative events) / (positive
# events) estimates the fraction of the library that is noise (CONTAM).
# Each library gets, for each target, the lowest charge from which CONTAM
# stays <= target; one cut for all its epochs keeps its joined charge
# distribution unbiased.  The key is the file suffix:
# cr_library_<library>_<key>.fits
CUT_TARGETS = {"1pct": 0.01, "10pct": 0.10}
# Stamps are made during detection, before the cuts are known, so only
# above this charge (electrons); it must stay below every cut.
STAMP_MIN_CHARGE = 300.0

# Rows and columns (0-based, [y0:y1, x0:x1]) masked on top of the BPM, per
# chip.  The chip 1 corner holds as many negative as positive events above
# the cuts, in both epochs: noise, probably amplifier glow.
EXTRA_BAD = {0: (0, 280, 1880, 2048)}

# ELONGATION is written for events with at least this many core pixels
# (NaN below).  It only describes the library: there is no shape filter.
# One used to reject round events (>= 6 core px, elongation < 1.5), meant
# for hot-pixel clusters, but those recur and the recurrence filter removes
# them.  What it rejected was real charge (no negative events among it):
# compact straight-on hits and the diffuse blobs of chip 4, ~10-13% of the
# events above the cuts.
ELONG_NPIX_MIN = 6

# A cosmic ray is a random impact: it lands on a pixel that has no reason to
# light up again.  An event whose core pixels ALL pass DETECTION_SIGMA in
# more than one frame, in either direction, is therefore something the
# detector is doing (RTN, unstable pixels that jump up and down), not a
# cosmic ray.  Events keep a RECUR flag, and the filtered libraries drop the
# recurring ones when this is True.  The cost is the ~1% of genuine cosmic
# rays that happen to land on a pixel that fired in another frame.
FILTER_RECURRING = True
# A cosmic ray bright enough to leave an afterglow makes its own pixel
# recur, so the impact gets flagged together with its echoes.  Where the
# events on one pixel run over consecutive frames and the first is the
# brightest — an impact followed by decay — that first event is kept.
# On this data it recovers ~3% of the recurring groups.
RESCUE_PERSISTENCE = True

stamp_size = 2 * STAMP_HALFSIZE + 1
# 8-connectivity: pixels touching only at a corner belong to the same
# event.  scipy's default is 4-connectivity, which splits diagonal tracks.
CONNECTIVITY = ndimage.generate_binary_structure(2, 2)


# ── Helper functions ──────────────────────────────────────────────────

def _find(parent, a):
    """Union-find root, with path compression."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def find_epochs(dark_dir):
    """Group the darks into epochs: one night, observation, EXPTIME, LNRS,
    COADDS.

    Each epoch gets its own median reference and recurrence test, so it
    must be one homogeneous sequence.  Darks without the blocking filter
    are skipped, the first SKIP_FIRST_FRAMES of each sequence (in file
    order, which is time order) are dropped, and epochs left with fewer
    than MIN_DARKS frames are skipped.
    """
    groups = {}
    for f in sorted(dark_dir.glob("*.fits")):
        h = fits.getheader(f)
        if (h.get("OBSTYPE") != "DARK"
                or not str(h.get("FILTER1", "")).startswith("Blocked")):
            print(f"  skipped {f.name}: OBSTYPE={h.get('OBSTYPE')}, "
                  f"FILTER1={h.get('FILTER1')}")
            continue
        key = (h["DATE-OBS"], h.get("OBSID", ""), float(h["EXPTIME"]),
               int(h["LNRS"]), int(h.get("COADDS", 1)))
        groups.setdefault(key, []).append(f)
    epochs = []
    for (date, obsid, exptime, lnrs, coadds), files in sorted(
            groups.items()):
        files = sorted(files)[SKIP_FIRST_FRAMES:]
        if len(files) < MIN_DARKS:
            print(f"  skipped epoch {date} {obsid} {exptime:.0f} s "
                  f"LNRS {lnrs}: only {len(files)} darks")
            continue
        epochs.append({"index": len(epochs), "date": date, "obsid": obsid,
                       "exptime": exptime, "lnrs": lnrs, "coadds": coadds,
                       "files": files})
    return epochs


def load_darks(dark_files):
    """Load dark frames, return chips dict, gains dict, exptime."""
    chips = {c: [] for c in range(N_CHIPS)}
    gains = {}
    exptime = None
    for fpath in dark_files:
        hdul = fits.open(fpath)
        if exptime is None:
            exptime = hdul[0].header["EXPTIME"]
        for c in range(N_CHIPS):
            ext = c + 1
            chips[c].append(hdul[ext].data.astype(np.float32))
            if c not in gains:
                gains[c] = hdul[ext].header["GAIN"]
        hdul.close()
    for c in range(N_CHIPS):
        chips[c] = np.array(chips[c])
    return chips, gains, exptime


def _rescue_persistence(events):
    """Persistence rescue (see RESCUE_PERSISTENCE), in place.

    Keep the first event of a recurring group whose frames are consecutive
    and whose first event is the brightest, which is what an impact plus
    its decaying afterglow looks like.  Anything else stays flagged.
    Events belong to the same group when they share at least one core
    pixel, not when their centroids match: an afterglow is usually smaller
    than the impact that caused it, so its centroid has moved.
    Returns the number of events rescued.
    """
    parent = list(range(len(events)))
    owner = {}
    for i, ev in enumerate(events):
        for p in zip(ev["pys"].tolist(), ev["pxs"].tolist()):
            j = owner.setdefault(p, i)
            if j != i:
                _union(parent, j, i)
    groups = {}
    for i in range(len(events)):
        groups.setdefault(_find(parent, i), []).append(i)
    n_rescued = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group = [events[i] for i in members]
        if not all(e["recurring"] for e in group):
            continue
        group.sort(key=lambda e: e["frame"])
        frames = [e["frame"] for e in group]
        if any(b - a != 1 for a, b in zip(frames, frames[1:])):
            continue
        if group[0]["charge"] < max(e["charge"] for e in group):
            continue
        group[0]["recurring"] = False
        n_rescued += 1
    return n_rescued


def extract_events(chips, gains, bad_masks, chip_list, epoch):
    """Detect events on +residual and on -residual of one epoch.

    chip_list: chips to process (0-based).  Returns {+1: events,
    -1: events}.  Positive events are the CR candidates and carry stamps;
    negative ones are noise only and are used to measure the contamination
    of the cut libraries.  Both go through exactly the same detection and
    flags.
    """
    all_events = {1: [], -1: []}
    for c in chip_list:
        gain = gains[c]
        n_frames = chips[c].shape[0]
        reference = np.median(chips[c], axis=0)

        # How many frames each pixel passed DETECTION_SIGMA in, in either
        # direction (see FILTER_RECURRING).
        fire_count = np.zeros((DETECTOR_NPIX, DETECTOR_NPIX), dtype=np.int16)
        chip_events = {1: [], -1: []}

        for iframe in range(n_frames):
            residual = chips[c][iframe] - reference
            med = np.median(residual)
            sigma = 1.4826 * np.median(np.abs(residual - med))
            residual -= med
            z = residual / sigma
            fire_count += (np.abs(z) > DETECTION_SIGMA) & ~bad_masks[c]

            for sign in (1, -1):
                seed = (sign * z > DETECTION_SIGMA) & ~bad_masks[c]
                labeled, _ = ndimage.label(
                    (sign * z > GROW_SIGMA) & ~bad_masks[c],
                    structure=CONNECTIVITY)
                obj_slices = ndimage.find_objects(labeled)
                # only the grown regions that contain a seed are events
                ev_ids = np.unique(labeled[seed])
                for ev_id in ev_ids[ev_ids > 0]:
                    slc = obj_slices[ev_id - 1]
                    event_mask = labeled[slc] == ev_id
                    core_mask = event_mask & seed[slc]
                    ncore = int(core_mask.sum())
                    if ncore > MAX_PIXELS:
                        continue
                    npix = int(event_mask.sum())

                    ys, xs = np.where(event_mask)
                    cy = int(np.mean(ys)) + slc[0].start
                    cx = int(np.mean(xs)) + slc[1].start

                    h = STAMP_HALFSIZE
                    y0, y1 = cy - h, cy + h + 1
                    x0, x1 = cx - h, cx + h + 1
                    if (y0 < 0 or x0 < 0
                            or y1 > DETECTOR_NPIX or x1 > DETECTOR_NPIX):
                        continue

                    # Charge is summed over the FULL event, not over the
                    # stamp: a track longer than the 21x21 stamp would
                    # otherwise be truncated and disagree with NPIX.
                    charge_electrons = float(
                        sign * np.sum(residual[slc][event_mask]) * gain)

                    # Elongation of the core pixels (see ELONG_NPIX_MIN).
                    # With a few pixels the covariance is degenerate (2-3
                    # pixels in a line give a zero width), so smaller
                    # events get NaN.
                    cys, cxs = np.where(core_mask)
                    if ncore >= ELONG_NPIX_MIN:
                        cov = np.cov(cys.astype(float), cxs.astype(float))
                        eigvals = np.clip(np.linalg.eigvalsh(cov),
                                          1e-10, None)
                        elongation = float(
                            np.sqrt(eigvals.max() / eigvals.min()))
                    else:
                        elongation = np.nan

                    ev = {
                        "charge": charge_electrons,
                        "npix": npix,
                        "ncore": ncore,
                        "elongation": elongation,
                        "chip": c + 1,
                        "epoch": epoch,
                        "frame": iframe,
                        "cx": cx,
                        "cy": cy,
                        # absolute coordinates of the core pixels, dropped
                        # below once fire_count is complete
                        "pys": cys + slc[0].start,
                        "pxs": cxs + slc[1].start,
                    }
                    # Stamps only where a cut library can keep the event;
                    # the recurrence flag is not known yet at this point.
                    if sign > 0 and charge_electrons >= STAMP_MIN_CHARGE:
                        local_mask = labeled[y0:y1, x0:x1] == ev_id
                        stamp = np.zeros((stamp_size, stamp_size),
                                         dtype=np.float32)
                        stamp[local_mask] = (
                            residual[y0:y1, x0:x1][local_mask] * gain)
                        n_clipped = npix - int(local_mask.sum())
                        if n_clipped:
                            print(f"    WARNING: event at ({cx},{cy}) "
                                  f"exceeds the stamp, {n_clipped} px "
                                  f"outside - stamp is clipped (CHARGE "
                                  f"is still complete)")
                        ev["stamp"] = stamp
                    chip_events[sign].append(ev)

        # fire_count is only complete once every frame of this chip has been
        # processed, so the recurrence flag is set here rather than above.
        n_rescued = {}
        for sign, evs in chip_events.items():
            for ev in evs:
                ev["recurring"] = bool(
                    np.all(fire_count[ev["pys"], ev["pxs"]] > 1))
            n_rescued[sign] = (_rescue_persistence(evs)
                               if RESCUE_PERSISTENCE else 0)
            for ev in evs:
                del ev["pys"], ev["pxs"]
            all_events[sign].extend(evs)

        pos = chip_events[1]
        n_rec = sum(e["recurring"] for e in pos)
        print(f"    Chip {c+1}: {len(pos)} events "
              f"({n_frames} frames, gain={gain:.3f} e/ADU), "
              f"{n_rec} recurring "
              f"({100 * n_rec / max(len(pos), 1):.1f}%), "
              f"{n_rescued[1]} persistence-rescued; "
              f"{len(chip_events[-1])} negative (noise) events")

    return all_events


# Table columns: (event key, FITS column name, FITS format, numpy dtype)
COLUMNS = [
    ("charge", "CHARGE", "E", np.float32),
    ("npix", "NPIX", "I", np.int16),
    ("ncore", "NCORE", "I", np.int16),
    ("elongation", "ELONGATION", "E", np.float32),
    ("chip", "CHIP", "I", np.int16),
    ("epoch", "EPOCH", "I", np.int16),   # row of the EPOCHS table
    ("frame", "FRAME", "I", np.int16),   # frame within its epoch
    ("cx", "CX", "I", np.int16),
    ("cy", "CY", "I", np.int16),
    ("recurring", "RECUR", "L", bool),
]


def events_to_arrays(all_events, with_stamps=True):
    """Pack an event list into a dict of numpy arrays.

    Only events with charge >= STAMP_MIN_CHARGE have a stamp: "stamps"
    holds those,
    and "stamp_idx" gives each event's row in it (-1 for none).
    """
    a = {key: np.array([ev[key] for ev in all_events], dtype=dt)
         for key, _, _, dt in COLUMNS}
    if with_stamps:
        has = np.array(["stamp" in ev for ev in all_events], dtype=bool)
        a["stamp_idx"] = np.full(len(all_events), -1, dtype=np.int64)
        a["stamp_idx"][has] = np.arange(int(has.sum()))
        a["stamps"] = np.array(
            [ev["stamp"] for ev in all_events if "stamp" in ev],
            dtype=np.float32).reshape(-1, stamp_size, stamp_size)
    return a


def join_events(parts, chips, with_stamps=True):
    """Join the event arrays of several epochs, keeping the given chips."""
    out = {key: [] for key, *_ in COLUMNS}
    idx, stamps, offset = [], [], 0
    for a in parts:
        m = np.isin(a["chip"], chips)
        for key in out:
            out[key].append(a[key][m])
        if with_stamps:
            si = a["stamp_idx"][m]
            idx.append(np.where(si >= 0, si + offset, -1))
            stamps.append(a["stamps"])
            offset += len(a["stamps"])
    out = {key: np.concatenate(v) for key, v in out.items()}
    if with_stamps:
        out["stamp_idx"] = np.concatenate(idx)
        out["stamps"] = np.concatenate(stamps)
    return out


def passes_filters(a, cutoff):
    """Charge and recurrence filters of the cut libraries."""
    filt = a["charge"] >= cutoff
    if FILTER_RECURRING:
        filt &= ~a["recurring"]
    return filt


def base_header(n_events, info):
    """Primary header keywords shared by all libraries."""
    phdu = fits.PrimaryHDU()
    phdu.header["N_EVENTS"] = n_events
    phdu.header["STAMPSZ"] = stamp_size
    phdu.header["DET_SIG"] = (DETECTION_SIGMA, "Event start threshold")
    phdu.header["GROW_SIG"] = (GROW_SIGMA, "Event growth threshold")
    phdu.header["INSTRUME"] = "GSAOI"
    phdu.header["DETTYPE"] = "H2RG"
    phdu.header["PIXSIZE"] = (18.0, "micron")
    phdu.header["N_EPOCHS"] = (len(info["epochs"]), "Dark sequences used")
    phdu.header["N_DARKS"] = (info["n_darks"], "Darks used (first ones "
                              "dropped)")
    phdu.header["SKIPFRST"] = (SKIP_FIRST_FRAMES,
                               "Darks dropped at the start of each epoch")
    phdu.header["LNRS"] = (info["lnrs"], "Read modes (LNRS) used")
    phdu.header["COADDS"] = (info["coadds"], "Coadds of the darks used")
    phdu.header["CHIPS"] = (info["chips"], "Chips used")
    return phdu


def table_hdu(a, idx):
    """METADATA table of the events idx (an index or mask array)."""
    return fits.BinTableHDU.from_columns(
        [fits.Column(name=name, format=fmt, array=a[key][idx],
                     unit="electrons" if key == "charge" else None)
         for key, name, fmt, _ in COLUMNS], name="METADATA")


def epochs_hdu(epochs):
    """EPOCHS table: the dark sequences a library is built from."""
    return fits.BinTableHDU.from_columns([
        fits.Column(name="EPOCH", format="I",
                    array=[e["index"] for e in epochs]),
        fits.Column(name="DATE_OBS", format="10A",
                    array=[e["date"] for e in epochs]),
        fits.Column(name="OBSID", format="30A",
                    array=[e["obsid"] for e in epochs]),
        fits.Column(name="EXPTIME", format="E", unit="s",
                    array=[e["exptime"] for e in epochs]),
        fits.Column(name="LNRS", format="I",
                    array=[e["lnrs"] for e in epochs]),
        fits.Column(name="COADDS", format="I",
                    array=[e["coadds"] for e in epochs]),
        fits.Column(name="N_DARKS", format="I",
                    array=[len(e["files"]) for e in epochs]),
        fits.Column(name="FIRST", format="20A",
                    array=[e["files"][0].stem for e in epochs]),
    ], name="EPOCHS")


def find_cuts(a, neg, label):
    """Contamination vs cut; return the cut of each CUT_TARGETS entry.

    The cut for a target is the lowest charge from which the contamination
    stays <= target for every higher cut (scanned up to 10 ke-, 10 e-
    steps).  A target not reached gives no cut.
    """
    cuts = np.arange(STAMP_MIN_CHARGE, 10001, 10)
    n_pos = np.array([passes_filters(a, c).sum() for c in cuts])
    n_neg = np.array([passes_filters(neg, c).sum() for c in cuts])
    contam = n_neg / np.maximum(n_pos, 1)
    print(f"  Contamination vs cut ({label}):")
    for c in (300, 400, 500, 600, 700, 800, 1000, 1500, 2000, 3000):
        i = np.searchsorted(cuts, c)
        print(f"    >= {c:4d} e: {n_pos[i]:6d} events, {n_neg[i]:5d} "
              f"negative, {100 * contam[i]:5.1f}%")
    found = {}
    for key, target in CUT_TARGETS.items():
        ok = [i for i in range(len(cuts)) if (contam[i:] <= target).all()]
        if not ok:
            print(f"    {key}: <= {100 * target:g}% never reached, skipped")
            continue
        found[key] = float(cuts[ok[0]])
        floor = " (at STAMP_MIN_CHARGE: the true cut may be lower)" \
            if ok[0] == 0 else ""
        print(f"    {key}: cut {found[key]:.0f} e "
              f"({100 * contam[ok[0]]:.1f}%){floor}")
    return found


def save_filtered(filepath, a, neg, cutoff, label, info):
    """Save a charge- and recurrence-filtered library.

    neg holds the negative (noise) events; the ones passing the same
    filters give the contamination estimate CONTAM.  info describes the
    epochs, exposure and area the library is built from.
    """
    charge_filt = a["charge"] >= cutoff
    n_recur_rejected = (int((charge_filt & a["recurring"]).sum())
                        if FILTER_RECURRING else 0)
    filt = passes_filters(a, cutoff)
    n = int(filt.sum())
    if n == 0:
        print(f"    {label}: no events after filtering, skipping")
        return
    neg_filt = passes_filters(neg, cutoff)
    n_neg = int(neg_filt.sum())
    contam = n_neg / n
    contam_err = np.sqrt(max(n_neg, 1)) / n

    phdu = base_header(n, info)
    phdu.header["CHG_CUT"] = (cutoff, "Charge cutoff (e)")
    phdu.header["N_NEG"] = (n_neg, "Negative (noise) events passing cuts")
    phdu.header["CONTAM"] = (round(contam, 4),
                             "Noise fraction, N_NEG / N_EVENTS")
    phdu.header["CONTAMER"] = (round(contam_err, 4),
                               "Poisson error of CONTAM")
    phdu.header["RECURFLT"] = (FILTER_RECURRING,
                               "Recurring-pixel events removed")
    phdu.header["PERSRESC"] = (RESCUE_PERSISTENCE,
                               "Persistence-like first events kept")
    phdu.header["N_RECRJ"] = (n_recur_rejected,
                              "Events rejected as recurring")
    cr_rate = n / (info["exposure"] * info["area"])
    phdu.header["CR_RATE"] = (cr_rate, "events/cm2/s")
    phdu.header["TOT_EXP"] = (info["exposure"],
                              "s, sum of darks x EXPTIME x COADDS")
    phdu.header["DET_AREA"] = (DETECTOR_AREA_CM2, "cm2 per chip")
    phdu.header["EFF_AREA"] = (EFFECTIVE_AREA_CM2,
                               "cm2 per chip without the edge margin")
    phdu.header["SRCH_ARE"] = (round(info["area"], 4),
                               "cm2, chips used, unmasked (CR_RATE uses)")

    sort_idx = np.argsort(a["charge"][filt])[::-1]
    filt_indices = np.where(filt)[0][sort_idx]

    stamps_hdu = fits.ImageHDU(a["stamps"][a["stamp_idx"][filt_indices]],
                               name="STAMPS")
    stamps_hdu.header["BUNIT"] = "electrons"
    stamps_hdu.header["SORTED"] = ("CHARGE_DESC",
                                   "Sorted by charge, descending")

    hdul = fits.HDUList([phdu, stamps_hdu, table_hdu(a, filt_indices),
                         epochs_hdu(info["epochs"])])
    hdul.writeto(str(filepath), overwrite=True)
    print(f"    {label}: {n} events, "
          f"{n_recur_rejected} recurring-rejected, "
          f"CONTAM={100 * contam:.1f} +- {100 * contam_err:.1f}% "
          f"({n_neg} negative), "
          f"CR_RATE={cr_rate:.4f} events/cm2/s")
    for c in np.unique(a["chip"]):
        nc = int((filt & (a["chip"] == c)).sum())
        nn = int((neg_filt & (neg["chip"] == c)).sum())
        print(f"      chip {c}: {nc:5d} events, {nn:4d} negative, "
              f"CONTAM={100 * nn / max(nc, 1):.1f}%")


def build_library(name, lib_epochs, chips):
    """Join the epochs on the given chips, scan and save every cut.

    Returns what the diagnostic plots need, or None without epochs.
    """
    if not lib_epochs:
        print(f"\n  Library {name}: no epochs, skipped")
        return None
    a = join_events([e["pos"] for e in lib_epochs], chips)
    neg = join_events([e["neg"] for e in lib_epochs], chips,
                      with_stamps=False)
    info = {
        "epochs": lib_epochs,
        "n_darks": sum(len(e["files"]) for e in lib_epochs),
        # EXPTIME is per coadd (see the coadd correction below)
        "exposure": sum(len(e["files"]) * e["exptime"] * e["coadds"]
                        for e in lib_epochs),
        "area": sum(SEARCHED_AREA_CM2[c - 1] for c in chips),
        "lnrs": ",".join(str(x) for x in
                         sorted({e["lnrs"] for e in lib_epochs})),
        "coadds": ",".join(str(x) for x in
                           sorted({e["coadds"] for e in lib_epochs})),
        "chips": ",".join(str(c) for c in chips),
    }
    print(f"\n{'='*60}")
    print(f"  LIBRARY: {name}  ({len(lib_epochs)} epochs, "
          f"{info['n_darks']} darks, {info['exposure']:.0f} s, "
          f"LNRS {info['lnrs']}, chips {info['chips']})")
    print(f"{'='*60}")
    print(f"  {len(a['charge'])} events (and {len(neg['charge'])} negative)")
    cuts = find_cuts(a, neg, name)
    for key, cutoff in cuts.items():
        save_filtered(OUTPUT_DIR / f"cr_library_{name}_{key}.fits",
                      a, neg, cutoff, f"{name} {key} (>= {cutoff:.0f} e)",
                      info)
    del a["stamps"], a["stamp_idx"]
    return {"pos": a, "neg": neg, "n_darks": info["n_darks"],
            "n_epochs": len(lib_epochs), "cuts": cuts}


def epoch_ok(lib, epoch):
    """Whether a library uses this epoch (read mode, coadds, exposure)."""
    coadds = lib.get("coadds")
    return ((lib["lnrs"] is None or epoch["lnrs"] in lib["lnrs"])
            and (coadds is None or epoch["coadds"] in coadds)
            and epoch["exptime"] >= lib.get("min_exptime", 0))


# ── Load BPM (shared across all epochs) ───────────────────────────────
print(f"Loading BPM from {BPM_FILE.name}...")
bpm_hdul = fits.open(BPM_FILE)
bad_masks = {}
for c in range(N_CHIPS):
    bad_masks[c] = bpm_hdul[c + 1].data != 0
    n_bpm = int(bad_masks[c].sum())
    if c in EXTRA_BAD:
        y0, y1, x0, x1 = EXTRA_BAD[c]
        bad_masks[c][y0:y1, x0:x1] = True
    print(f"  Chip {c+1}: {n_bpm} BPM bad pixels, "
          f"{int(bad_masks[c].sum()) - n_bpm} more in EXTRA_BAD")
bpm_hdul.close()

# Unmasked pixels inside the searched region (edge margin removed), per
# chip.  A CR that only partly hits masked pixels is still found, so for
# multi-pixel events this slightly over-corrects.
h = STAMP_HALFSIZE
SEARCHED_AREA_CM2 = [
    int((~bad_masks[c][h:DETECTOR_NPIX - h, h:DETECTOR_NPIX - h]).sum())
    * PIXEL_SIZE_CM ** 2 for c in range(N_CHIPS)]
print(f"  Searched area per chip (cm2): "
      f"{', '.join(f'{x:.2f}' for x in SEARCHED_AREA_CM2)} "
      f"(edge-trimmed: {EFFECTIVE_AREA_CM2:.2f})")

# ── Find the epochs ───────────────────────────────────────────────────
print(f"\nGrouping the darks in {DARK_DIR}...")
epochs = find_epochs(DARK_DIR)
print(f"  {len(epochs)} epochs, "
      f"{sum(len(e['files']) for e in epochs)} darks:")
for e in epochs:
    print(f"    {e['index']:2d}  {e['date']}  {e['obsid']:22s} "
          f"{e['exptime']:5.0f} s  LNRS {e['lnrs']:2d}  "
          f"COADDS {e['coadds']}  {len(e['files']):3d} darks")

# Only process the epochs and chips some library uses
used_chips = sorted({c for lib in LIBRARIES.values() for c in lib["chips"]})
if SAVE_EPOCH_LIBRARIES:
    to_process = epochs
else:
    to_process = [e for e in epochs
                  if any(epoch_ok(lib, e) for lib in LIBRARIES.values())]

# ── Process each epoch ────────────────────────────────────────────────
for e in to_process:
    print(f"\n{'='*60}")
    print(f"  EPOCH {e['index']}: {e['date']} {e['obsid']}, "
          f"{e['exptime']:.0f} s, LNRS {e['lnrs']}, COADDS {e['coadds']}, "
          f"{len(e['files'])} darks")
    print(f"{'='*60}")
    chips, gains, exptime = load_darks(e["files"])
    # Coadded GSAOI frames are the AVERAGE of the coadds (their noise is
    # 1/sqrt(COADDS) of a single read), so a cosmic ray in one coadd shows
    # 1/COADDS of its charge.  Scaling the gain by COADDS restores it; the
    # detection itself works in sigma units and is unaffected.
    gains = {c: g * e["coadds"] for c, g in gains.items()}
    events = extract_events(chips, gains, bad_masks,
                            [c - 1 for c in used_chips], e["index"])
    del chips
    e["pos"] = events_to_arrays(events[1])
    e["neg"] = events_to_arrays(events[-1], with_stamps=False)
    del events
    if SAVE_EPOCH_LIBRARIES:
        build_library(f"epoch{e['index']:02d}_{e['date']}", [e],
                      used_chips)

# ── Build the libraries ───────────────────────────────────────────────
library_results = {}
for name, lib in LIBRARIES.items():
    lib_epochs = [e for e in to_process if epoch_ok(lib, e)]
    res = build_library(name, lib_epochs, list(lib["chips"]))
    if res is not None:
        library_results[name] = res

# ── Diagnostic plots (one figure per library) ─────────────────────────

# Electron-to-keV conversion for HgCdTe (GSAOI H2RG, 2.5 μm cutoff)
# Pair creation energy from Alig & Bloom (1975): ε = 2.73·Eg + 0.55 eV
# Bandgap: Eg = hc/λ_cutoff = 0.496 eV for λ=2.5 μm
# Fox et al. (2009, arXiv:0906.0579) measured ε=2.61±0.16 eV for 1.7 μm
# HgCdTe, consistent with Alig formula (predicts 2.55 eV at that cutoff).
# For 2.5 μm cutoff: ε ≈ 1.91 eV/e-h pair
PAIR_ENERGY_EV = 2.73 * 0.496 + 0.55  # eV per electron-hole pair (KeV axis)
EV_PER_KEV = 1000.0

CUT_COLORS = dict(zip(CUT_TARGETS,
                      ("navy", "firebrick", "darkgreen", "purple")))
LOG_BINS = np.logspace(np.log10(30), np.log10(7e4), 60)
LOG_CENTERS = np.sqrt(LOG_BINS[:-1] * LOG_BINS[1:])


def draw_cuts(ax, cuts):
    for key, cut in cuts.items():
        ax.axvline(cut, color=CUT_COLORS[key], ls='--', lw=1.5,
                   label=f'{key}: {cut:.0f} e')


for lib_name, res in library_results.items():
    a, neg = res["pos"], res["neg"]
    charges = a["charge"]
    npixels = a["npix"]
    recurring = a["recurring"]
    n_darks = res["n_darks"]
    cuts = res["cuts"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Library: {lib_name} ({res['n_epochs']} epochs, "
                 f"{n_darks} darks)", fontsize=14)

    # Top left: positive (CR candidates) and negative (noise) events, after
    # the recurrence filter.  Where the two curves overlap, the
    # events are noise.
    ax = axes[0, 0]
    hp, _ = np.histogram(charges[passes_filters(a, 0.0)], bins=LOG_BINS)
    hn, _ = np.histogram(neg["charge"][passes_filters(neg, 0.0)],
                         bins=LOG_BINS)
    ax.step(LOG_CENTERS, hp, where='mid', color='tab:blue',
            label=f'positive (CR candidates): {hp.sum():,}')
    ax.step(LOG_CENTERS, hn, where='mid', color='tab:red',
            label=f'negative (noise): {hn.sum():,}')
    draw_cuts(ax, cuts)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.5, None)
    ax.set_xlabel('Charge (electrons)')
    ax.set_ylabel('N events per bin')
    ax.set_title('Positive vs negative events (after filters)')
    ax.legend(fontsize=7)

    # Top middle: the excess, i.e. the real cosmic rays per bin
    ax = axes[0, 1]
    diff, err = hp - hn, np.sqrt(hp + hn)
    ax.errorbar(LOG_CENTERS, diff, yerr=err, fmt='o', ms=3, color='k',
                capsize=2)
    ax.axhline(0, color='tab:red', lw=1)
    draw_cuts(ax, cuts)
    top = max(1.0, np.max((diff + err)[LOG_CENTERS > 300]))
    ax.set_ylim(-0.5 * top, 1.2 * top)
    ax.set_xscale('log')
    ax.set_xlabel('Charge (electrons)')
    ax.set_ylabel('positive - negative per bin')
    ax.set_title('Excess over noise (error = sqrt(pos + neg))')
    ax.legend(fontsize=7)

    # Bottom left: charge distribution split by recurrence.  Recurring events
    # are what the detector does; single-firing ones are candidate impacts.
    ax = axes[1, 0]
    pos = charges[charges > 0]
    if len(pos):
        log_bins = np.logspace(np.log10(max(pos.min(), 1)),
                               np.log10(pos.max()), 120)
        ch_rec = charges[recurring]
        ch_one = charges[~recurring]
        ax.hist(ch_rec, bins=log_bins, color='darkorange',
                edgecolor='none', alpha=0.55,
                label=f'recurring: {len(ch_rec):,}')
        ax.hist(ch_one, bins=log_bins, color='teal',
                edgecolor='none', alpha=0.55,
                label=f'fires once: {len(ch_one):,}')
        draw_cuts(ax, cuts)
        ax.legend(fontsize=7)
    ax.set_title('Charge distribution: recurring vs single-firing')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Charge (electrons)')
    ax.set_ylabel('N events')
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    e_lims = ax.get_xlim()
    ax2.set_xlim(e_lims[0] * PAIR_ENERGY_EV / EV_PER_KEV,
                 e_lims[1] * PAIR_ENERGY_EV / EV_PER_KEV)
    ax2.set_xlabel(
        'Energy (keV)  [ε = {:.2f} eV/pair, Alig & Bloom 1975]'
        .format(PAIR_ENERGY_EV), fontsize=8)

    # Bottom right: pixels per event, for each library, before and after the
    # recurrence filter.  One bin per pixel, log counts: the npix=1 bin holds
    # most of the catalogue and would flatten everything else on a linear axis.
    ax = axes[1, 1]
    pix_bins = np.arange(0.5, 30.5, 1)
    sets = [('all events', charges > 0, 'steelblue')]
    for key, cut in cuts.items():
        sets.append((f'{key} (>={cut:.0f} e)', charges >= cut,
                     CUT_COLORS[key]))
    for lbl, m, col in sets:
        ax.hist(npixels[m], bins=pix_bins, histtype='step', lw=1.4,
                color=col, label=f'{lbl}: {int(m.sum()):,}')
        ax.hist(npixels[m & ~recurring], bins=pix_bins, color=col,
                edgecolor='none', alpha=0.35,
                label=f'   non-recurring: {int((m & ~recurring).sum()):,}')
    ax.set_yscale('log')
    ax.legend(fontsize=6)
    ax.set_title('Pixels per event (filled = after recurrence filter)')
    ax.set_xlabel('N pixels')
    ax.set_ylabel('N events')

    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"charge_distribution_{lib_name}.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"\nPlot saved to {plot_path.name}")
    plt.close(fig)

print("\nDone! All libraries saved.")
