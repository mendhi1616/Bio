import matplotlib
matplotlib.use("Agg")

import os, time, io, base64
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy import ndimage as ndi, stats
from scipy.spatial import cKDTree
from skimage import filters, exposure, measure, morphology
from skimage.util import img_as_float


try:
    import torch
    TORCH_OK = True
    CUDA_OK = torch.cuda.is_available()
except Exception:
    TORCH_OK, CUDA_OK = False, False
try:
    from cellpose import models
    CELLPOSE_OK = True
except Exception:
    CELLPOSE_OK = False
ADV_PARAMS = {
    "sigma": 1.2,
    "min_size": 20,
    "max_size": 6000,
    "clahe_clip": 0.03,
    "edge_margin": 30,
    "deep_enhance": False,
}

import tifffile
import re

def _read_metadata(path, logger=None):
    """
    Lecture robuste des métadonnées TIF :
    - pixel_size (µm/pixel)
    - dt (sec par frame)
    """

    pixel_size = None
    dt = None

    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            desc_tag = tags.get("ImageDescription")

            if logger:
                logger("------- METADATA RAW -------")

            # ---- Lire texte brut ----
            if desc_tag is not None:
                desc_value = desc_tag.value
                if isinstance(desc_value, bytes):
                    desc_value = desc_value.decode(errors="ignore")
                if logger:
                    logger(desc_value)
            else:
                desc_value = ""
                if logger:
                    logger("Aucune ImageDescription trouvée.")

            if logger:
                logger("----------------------------")

            # Si vide → impossible de parser
            if not desc_value:
                return None, None

            # -------- Pixel Size --------
            patterns_px = [
                r"PixelSizeUm\s*=\s*([\d\.]+)",
                r"pixel_size\s*=\s*([\d\.]+)",
                r"PhysicalSizeX\s*=\s*([\d\.]+)",
                r"XPixelSize\s*=\s*([\d\.]+)",
                r"spacing\s*=\s*([\d\.]+)",
            ]

            for p in patterns_px:
                m = re.search(p, desc_value)
                if m:
                    pixel_size = float(m.group(1))
                    break

            # -------- Time per frame --------
            patterns_dt = [
                r"FrameTime\s*=\s*([\d\.]+)",
                r"TimeIncrement\s*=\s*([\d\.]+)",
                r"finterval\s*=\s*([\d\.]+)",   # ImageJ !!!
                r"dt\s*=\s*([\d\.]+)",
                r"Interval_ms\s*=\s*([\d\.]+)",
                r"ExposureTime\s*=\s*([\d\.]+)",
            ]

            for p in patterns_dt:
                m = re.search(p, desc_value)
                if m:
                    dt = float(m.group(1))
                    break

    except Exception as e:
        if logger:
            logger(f"[WARN] Impossible de lire les métadonnées ({e})")

    # Convert ms → seconds si trop grand
    if dt is not None and dt > 5:
        dt = dt / 1000.0

    if logger:
        logger(f"[META] pixel_size = {pixel_size} µm/pixel — dt = {dt} s")

    return pixel_size, dt

from cellpose.models import CellposeModel
GLOBAL_MODEL = None

def process_file(path, seg_method="auto", logger=None, debug=False, fast_mode=False):
    t0 = time.time()

    if logger:
        logger(f"➡️ Fichier: {os.path.basename(path)} — méthode={seg_method} — fast={fast_mode}")

    # --- Charger stack ---
    stack = _load_stack(path, logger=logger)

    # --- Lire métadonnées (pixel_size, dt) ---
    pixel_size, dt = _read_metadata(path, logger=logger)
    if pixel_size is None:
        pixel_size = 0.1
    if dt is None:
        dt = 60.0

    # --- FAST MODE : 1 frame sur 2 ---
    if fast_mode:
        if logger: logger("⚡ Mode Rapide activé : traitement de 1 frame sur 2.")
        stack = stack[::2]
        dt *= 2.0

    # --- Choix méthode ---
    if seg_method == "auto":
        method = "cellpose" if CELLPOSE_OK else "gam"
    else:
        method = seg_method

    # --- GPU ---
    use_gpu = torch.cuda.is_available()

    # --- Init modèle cellpose (une seule fois) ---
    global GLOBAL_MODEL
    if method == "cellpose" and GLOBAL_MODEL is None:
        if logger: logger("[INFO] Initialisation du modèle Cellpose…")
        GLOBAL_MODEL = CellposeModel(gpu=use_gpu)

    # --- Segmentation ---
    labels_list = _segment_stack(
        stack,
        method=method,
        logger=logger,
        sigma=ADV_PARAMS["sigma"],
        min_size=ADV_PARAMS["min_size"],
        max_size=ADV_PARAMS["max_size"],
        clahe_clip=ADV_PARAMS["clahe_clip"],
        edge_margin=ADV_PARAMS["edge_margin"],
        deep_enhance=ADV_PARAMS["deep_enhance"],
    )

    # --- Tracking ---
    tracks = _track_labels(labels_list, max_dist=20.0, logger=logger)

    if tracks is None or len(tracks) == 0:
        if logger:
            logger("[WARN] Tracking vide.")
        # >>> NO EARLY RETURN <<<
        metrics = pd.DataFrame({"file": [os.path.basename(path)]})
        return metrics, pd.DataFrame()

    # --- Motion features ---
    tracks = compute_motion_features(tracks, pixel_size, dt)

    # --- Metrics ---
    metrics = _metrics_from_tracks(tracks)
    metrics["file"] = os.path.basename(path)

    # --- Debug overlay ---
    if debug:
        out_dir = os.path.join(os.path.dirname(path), "outputs")
        os.makedirs(out_dir, exist_ok=True)

        overlay_path = os.path.join(out_dir, "overlay_" + os.path.basename(path) + ".png")
        mid = len(stack) // 2

        fig, ax = plt.subplots()
        ax.imshow(stack[mid], cmap="gray")
        ax.contour(labels_list[mid], colors="lime", linewidths=0.9)
        ax.axis("off")
        fig.savefig(overlay_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if logger:
            logger(f"🖼 Overlay sauvegardé : {overlay_path}")

    # --- FIN ---
    return metrics, tracks, stack, labels_list


def lap_match(prev_df, next_df, max_dist=25.0):
    if prev_df is None or len(prev_df) == 0:
        return [None] * len(prev_df)

    if next_df is None or len(next_df) == 0:
        return [None] * len(prev_df)

    # Matrice des distances
    cost = cdist(prev_df[["x", "y"]], next_df[["x", "y"]])

    # Interdire les distances trop grandes
    cost[cost > max_dist] = 1e9

    # Hungarian / LAP solver
    rows, cols = linear_sum_assignment(cost)

    match = [None] * len(prev_df)

    for r, c in zip(rows, cols):
        if cost[r, c] < 1e9:   # assignment valide
            match[r] = next_df.index[c]

    return match


def set_advanced_params(**kwargs):
    for k, v in kwargs.items():
        if k not in ADV_PARAMS or v is None:
            continue
        if k in ("min_size", "max_size", "edge_margin"):
            ADV_PARAMS[k] = int(v)
        elif k == "deep_enhance":
            ADV_PARAMS[k] = bool(v)
        else:
            ADV_PARAMS[k] = float(v)

from skimage import io as skio
def _load_stack(path, logger=None):
    ext = os.path.splitext(path)[1].lower()
    if logger:
        logger(f"Chargement: {os.path.basename(path)} ({ext})")
    if ext in (".tif", ".tiff"):
        arr = np.asarray(tiff.imread(path))
    elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
        img = skio.imread(path)
        if img.ndim == 3 and img.shape[2] in (3, 4):
            img = np.mean(img[..., :3], axis=-1)
        arr = img[np.newaxis, ...]  
    else:
        raise ValueError(f"Format non supporté: {ext}")

    if arr.ndim == 2:
        stack = arr[None, ...]
    elif arr.ndim == 3:
        stack = arr if arr.shape[0] > 10 else np.max(arr, axis=0)[None, ...]
    elif arr.ndim == 4:
        T_first = arr.shape[0] >= arr.shape[1]
        stack = np.max(arr, axis=1) if T_first else np.max(arr, axis=0)
    else:
        raise ValueError(f"Dimensions non gérées: {arr.shape}")

    frames = []
    for frame in stack:
        f = img_as_float(frame)
        f = exposure.rescale_intensity(f, in_range="image", out_range=(0, 1))
        frames.append(f)
    out = np.stack(frames, axis=0)
    if logger:
        logger(f"Frames normalisées: T={out.shape[0]}")
    return out

def _edge_mask(shape, margin):
    h, w = shape
    m = int(max(0, margin))
    mask = np.ones((h, w), dtype=bool)
    if m > 0:
        mask[:m, :] = False; mask[-m:, :] = False
        mask[:, :m] = False; mask[:, -m:] = False
    return mask

def _filter_by_shape(labels, *, min_size, max_size, circ=(0.35, 1.35), compact_min=0.55, elong_max=2.8):
    props = measure.regionprops(labels)
    keep = np.zeros_like(labels, dtype=bool)
    for p in props:
        a = p.area
        if a < min_size or a > max_size:
            continue
        per = p.perimeter if p.perimeter > 0 else 1.0
        circ_val = 4 * np.pi * a / (per ** 2)
        compact = (p.area / p.convex_area) if getattr(p, "convex_area", 0) > 0 else 0
        elong = (p.major_axis_length / p.minor_axis_length) if getattr(p, "minor_axis_length", 0) > 0 else 1.0
        if (circ[0] < circ_val < circ[1]) and (compact > compact_min) and (elong < elong_max):
            keep[labels == p.label] = True
    return measure.label(keep)

_SOBEL_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)/4.0
_SOBEL_Y = _SOBEL_X.T

def _enhance_edges_deep(img_f32):
    if TORCH_OK:
        dev = torch.device("cuda" if CUDA_OK else "cpu")
        with torch.no_grad():
            t = torch.from_numpy(img_f32).to(dev).unsqueeze(0).unsqueeze(0).float()
            k1 = torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=torch.float32, device=dev).view(1,1,3,3)
            x = torch.nn.functional.conv2d(t, k1, padding=1)
            x = torch.nn.functional.relu(x)
            kx = torch.from_numpy(_SOBEL_X).to(dev).view(1,1,3,3)
            ky = torch.from_numpy(_SOBEL_Y).to(dev).view(1,1,3,3)
            gx = torch.nn.functional.conv2d(x, kx, padding=1)
            gy = torch.nn.functional.conv2d(x, ky, padding=1)
            g = torch.sqrt(torch.clamp_min(gx*gx + gy*gy, 0.0))
            prob = torch.sigmoid(g).squeeze().cpu().numpy()
    else:
        gx = ndi.convolve(img_f32, _SOBEL_X, mode="nearest")
        gy = ndi.convolve(img_f32, _SOBEL_Y, mode="nearest")
        g = np.sqrt(np.clip(gx*gx + gy*gy, 0, None))
        prob = 1/(1+np.exp(-g))
    return exposure.rescale_intensity(prob, in_range="image", out_range=(0,1))

def _segment_frame_cellpose(img, logger=None):
    if logger is None:
        logger = print

    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"_segment_frame_cellpose attend une image 2D, reçu {img.shape}")

    # Normalisation
    i_min, i_max = float(img.min()), float(img.max())
    if i_max > i_min:
        img = (img - i_min) / (i_max - i_min)
    img = img.astype(np.float32, copy=False)

    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        raise RuntimeError("GLOBAL_MODEL non initialisé.")

    try:
        # ---- Appel Cellpose = compatible v3/v4/v5 ----
        pred = GLOBAL_MODEL.eval(img, channels=[0, 0])

        # ---- Format compatible toutes versions ----
        if isinstance(pred, dict):
            masks = pred.get("masks", None)
        elif isinstance(pred, (list, tuple)):
            masks = pred[0]  # v3/v4
        else:
            masks = pred  # fallback

        if masks is None:
            if logger: logger("[WARN] Cellpose n’a retourné aucun masque.")
            return np.zeros_like(img, dtype=np.int32)

        return masks.astype(np.int32)

    except Exception as e:
        logger(f"[ERREUR] Cellpose a échoué ({e}) — fallback GAM++.")
        return np.zeros_like(img, dtype=np.int32)



def _segment_frame_gam(img, sigma=1.6, min_size=1200, max_size=120000,
                       clahe_clip=0.025, edge_margin=8, deep_enhance=False):
    from skimage.filters import rank
    from skimage.morphology import disk
    bg = filters.gaussian(img, sigma=70, preserve_range=True)
    norm = np.clip(img / (bg + 1e-6), 0, 2)
    norm = exposure.rescale_intensity(norm, in_range="image", out_range=(0, 1))
    if deep_enhance:
        norm = np.clip(norm + 0.25 * _enhance_edges_deep(norm), 0, 1)
    norm = exposure.equalize_adapthist(norm, clip_limit=float(clahe_clip))
    sm = filters.gaussian(norm, sigma=sigma)
    grad = filters.scharr(sm)
    grad = exposure.rescale_intensity(grad, in_range="image", out_range=(0, 1))
    local_mean = rank.mean((sm * 255).astype(np.uint8), disk(15)) / 255.0
    local_thr = sm > (local_mean - 0.05)
    edge_mask = (grad > 0.12) | local_thr
    edge_mask = morphology.binary_closing(edge_mask, disk(3))
    edge_mask = ndi.binary_fill_holes(edge_mask)
    edge_mask = morphology.binary_opening(edge_mask, disk(2))
    edge_mask = morphology.remove_small_objects(edge_mask, min_size=int(min_size / 3))
    edge_mask &= _edge_mask(edge_mask.shape, edge_margin)
    labeled = measure.label(edge_mask)
    out = np.zeros_like(edge_mask, dtype=bool)
    for p in measure.regionprops(labeled, intensity_image=sm):
        if min_size < p.area < max_size and getattr(p, "solidity", 0.0) > 0.65:
            out[labeled == p.label] = True
    return measure.label(out)

def _segment_stack(stack, method="gam", logger=None, **kwargs):
    labels_list = []

    if logger:
        logger(f"  ↳ Segmentation: méthode = {method}")

    # ----- Mode auto -----
    if method == "auto":
        method = "cellpose" if CELLPOSE_OK else "gam"

    # ----- Mode Cellpose v4 -----
    if method == "cellpose" and CELLPOSE_OK:
        for t, frame in enumerate(stack):

            # ✔️ Log du modèle ici (plus propre & sans paramètre inutile)
            if logger and t == 0:
                logger(f"[INFO] Segmentation Cellpose (modèle 'cyto2')")

            # ✔️ Correction : suppression du paramètre model_type
            labels = _segment_frame_cellpose(
                frame,
                logger=logger
            )

            labels_list.append(labels)

            # Log toutes les 10 frames et dernière
            if logger and (t % 10 == 0 or t == len(stack) - 1):
                logger(f"    • Frames Cellpose: {t+1}/{len(stack)}")

        return labels_list

    # ----- Mode GAM++ -----
    deep = kwargs.get("deep_enhance", ADV_PARAMS["deep_enhance"])

    for t, frame in enumerate(stack):
        labels = _segment_frame_gam(
            frame,
            sigma=kwargs.get("sigma", ADV_PARAMS["sigma"]),
            min_size=kwargs.get("min_size", ADV_PARAMS["min_size"]),
            max_size=kwargs.get("max_size", ADV_PARAMS["max_size"]),
            clahe_clip=kwargs.get("clahe_clip", ADV_PARAMS["clahe_clip"]),
            edge_margin=kwargs.get("edge_margin", ADV_PARAMS["edge_margin"]),
            deep_enhance=deep,
        )

        labels_list.append(labels)

        if logger and (t % 10 == 0 or t == len(stack) - 1):
            logger(f"    • Frames segmentées: {t+1}/{len(stack)}")

    return labels_list


def _props_from_labels(labels):
    props = measure.regionprops_table(
        labels,
        properties=(
            "label",
            "area",
            "perimeter",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "solidity",
            "feret_diameter_max",
            "centroid",
        ),
    )

    df = pd.DataFrame(props)

    # Renommer coordonnées
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)

    return df

def link_frames(df_t, df_tp1, max_dist=15.0):
    """Match objects between frame t and t+1 using KD-tree."""
    if len(df_t) == 0 or len(df_tp1) == 0:
        return pd.Series([None] * len(df_t), index=df_t.index, dtype=object)

    tree = cKDTree(df_tp1[["x", "y"]].values)
    dists, idxs = tree.query(
        df_t[["x", "y"]].values,
        k=1,
        distance_upper_bound=max_dist
    )

    links = pd.Series([None] * len(df_t), index=df_t.index, dtype=object)

    for i, (d, idx) in enumerate(zip(dists, idxs)):
        if np.isfinite(d) and idx < len(df_tp1):
            links.iloc[i] = df_tp1.index[idx]

    return links


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def _track_labels(labels_list, max_dist=15.0, logger=None, gap_frames=2):
    if logger:
        logger("  ↳ Tracking (LAP + gap closing)…")

    tracks = []
    prev = None
    track_end = {}      # track_id → (x, y, t)
    pending_start = {}  # new objects to try reconnect

    next_track_id = 1

    # -----------------------
    # PASS 1 : LAP frame-to-frame
    # -----------------------
    for t, lab in enumerate(labels_list):

        df = _props_from_labels(lab).copy()
        df["t"] = t
        df.index = pd.Index([f"t{t}_l{int(x)}" for x in df["label"]], name="obj")

        if prev is None:
            df["track_id"] = range(next_track_id, next_track_id + len(df))
            for _, row in df.iterrows():
                track_end[row["track_id"]] = (row["x"], row["y"], t)
            next_track_id += len(df)
            prev = df
            continue

        # Hungarian cost matrix
        cost = cdist(prev[["x", "y"]], df[["x", "y"]])
        cost[cost > max_dist] = 1e9

        rows, cols = linear_sum_assignment(cost)

        track_id = pd.Series(np.nan, index=df.index)
        used = set()

        for r, c in zip(rows, cols):
            if cost[r, c] < 1e9:
                p_obj = prev.index[r]
                c_obj = df.index[c]

                tid = prev.loc[p_obj, "track_id"]
                track_id.loc[c_obj] = tid
                used.add(c_obj)
                track_end[tid] = (df.loc[c_obj, "x"], df.loc[c_obj, "y"], t)

        # unassigned new objects → pending_start
        for obj in df.index:
            if obj not in used:
                new_tid = next_track_id
                next_track_id += 1

                track_id.loc[obj] = new_tid
                pending_start[obj] = (df.loc[obj, "x"], df.loc[obj, "y"], t, new_tid)

        df["track_id"] = track_id.values

        tracks.append(df[[
            "t",
            "label",
            "x",
            "y",
            "area",
            "perimeter",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "solidity",
            "feret_diameter_max",
            "track_id",
        ]])

        prev = df

        if logger and (t % 10 == 0 or t == len(labels_list)-1):
            logger(f"    • Frames trackées: {t+1}/{len(labels_list)}")


    # -----------------------
    # PASS 2 : GAP CLOSING
    # -----------------------

    if logger:
        logger("  ↳ Gap closing…")

    for obj, (x2, y2, t2, new_tid) in list(pending_start.items()):
        # Try to reconnect to an older track
        best_tid = None
        best_dist = 9999

        for tid, (x1, y1, t1) in track_end.items():

            if 1 <= (t2 - t1) <= gap_frames:
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist < max_dist and dist < best_dist:
                    best_tid = tid
                    best_dist = dist

        # reconnect
        if best_tid is not None:
            for df in tracks:
                df.loc[df["track_id"] == new_tid, "track_id"] = best_tid
            if logger:
                logger(f"    ↳ Gap closing : track {new_tid} → {best_tid}")

    # -----------------------
    # Résultat final
    # -----------------------
    out = pd.concat(tracks).reset_index()
    return out

def compute_motion_features(tracks, pixel_size, dt):
    """
    Ajout des métriques de mouvement (µm/s) et morphologiques (µm, µm²)
    pour chaque cellule (track_id, t).
    """

    # --- ORDONNER PAR CELLULE ET TEMPS ---
    tracks = tracks.sort_values(["track_id", "t"])

    # --- DIFFÉRENCES SPATIALES EN PIXELS ---
    tracks["dx"] = tracks.groupby("track_id")["x"].diff()
    tracks["dy"] = tracks.groupby("track_id")["y"].diff()

    # Distance par frame (en pixels)
    tracks["distance_px"] = np.sqrt(tracks["dx"]**2 + tracks["dy"]**2)
    tracks["distance_px"] = tracks["distance_px"].fillna(0)

    # --- UNITÉS PHYSIQUES ---
    # sécurité si dt ou pixel_size manquants
    if pixel_size is None or pixel_size <= 0:
        pixel_size = 1.0
    if dt is None or dt <= 0:
        dt = 1.0

    # Conversion en µm
    tracks["distance_um"] = tracks["distance_px"] * pixel_size

    # Vitesse instantanée en µm/s
    tracks["speed_um_s"] = tracks["distance_um"] / dt

    # Composantes de vitesse (µm/s)
    tracks["vx_um_s"] = tracks["dx"] * pixel_size / dt
    tracks["vy_um_s"] = tracks["dy"] * pixel_size / dt

    # Angle du mouvement (0° = vers la droite)
    tracks["angle_deg"] = np.degrees(np.arctan2(tracks["vy_um_s"], tracks["vx_um_s"]))

    # Distance cumulée (µm)
    tracks["cum_distance_um"] = tracks.groupby("track_id")["distance_um"].cumsum()

    # On garde aussi des colonnes compatibles avec ton UI actuelle
    tracks["speed"] = tracks["speed_um_s"]
    tracks["cum_distance"] = tracks["cum_distance_um"]

    # --- FEATURES MORPHOLOGIQUES EN µm / µm² ---

    # Aire (µm²)
    tracks["area_um2"] = tracks["area"] * (pixel_size ** 2)

    # Périmètre (µm)
    tracks["perimeter_um"] = tracks["perimeter"] * pixel_size

    # Circularité
    # 4π * area / perimeter², en prenant garde aux zéros
    tracks["circularity"] = np.nan
    valid = tracks["perimeter_um"] > 0
    tracks.loc[valid, "circularity"] = (
        4 * np.pi * tracks.loc[valid, "area_um2"] /
        (tracks.loc[valid, "perimeter_um"] ** 2)
    )

    # Aspect ratio = major / minor
    tracks["aspect_ratio"] = np.nan
    valid_minor = tracks["minor_axis_length"] > 0
    tracks.loc[valid_minor, "aspect_ratio"] = (
        tracks.loc[valid_minor, "major_axis_length"] /
        tracks.loc[valid_minor, "minor_axis_length"]
    )

    # Eccentricité (déjà fournie par regionprops)
    # On la laisse en l'état : 0 = rond, 1 = très allongé

    # Solidity (déjà fournie, entre 0 et 1)
    # Rien à changer.

    # Feret max (µm)
    tracks["feret_max_um"] = tracks["feret_diameter_max"] * pixel_size

    # --- Straightness (net / distance cumulée) ---
    # déplacement net (du premier au dernier point)
    def _net_disp_um(df):
        if len(df) < 2:
            return 0.0
        dx = df["x"].iloc[-1] - df["x"].iloc[0]
        dy = df["y"].iloc[-1] - df["y"].iloc[0]
        return np.sqrt(dx**2 + dy**2) * pixel_size

    # Correction pour éviter FutureWarning pandas
    net_disp = tracks.groupby("track_id")[["x", "y"]].apply(_net_disp_um)
    tracks["net_displacement_um"] = tracks["track_id"].map(net_disp)

    tracks["straightness"] = 0.0
    valid_cd = tracks["cum_distance_um"] > 0
    tracks.loc[valid_cd, "straightness"] = (
        tracks.loc[valid_cd, "net_displacement_um"] /
        tracks.loc[valid_cd, "cum_distance_um"]
    )

    return tracks



def _metrics_from_tracks(tracks):
    n_cells = tracks.groupby("t")["track_id"].nunique().rename("n_cells")
    t0 = tracks["t"].min()
    ids_t0 = set(tracks.loc[tracks["t"] == t0, "track_id"])
    survival = tracks.groupby("t")["track_id"].apply(
        lambda s: len(ids_t0.intersection(set(s))) / max(1, len(ids_t0))
    ).rename("survival_frac")
    return pd.concat([n_cells, survival], axis=1).reset_index()

def _auc(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2: return np.nan
    order = np.argsort(x)
    return np.trapz(y[order], x[order])

def compute_group_stats(results):
    auc_rows = []
    for (cond, f), df in results.groupby(["condition", "file"]):
        auc_p = _auc(df["t"], df["n_cells"]); auc_s = _auc(df["t"], df["survival_frac"])
        auc_rows.append({"condition": cond, "file": f, "AUC_prolif": auc_p, "AUC_survival": auc_s})
    auc_df = pd.DataFrame(auc_rows)

    last_t = results["t"].max() if not results.empty else np.nan
    ttest = {}
    if not results.empty and not np.isnan(last_t):
        conds = list(results["condition"].unique())
        if len(conds) == 2:
            c1, c2 = conds
            pivot_last = results[results["t"] == last_t]
            n1 = pivot_last[pivot_last["condition"] == c1]["n_cells"]
            n2 = pivot_last[pivot_last["condition"] == c2]["n_cells"]
            t_ncells = stats.ttest_ind(n1, n2, equal_var=False, nan_policy="omit")
            a1 = auc_df[auc_df["condition"] == c1]["AUC_prolif"]; a2 = auc_df[auc_df["condition"] == c2]["AUC_prolif"]
            t_auc_p = stats.ttest_ind(a1, a2, equal_var=False, nan_policy="omit")
            s1 = auc_df[auc_df["condition"] == c1]["AUC_survival"]; s2 = auc_df[auc_df["condition"] == c2]["AUC_survival"]
            t_auc_s = stats.ttest_ind(s1, s2, equal_var=False, nan_policy="omit")
            ttest = {"conditions": (c1, c2),
                     "last_timepoint_ncells_p": float(t_ncells.pvalue),
                     "AUC_prolif_p": float(t_auc_p.pvalue),
                     "AUC_survival_p": float(t_auc_s.pvalue)}
    return auc_df, ttest

def plot_curves(results, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True); saved = []
    if results.empty: return saved
    mean_prolif = results.groupby(["condition","t"])["n_cells"].mean().reset_index()
    mean_surv = results.groupby(["condition","t"])["survival_frac"].mean().reset_index()
    for cond in results["condition"].unique():
        sub_p = mean_prolif[mean_prolif["condition"]==cond]
        fig1 = plt.figure(); plt.plot(sub_p["t"], sub_p["n_cells"], marker="o")
        plt.xlabel("Frame"); plt.ylabel("Nombre de cellules"); plt.title(f"Prolifération — {cond}")
        p1 = os.path.join(out_dir, f"proliferation_{cond}.png"); fig1.savefig(p1, bbox_inches="tight", dpi=160)
        plt.close(fig1); saved.append(p1)

        sub_s = mean_surv[mean_surv["condition"]==cond]
        fig2 = plt.figure(); plt.plot(sub_s["t"], sub_s["survival_frac"], marker="o")
        plt.xlabel("Frame"); plt.ylabel("Survie"); plt.title(f"Survie — {cond}"); plt.ylim(0,1.05)
        p2 = os.path.join(out_dir, f"survival_{cond}.png"); fig2.savefig(p2, bbox_inches="tight", dpi=160)
        plt.close(fig2); saved.append(p2)
    return saved

def _select_sharpest_frame(stack, logger=None):
    from skimage.filters import laplace
    sharpness = [np.var(laplace(f)) for f in stack]
    best_idx = int(np.argmax(sharpness))
    if logger: logger(f"Frame la plus nette: index {best_idx} (score={sharpness[best_idx]:.2e})")
    return stack[best_idx]

def auto_adjust_gam_params(path, logger=None):
    stack = _load_stack(path, logger=logger)
    pixel_size, dt = _read_metadata(path, logger=logger)
    if pixel_size is None:
        pixel_size = 0.1   # µm/pixel par défaut
    if dt is None:
        dt = 60.0          # 60 sec = 1 min par défaut
    frame = _select_sharpest_frame(stack)
    contrast = np.std(frame)
    params = dict(ADV_PARAMS)
    if contrast < 0.07:
        params["sigma"] = 1.0; params["clahe_clip"] = 0.06; params["min_size"] = 15
    elif contrast < 0.12:
        params["sigma"] = 1.3; params["clahe_clip"] = 0.04; params["min_size"] = 25
    else:
        params["sigma"] = 1.6; params["clahe_clip"] = 0.025; params["min_size"] = 30
    return params

def generate_overlay_preview(path, sigma, min_size, clahe_clip, deep_enhance, logger=None):
    import io, base64
    from skimage import exposure

    if logger:
        logger(f"Preview IA sur: {os.path.basename(path)}")

    stack = _load_stack(path, logger=logger)
    frame = _select_sharpest_frame(stack, logger=logger)

    # 🔥 Correction : remettre l'image en 0-255 pour Cellpose
    img = exposure.rescale_intensity(frame, in_range="image", out_range=(0, 1))
    img = img.astype(np.float32)

    # INITIALISATION CELLPOSE POUR LA PREVIEW
    global GLOBAL_MODEL

    # --- GPU ---
    use_gpu = torch.cuda.is_available() if TORCH_OK else False

    if GLOBAL_MODEL is None and CELLPOSE_OK:
        if logger: logger("[INFO] Initialisation du modèle Cellpose pour la PREVIEW…")
        try:
            GLOBAL_MODEL = CellposeModel(model_type="cyto3sam", gpu=use_gpu)
        except Exception as e:
            if logger: logger(f"[ERREUR] Échec initialisation Cellpose preview : {e}")


    # --- Cellpose ---
    mask = None
    if CELLPOSE_OK:
        if logger: logger("Tentative de segmentation via Cellpose (cyto3)…")
        try:
            mask = _segment_frame_cellpose(img, logger=logger)
        except Exception as e:
            if logger: logger(f"Cellpose a échoué ({e}) — fallback GAM++.")

    # --- Fallback GAM++ ---
    if mask is None or np.max(mask) == 0:
        mask = _segment_frame_gam(
            img,
            sigma=float(sigma or ADV_PARAMS["sigma"]),
            min_size=int(min_size or ADV_PARAMS["min_size"]),
            max_size=int(ADV_PARAMS["max_size"]),
            clahe_clip=float(clahe_clip or ADV_PARAMS["clahe_clip"]),
            edge_margin=int(ADV_PARAMS["edge_margin"]),
            deep_enhance=deep_enhance if deep_enhance is not None else ADV_PARAMS["deep_enhance"],
        )
        if logger: logger("↩️ Fallback GAM++ effectué.")

    # --- Contours ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    if mask is not None and np.max(mask) > 0:
        ax.contour(mask, colors="lime", linewidths=0.8)
        if logger: logger(f"Preview : {int(mask.max())} objets détectés.")
    else:
        if logger: logger("Aucun objet détecté pour la preview.")

    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if logger: logger("Preview encodée (base64) prête pour affichage Flet.")
    return img_b64
