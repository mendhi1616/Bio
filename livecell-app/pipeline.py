import matplotlib
matplotlib.use("Agg")

import os, time, io, base64
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import ndimage as ndi, stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
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
    "min_size": 600,
    "max_size": 10000,
    "clahe_clip": 0.03,
    "edge_margin": 30,
    "deep_enhance": False,
}

import tifffile
import re

def _read_metadata(path, logger=None):
    pixel_size = None
    dt = None

    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            desc_tag = tags.get("ImageDescription")

            if logger:
                logger("------- METADATA RAW -------")

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


            if not desc_value:
                return None, None

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

            patterns_dt = [
                r"FrameTime\s*=\s*([\d\.]+)",
                r"TimeIncrement\s*=\s*([\d\.]+)",
                r"finterval\s*=\s*([\d\.]+)",   
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

    if dt is not None and dt > 5:
        dt = dt / 1000.0

    if logger:
        logger(f"[META] pixel_size = {pixel_size} µm/pixel — dt = {dt} s")

    return pixel_size, dt

from cellpose.models import CellposeModel
GLOBAL_MODEL = None

def process_file(path, seg_method="auto", logger=None, debug=False, fast_mode=False):
    import traceback
    t0 = time.time()

    if logger:
        logger(f"➡️ Fichier: {os.path.basename(path)} — méthode={seg_method} — fast={fast_mode}")

    try:
        stack = _load_stack(path, logger=logger)

        pixel_size, dt = _read_metadata(path, logger=logger)
        if pixel_size is None:
            pixel_size = 0.1
        if dt is None:
            dt = 60.0

        if fast_mode:
            if logger: logger("Mode Rapide activé : traitement de 1 frame sur 2.")
            stack = stack[::2]
            dt *= 2.0

        if seg_method == "auto":
            method = "cellpose" if CELLPOSE_OK else "gam"
        else:
            method = seg_method

        use_gpu = torch.cuda.is_available()

        global GLOBAL_MODEL
        if method == "cellpose" and GLOBAL_MODEL is None:
            if logger: logger("[INFO] Initialisation du modèle Cellpose…")
            GLOBAL_MODEL = CellposeModel(gpu=use_gpu)

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

        tracks = _track_labels(labels_list, max_dist=45.0, logger=logger)

        if tracks is None or len(tracks) == 0:
            if logger:
                logger("[WARN] Tracking vide.")
            
            metrics = pd.DataFrame({"file": [os.path.basename(path)]})
            return metrics, pd.DataFrame(), stack, labels_list, pd.DataFrame()

        def filter_short_tracks(tracks, min_frames=5, logger=None):
            if tracks is None or tracks.empty:
                return tracks

            counts = tracks["track_id"].value_counts()
            
            valid_ids = counts[counts >= min_frames].index
            
            n_removed = len(counts) - len(valid_ids)
            
            if n_removed > 0 and logger:
                logger(f"Nettoyage : {n_removed} pistes courtes supprimées (< {min_frames} frames)")
                
            return tracks[tracks["track_id"].isin(valid_ids)].copy()


        tracks = compute_motion_features(tracks, pixel_size, dt)


        mitoses = detect_mitosis_events(tracks, max_dist=30.0)
        if not mitoses.empty and logger:
            logger(f"    Mitoses détectées : {len(mitoses)} événements")

        metrics = _metrics_from_tracks(tracks)
        metrics["file"] = os.path.basename(path)

        if debug:
            try:
                out_dir = os.path.join(os.path.dirname(path), "outputs")
                os.makedirs(out_dir, exist_ok=True)

                overlay_path = os.path.join(out_dir, "overlay_" + os.path.basename(path) + ".png")
                
                mid = len(stack) // 2

                if mid < len(labels_list):
                    fig, ax = plt.subplots()
                    ax.imshow(stack[mid], cmap="gray")
                    ax.contour(labels_list[mid], colors="lime", linewidths=0.9)
                    ax.axis("off")
                    fig.savefig(overlay_path, dpi=160, bbox_inches="tight")
                    plt.close(fig)

                    if logger:
                        logger(f"🖼 Overlay sauvegardé : {overlay_path}")
                else:
                    if logger: logger(f"[WARN] Overlay ignoré : index {mid} hors limites (stack={len(stack)}, masques={len(labels_list)})")

            except Exception as e_img:
                if logger: logger(f"[WARN] Erreur génération image debug : {e_img}")

        return metrics, tracks, stack, labels_list, mitoses

    except Exception as e:
        if logger: 
            logger(f"[CRASH] Erreur critique dans process_file: {e}")
            logger(traceback.format_exc()) 
        raise e


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

    i_min, i_max = float(img.min()), float(img.max())
    if i_max > i_min:
        img = (img - i_min) / (i_max - i_min)
    img = img.astype(np.float32, copy=False)

    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        raise RuntimeError("GLOBAL_MODEL non initialisé.")

    try:
        pred = GLOBAL_MODEL.eval(img, channels=[0, 0])
        if isinstance(pred, dict):
            masks = pred.get("masks", None)
        elif isinstance(pred, (list, tuple)):
            masks = pred[0]  
        else:
            masks = pred  

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
    import cv2
    import logging
    from skimage import morphology 
    import numpy as np
    
    logging.getLogger("cellpose").setLevel(logging.ERROR)

    labels_list = []

    if logger:
        logger(f"  ↳ Segmentation: méthode = {method}")

    if method == "auto":
        method = "cellpose" if CELLPOSE_OK else "gam"

    if method == "cellpose" and CELLPOSE_OK:
        global GLOBAL_MODEL
        use_gpu = torch.cuda.is_available() if TORCH_OK else False
        
        if GLOBAL_MODEL is None:
            if logger: logger(f"[INFO] Init Cellpose (GPU={use_gpu})")
            GLOBAL_MODEL = CellposeModel(gpu=use_gpu)

        try:
            h, w = stack.shape[-2:]
            scale = 0.75 if (h > 800 or w > 800) else 1.0
            diam = 30.0 * scale

            if logger and scale != 1.0: 
                logger(f"Turbo activé: Analyse à {int(scale*100)}% de la taille")

            inputs = [
                cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) 
                if scale != 1.0 else img
                for img in stack
            ]

            chunk_size = 50
            
            for i in range(0, len(inputs), chunk_size):
                batch = inputs[i : i + chunk_size]
                
                if logger: 
                    logger(f"    ... Cellpose: {i+1}-{min(i+chunk_size, len(inputs))}/{len(inputs)}")

                res = GLOBAL_MODEL.eval(
                    batch, 
                    batch_size=16, 
                    channels=[0,0], 
                    diameter=diam, 
                    do_3D=False
                )
                
                masks = res[0] if isinstance(res, tuple) else res
                masks = masks if isinstance(masks, list) else [masks[k] for k in range(masks.shape[0])]

                if scale != 1.0:
                    resized = [
                        cv2.resize(m.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST) 
                        for m in masks
                    ]
                    labels_list.extend(resized)
                else:
                    labels_list.extend(masks)
                
                if use_gpu and TORCH_OK:
                    torch.cuda.empty_cache()

            if logger: logger(f"    • Cellpose terminé : {len(labels_list)} frames.")

        except Exception as e:
            if logger: logger(f"[ERREUR] Batch échoué ({e}) -> Fallback frame-by-frame.")
            labels_list = []
            for frame in stack:
                labels_list.append(_segment_frame_cellpose(frame, logger=None))

        min_sz = kwargs.get("min_size", ADV_PARAMS["min_size"])
        
        if min_sz > 1 and labels_list:
            if logger: logger(f"   Nettoyage des débris < {min_sz} px")
            labels_list = [
                morphology.remove_small_objects(m, min_size=min_sz).astype(np.int32)
                for m in labels_list
            ]

        return labels_list

    deep = kwargs.get("deep_enhance", ADV_PARAMS["deep_enhance"])
    
    for t, frame in enumerate(stack):
        labels_list.append(_segment_frame_gam(
            frame,
            sigma=kwargs.get("sigma", ADV_PARAMS["sigma"]),
            min_size=kwargs.get("min_size", ADV_PARAMS["min_size"]),
            max_size=kwargs.get("max_size", ADV_PARAMS["max_size"]),
            clahe_clip=kwargs.get("clahe_clip", ADV_PARAMS["clahe_clip"]),
            edge_margin=kwargs.get("edge_margin", ADV_PARAMS["edge_margin"]),
            deep_enhance=deep,
        ))
        if logger and t % 10 == 0: logger(f"    • GAM++: {t+1}/{len(stack)}")

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


def _track_labels(labels_list, max_dist=15.0, logger=None, gap_frames=2):
    if logger:
        logger("  ↳ Tracking (LAP + gap closing)…")

    tracks = []
    prev = None
    track_end = {}     
    pending_start = {}  

    next_track_id = 1
    cols_to_keep = [
        "t", "label", "x", "y", "area", "perimeter",
        "eccentricity", "major_axis_length", "minor_axis_length",
        "solidity", "feret_diameter_max", "track_id"
    ]

    for t, lab in enumerate(labels_list):

        df = _props_from_labels(lab).copy()
        df["t"] = t
        df.index = pd.Index([f"t{t}_l{int(x)}" for x in df["label"]], name="obj")

        if prev is None:
            df["track_id"] = range(next_track_id, next_track_id + len(df))
            for _, row in df.iterrows():
                track_end[row["track_id"]] = (row["x"], row["y"], t)
            next_track_id += len(df)
            
            if not df.empty:
                tracks.append(df[cols_to_keep])

            prev = df
            continue

        if len(prev) > 0 and len(df) > 0:
            cost = cdist(prev[["x", "y"]], df[["x", "y"]])
            cost[cost > max_dist] = 1e9
            rows, cols = linear_sum_assignment(cost)
        else:
            rows, cols = [], []

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

        for obj in df.index:
            if obj not in used:
                new_tid = next_track_id
                next_track_id += 1

                track_id.loc[obj] = new_tid
                pending_start[obj] = (df.loc[obj, "x"], df.loc[obj, "y"], t, new_tid)

        df["track_id"] = track_id.values

        if not df.empty:
            tracks.append(df[cols_to_keep])

        prev = df

        if logger and (t % 10 == 0 or t == len(labels_list)-1):
            logger(f"    • Frames trackées: {t+1}/{len(labels_list)}")


    if logger:
        logger("  ↳ Gap closing (Reconnexion)...")

    if pending_start and tracks:
        sorted_pending = sorted(pending_start.items(), key=lambda x: x[1][2]) 
        
        for obj, (x2, y2, t2, new_tid) in sorted_pending:
            best_tid = None
            best_dist = 9999

            for tid, (x1, y1, t1) in track_end.items():
                delta_t = t2 - t1
                
                if 1 <= delta_t <= gap_frames:
                    dist = np.hypot(x2 - x1, y2 - y1)                  
                    dynamic_max_dist = max_dist * delta_t
                    
                    if dist < dynamic_max_dist and dist < best_dist:
                        best_tid = tid
                        best_dist = dist

            if best_tid is not None:
                track_end[best_tid] = (x2, y2, t2)
                for df_track in tracks:
                    mask = df_track["track_id"] == new_tid
                    if mask.any():
                        df_track.loc[mask, "track_id"] = best_tid
                
                if logger:
                    logger(f"    ↳ Reconnexion : ID temporaire {new_tid} -> ID {best_tid}")

    if not tracks:
        if logger: logger("[WARN] Aucune cellule suivie détectée.")
        return pd.DataFrame(columns=cols_to_keep)

    out = pd.concat(tracks).reset_index(drop=True)
    return out

def compute_motion_features(tracks, pixel_size, dt):

    tracks = tracks.sort_values(["track_id", "t"])

    tracks["dx"] = tracks.groupby("track_id")["x"].diff()
    tracks["dy"] = tracks.groupby("track_id")["y"].diff()

    tracks["distance_px"] = np.sqrt(tracks["dx"]**2 + tracks["dy"]**2)
    tracks["distance_px"] = tracks["distance_px"].fillna(0)

    if pixel_size is None or pixel_size <= 0:
        pixel_size = 1.0
    if dt is None or dt <= 0:
        dt = 1.0

    tracks["distance_um"] = tracks["distance_px"] * pixel_size
    tracks["speed_um_s"] = tracks["distance_um"] / dt
    tracks["vx_um_s"] = tracks["dx"] * pixel_size / dt
    tracks["vy_um_s"] = tracks["dy"] * pixel_size / dt
    tracks["angle_deg"] = np.degrees(np.arctan2(tracks["vy_um_s"], tracks["vx_um_s"]))
    tracks["cum_distance_um"] = tracks.groupby("track_id")["distance_um"].cumsum()
    tracks["speed"] = tracks["speed_um_s"]
    tracks["cum_distance"] = tracks["cum_distance_um"]
    tracks["area_um2"] = tracks["area"] * (pixel_size ** 2)
    tracks["perimeter_um"] = tracks["perimeter"] * pixel_size
    tracks["circularity"] = np.nan
    valid = tracks["perimeter_um"] > 0
    tracks.loc[valid, "circularity"] = (
        4 * np.pi * tracks.loc[valid, "area_um2"] /
        (tracks.loc[valid, "perimeter_um"] ** 2)
    )

    tracks["aspect_ratio"] = np.nan
    valid_minor = tracks["minor_axis_length"] > 0
    tracks.loc[valid_minor, "aspect_ratio"] = (
        tracks.loc[valid_minor, "major_axis_length"] /
        tracks.loc[valid_minor, "minor_axis_length"]
    )

    tracks["feret_max_um"] = tracks["feret_diameter_max"] * pixel_size

    def _net_disp_um(df):
        if len(df) < 2:
            return 0.0
        dx = df["x"].iloc[-1] - df["x"].iloc[0]
        dy = df["y"].iloc[-1] - df["y"].iloc[0]
        return np.sqrt(dx**2 + dy**2) * pixel_size

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
    from matplotlib.figure import Figure
    import os
    
    os.makedirs(out_dir, exist_ok=True)
    
    figures = [] 
    
    if results is None or results.empty:
        return figures

    mean_prolif = results.groupby(["condition","t"])["n_cells"].mean().reset_index()
    mean_surv = results.groupby(["condition","t"])["survival_frac"].mean().reset_index()

    unique_conds = results["condition"].unique()

    for cond in unique_conds:
        sub_p = mean_prolif[mean_prolif["condition"]==cond]
        
        fig1 = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.plot(sub_p["t"], sub_p["n_cells"], marker="o", color="tab:blue", label="N cellules")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Nombre moyen")
        ax1.set_title(f"Prolifération — {cond}")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        figures.append((f"Prolifération ({cond})", fig1))

        try:
            fig1.savefig(os.path.join(out_dir, f"proliferation_{cond}.png"))
        except: pass

        sub_s = mean_surv[mean_surv["condition"]==cond]
        
        fig2 = Figure(figsize=(6, 4), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.plot(sub_s["t"], sub_s["survival_frac"], marker="o", color="tab:green", label="Survie")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Fraction (0-1)")
        ax2.set_title(f"Survie — {cond}")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()

        figures.append((f"Survie ({cond})", fig2))

        try:
            fig2.savefig(os.path.join(out_dir, f"survival_{cond}.png"))
        except: pass
        
    return figures

def get_interactive_charts(results):
    import plotly.graph_objects as go
    figures = []
    
    if results is None or results.empty:
        return figures

    df = results.copy()
    
    def normalize_group(g):
        t_min = g["t"].min()
        n0 = g.loc[g["t"] == t_min, "n_cells"].mean()
        g["n_cells_norm"] = g["n_cells"] / max(1, n0) 
        return g

    try:
        df = df.groupby(["condition", "file"]).apply(normalize_group).reset_index(drop=True)
    except Exception:
        df["n_cells_norm"] = df["n_cells"]

    stats = df.groupby(["condition", "t"]).agg(
        n_mean=("n_cells", "mean"),
        n_sem=("n_cells", "sem"),       
        norm_mean=("n_cells_norm", "mean"),
        norm_sem=("n_cells_norm", "sem"),
        surv_mean=("survival_frac", "mean"),
        surv_sem=("survival_frac", "sem")
    ).reset_index()

    def add_trace_with_error(fig, df_cond, x_col, y_mean_col, y_sem_col, label, color):
        x = df_cond[x_col]
        y = df_cond[y_mean_col]
        y_err = df_cond[y_sem_col].fillna(0)
        y_upper = y + y_err
        y_lower = y - y_err

        fig.add_trace(go.Scatter(
            x=pd.concat([x, x[::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=f"rgba({color}, 0.2)", 
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name=f"{label} (SEM)"
        ))

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines+markers',
            line=dict(color=f"rgb({color})", width=2),
            name=label
        ))

    colors = ["255,0,0", "0,128,255", "0,128,0", "255,128,0", "128,0,128"]
    unique_conds = stats["condition"].unique()

    fig_p = go.Figure()
    for i, cond in enumerate(unique_conds):
        sub = stats[stats["condition"] == cond]
        c_code = colors[i % len(colors)]
        add_trace_with_error(fig_p, sub, "t", "norm_mean", "norm_sem", cond, c_code)

    fig_p.update_layout(
        title="Prolifération (Normalisée)",
        xaxis_title="Frame (t)",
        yaxis_title="Fold Change (N_t / N_0)",
        template="plotly_white",
        hovermode="x unified"
    )
    figures.append(fig_p)

    fig_s = go.Figure()
    for i, cond in enumerate(unique_conds):
        sub = stats[stats["condition"] == cond]
        c_code = colors[i % len(colors)]
        add_trace_with_error(fig_s, sub, "t", "surv_mean", "surv_sem", cond, c_code)

    fig_s.update_layout(
        title="Taux de Survie",
        xaxis_title="Frame (t)",
        yaxis_title="Fraction de survie",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        hovermode="x unified"
    )
    figures.append(fig_s)

    return figures

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
        pixel_size = 0.1   
    if dt is None:
        dt = 60.0         
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

def _select_sharpest_frame(stack, logger=None):
    from skimage.filters import laplace
    sharpness = [np.var(laplace(f)) for f in stack]
    best_idx = int(np.argmax(sharpness))
    if logger: logger(f"Frame la plus nette: index {best_idx} (score={sharpness[best_idx]:.2e})")
    return stack[best_idx], best_idx

def auto_adjust_gam_params(path, logger=None):
    stack = _load_stack(path, logger=logger)
    pixel_size, dt = _read_metadata(path, logger=logger)
    if pixel_size is None: pixel_size = 0.1
    if dt is None: dt = 60.0
    
    frame, _ = _select_sharpest_frame(stack)
    
    contrast = np.std(frame)
    params = dict(ADV_PARAMS)
    if contrast < 0.07:
        params["sigma"] = 1.0; params["clahe_clip"] = 0.06; params["min_size"] = 15
    elif contrast < 0.12:
        params["sigma"] = 1.3; params["clahe_clip"] = 0.04; params["min_size"] = 25
    else:
        params["sigma"] = 1.6; params["clahe_clip"] = 0.025; params["min_size"] = 30
    return params

def generate_overlay_preview(path, sigma, min_size, clahe_clip, deep_enhance, frame_index=None, logger=None):
    import io, base64
    from skimage import exposure, morphology 

    if logger:
        logger(f"Preview IA sur: {os.path.basename(path)}")

    stack = _load_stack(path, logger=logger)
    n_frames = len(stack)
    
    if frame_index is not None:
        idx = int(frame_index)
        idx = max(0, min(idx, n_frames - 1))
        frame = stack[idx]
        selected_idx = idx
        if logger: logger(f"Frame manuelle : {idx+1}/{n_frames}")
    else:
        frame, selected_idx = _select_sharpest_frame(stack, logger=logger)

    img = exposure.rescale_intensity(frame, in_range="image", out_range=(0, 1))
    img = img.astype(np.float32)

    global GLOBAL_MODEL
    use_gpu = torch.cuda.is_available() if TORCH_OK else False

    if GLOBAL_MODEL is None and CELLPOSE_OK:
        if logger: logger("[INFO] Initialisation du modèle Cellpose pour la PREVIEW…")
        try:
            GLOBAL_MODEL = CellposeModel(model_type="cyto3sam", gpu=use_gpu)
        except Exception as e:
            if logger: logger(f"[ERREUR] Échec initialisation Cellpose preview : {e}")

    mask = None
    if CELLPOSE_OK:
        try:
            mask = _segment_frame_cellpose(img, logger=logger)
            if mask is not None and min_size > 1:
                mask = morphology.remove_small_objects(mask, min_size=min_size).astype(np.int32)
                if logger: logger(f"Filtre taille appliqué (Preview): min {min_size} px")

        except Exception as e:
            if logger: logger(f"Cellpose a échoué ({e}) — fallback GAM++.")

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

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    if mask is not None and np.max(mask) > 0:
        ax.contour(mask, colors="lime", linewidths=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return img_b64, n_frames, selected_idx

def recalculate_with_new_masks(path, new_masks, logger=None):
    if logger: logger(f"🔄 Recalcul des métriques pour {os.path.basename(path)}...")

    pixel_size, dt = _read_metadata(path, logger=logger)
    if pixel_size is None: pixel_size = 0.1
    if dt is None: dt = 60.0

    tracks = _track_labels(new_masks, max_dist=20.0, logger=logger)

    if tracks is None or len(tracks) == 0:
        if logger: logger("[WARN] Tracking vide après correction.")
        metrics = pd.DataFrame({"file": [os.path.basename(path)]})
        return metrics, pd.DataFrame()

    tracks = compute_motion_features(tracks, pixel_size, dt)

    metrics = _metrics_from_tracks(tracks)
    metrics["file"] = os.path.basename(path)

    if logger: logger("✅ Recalcul terminé.")
    return metrics, tracks

def detect_mitosis_events(tracks, max_dist=35.0, relative_area_tol=0.5):
    events = []
    
    if tracks is None or tracks.empty:
        return pd.DataFrame()

    track_stats = tracks.groupby("track_id")["t"].agg(["min", "max"])
    
    t_max_movie = tracks["t"].max()
    
    for t in range(tracks["t"].min(), t_max_movie):
        dying_ids = track_stats[track_stats["max"] == t].index
        if len(dying_ids) == 0: continue
        
        born_ids = track_stats[track_stats["min"] == (t + 1)].index
        if len(born_ids) < 2: continue 
        
        mothers = tracks[(tracks["t"] == t) & (tracks["track_id"].isin(dying_ids))]
        daughters = tracks[(tracks["t"] == t + 1) & (tracks["track_id"].isin(born_ids))]
        
        if mothers.empty or daughters.empty: continue
        
        coords_m = mothers[["x", "y"]].values
        coords_d = daughters[["x", "y"]].values
        
        from scipy.spatial.distance import cdist
        dists = cdist(coords_m, coords_d)

        for i, m_id in enumerate(mothers["track_id"]):
            close_indices = np.where(dists[i] < max_dist)[0]
            
            if len(close_indices) >= 2:
                area_m = mothers.iloc[i]["area"]
                
                from itertools import combinations
                for idx1, idx2 in combinations(close_indices, 2):
                    d1 = daughters.iloc[idx1]
                    d2 = daughters.iloc[idx2]
                    
                    area_sum = d1["area"] + d2["area"]
                    
                    if (1.0 - relative_area_tol) * area_m < area_sum < (1.0 + relative_area_tol) * area_m:
                        events.append({
                            "t": t,
                            "mother_id": m_id,
                            "daughter1_id": d1["track_id"],
                            "daughter2_id": d2["track_id"],
                            "x": mothers.iloc[i]["x"],
                            "y": mothers.iloc[i]["y"]
                        })
                        break 

    return pd.DataFrame(events)