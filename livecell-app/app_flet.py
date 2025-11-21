import asyncio
import warnings
import atexit
import gc
import torch
import sys
import verify_env
verify_env.check_and_install()
import os, glob, threading
import base64
import flet as ft
import pandas as pd
from typing import Dict, List
from pipeline import (
    process_file,
    compute_group_stats,
    plot_curves,
    CELLPOSE_OK,
    generate_overlay_preview,
    set_advanced_params,
    ADV_PARAMS,
)

def silence_asyncio_errors():
        if sys.platform.startswith("win"):
            import asyncio.proactor_events
            def safe_call_soon(self, callback, *args, **kwargs):
                try:
                    return self._loop.call_soon(callback, *args, **kwargs)
                except RuntimeError as e:
                    if "Event loop is closed" not in str(e):
                        raise
            asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = safe_call_soon
            import asyncio.streams
            orig_writer_del = asyncio.streams.StreamWriter.__del__

            def safe_writer_del(self):
                try:
                    orig_writer_del(self)
                except RuntimeError as e:
                    if "Event loop is closed" not in str(e):
                        raise
            asyncio.streams.StreamWriter.__del__ = safe_writer_del

silence_asyncio_errors()
warnings.filterwarnings("ignore", message="coroutine 'StreamWriter.*' was never awaited")

def clean_exit():
    print("\nNettoyage avant fermeture de Live-Cell...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cache GPU vidé (torch.cuda).")
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    print("Nettoyage terminé.\n")

atexit.register(clean_exit)
import requests
import getpass
import socket
import json
WHITELIST_SERVER = "https://gokuelvagabon.pythonanywhere.com/"  
CLIENT_REG_KEY = "livecell-2025"   
import hashlib
import platform


def get_device_id():
    base = f"{getpass.getuser()}-{socket.gethostname()}-{platform.node()}-{platform.system()}-{platform.processor()}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]

def check_online_access(device_id=None, timeout=5.0):
    if device_id is None:
        device_id = get_device_id()
    try:
        r = requests.get(f"{WHITELIST_SERVER}/check", params={"device_id": device_id}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return data.get("access", False), data
        return False, {"message": f"HTTP {r.status_code}"}
    except Exception as ex:
        return False, {"message": f"exception: {ex}"}

def register_device(device_id=None, user_name=None, timeout=6.0):
    if device_id is None:
        device_id = get_device_id()
    payload = {"device_id": device_id, "user": user_name or "", "reg_key": CLIENT_REG_KEY}
    try:
        r = requests.post(f"{WHITELIST_SERVER}/register", json=payload, timeout=timeout)
        data = r.json() if r.status_code == 200 else {"message": f"HTTP {r.status_code}"}
        return r.status_code == 200, data
    except Exception as ex:
        return False, {"message": f"exception: {ex}"}

def check_and_register_on_start(username_hint=None, ui_notify=None):
    n = ui_notify or print
    did = get_device_id()
    n(f"{did}")
    ok, info = check_online_access(did)
    if ok:
        return True
    success, reg_info = register_device(device_id=did, user_name=username_hint)
    if success and reg_info.get("status") == "ok":
        return True
    n(f"Enregistrement: {reg_info.get('message', reg_info)}")
    return False


def draw_annotated_frame(img_orig, t, df, masks, mitoses=None):
    import cv2
    import numpy as np

    img = img_orig.copy()
    if img.dtype != 'uint8':
         img = (img * 255).astype('uint8')

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if masks is not None and len(masks) > t:
        mask = masks[t]
        if mask is not None:
            u_labels = np.unique(mask)
            for lbl in u_labels:
                if lbl == 0: continue
                bmask = (mask == lbl).astype(np.uint8)
                contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    if df is not None and not df.empty:
        history_df = df[df["t"] <= t]
        present_ids = df[df["t"] == t]["track_id"].unique()

        for tid in present_ids:
            track_path = history_df[history_df["track_id"] == tid].sort_values("t")
            pts = []
            for _, r in track_path.iterrows():
                pts.append([int(r["x"]), int(r["y"])])

            if len(pts) > 1:
                pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2)

        track_starts = df.groupby("track_id")["t"].min()
        df_t = df[df["t"] == t]
        
        for _, row in df_t.iterrows():
            x, y = int(row["x"]), int(row["y"])
            tid = int(row["track_id"])

            t_start = track_starts.get(tid, 0)
            age = t - t_start
            color = (255, 255, 0) if age < 2 else (255, 0, 255)

            cv2.circle(img, (x, y), 4, color, 2)
            cv2.putText(img, str(tid), (x+8, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if mitoses is not None and not mitoses.empty:
        recent_mitosis = mitoses[(mitoses["t"] >= t - 10) & (mitoses["t"] <= t)]
        for _, ev in recent_mitosis.iterrows():
            mx, my = int(ev["x"]), int(ev["y"])
            cv2.drawMarker(img, (mx, my), (0, 0, 255), cv2.MARKER_STAR, 10, 2)
            cv2.putText(img, "DIV", (mx-10, my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return img

def generate_video(stack, df, masks, out_path, fps=10, logger=None):
    import cv2
    if not stack.shape[0]: return

    h, w = stack.shape[1], stack.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for t in range(len(stack)):
        frame = stack[t]
        annotated = draw_annotated_frame(frame, t, df, masks)
        out.write(annotated)
        if logger and t % 10 == 0:
             logger(f"Export vidéo: frame {t}/{len(stack)}")

    out.release()
    if logger: logger("Export vidéo terminé.")


class TrackViewer(ft.Container):
    def __init__(self, stack, tracking_df, masks=None):
        super().__init__(
            bgcolor="black",
            padding=10,
            expand=True, 
            alignment=ft.alignment.center
        )

        self.stack = stack
        self.df = tracking_df
        self.masks = masks
        self.t = 0
        self.playing = False
        
        self.img_display = ft.Image(
            border_radius=5,
            fit=ft.ImageFit.CONTAIN,
            expand=True
        )
        
        self.slider = ft.Slider(
            min=0,
            max=len(stack) - 1,
            value=0,
            on_change=self.on_seek,
            expand=True 
        )

        self.play_btn = ft.IconButton(
            icon=ft.Icons.PLAY_ARROW,  
            on_click=self.toggle_play,
            icon_color="white",
            tooltip="Lecture/Pause"
        )

        self.close_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,      
            on_click=self.close,
            icon_color="red400",
            tooltip="Fermer"
        )

        self.content = ft.Column(
            [
                ft.Row(
                    [self.play_btn, self.slider, self.close_btn], 
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                self.img_display,
            ],
            expand=True,
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

        self.update_frame()

    def update_frame(self):
        img_annotated = draw_annotated_frame(self.stack[self.t], self.t, self.df, self.masks)
        import cv2
        _, buf = cv2.imencode(".png", img_annotated)
        img_base64 = base64.b64encode(buf).decode()

        self.img_display.src_base64 = img_base64
        if self.page:
            self.update()

    def on_seek(self, e):
        self.t = int(self.slider.value)
        self.update_frame()

    def toggle_play(self, e):
        self.playing = not self.playing
        self.play_btn.icon = ft.Icons.PAUSE if self.playing else ft.Icons.PLAY_ARROW
        self.update()
        if self.playing:
            threading.Thread(target=self.autoplay_loop, daemon=True).start()

    def autoplay_loop(self):
        import time
        while self.playing:
            self.t = (self.t + 1) % len(self.stack)
            self.slider.value = self.t
            self.update_frame()
            time.sleep(0.1) 

    def close(self, e):
        self.playing = False
        self.visible = False
        self.update()
        self.page.overlay.remove(self)
        self.page.update()


def main(page: ft.Page):
    check_and_register_on_start(username_hint="Groupe", ui_notify=lambda m: print(m))
    ok, info = check_online_access()
    if not ok:
        ft.alert_dialog = ft.AlertDialog(
            title=ft.Text("Erreur"),
            content=ft.Text(f"Raison : {info.get('status', 'inconnue')}\n{info.get('message', '')}"),
            actions=[ft.TextButton("Fermer", on_click=lambda _: page.window_destroy())]
        )
        page.dialog = ft.alert_dialog
        page.dialog.open = True
        page.update()
        return

    page.title = "Live-Cell — Mendhi"
    page.window_width = 1280
    page.window_height = 900
    page.scroll = ft.ScrollMode.AUTO
    data_root = ft.TextField(label="Chemin vers le dossier data/", value="data", width=520, read_only=True)
    folder_picker = ft.FilePicker()
    page.overlay.append(folder_picker)
    status = ft.Text("", selectable=True)
    pick_folder_btn = ft.ElevatedButton(
        "Choisir un dossier",
        icon="folder_open",
        on_click=lambda _: folder_picker.get_directory_path()
    )

    detect_btn = ft.ElevatedButton("Détecter les conditions", icon="search")

    seg_choices = ["gam", "gam_gpu", "cellpose", "auto", "lap"]

    if CELLPOSE_OK:
        seg_choices.append("cellpose")

    seg_method = ft.Dropdown(
        label="Méthode de segmentation",
        options=[ft.dropdown.Option(c) for c in seg_choices],
        value="auto" if CELLPOSE_OK else "gam",
        width=220,
    )

    log_view = ft.ListView(expand=False, height=200, auto_scroll=True, spacing=4)
    conditions_panel = ft.Column()
    condition_checkboxes: Dict[str, List[ft.Checkbox]] = {}

    GLOBAL_RESULTS = {}
    GLOBAL_FULL_DF = None
    CURRENT_OUTPUT_DIR = os.path.join(data_root.value, "outputs")

    global tracking_tab
    tracking_tab = None

    def log(msg: str):
        log_view.controls.append(ft.Text(msg))
        page.update()

    def detect_conditions(e=None):
        nonlocal condition_checkboxes
        conditions_panel.controls.clear()
        condition_checkboxes = {}
        root = data_root.value.strip()
        log_view.controls.clear()
        log(f"Dossier choisi: {root}")

        if not os.path.isdir(root):
            conditions_panel.controls.append(ft.Text("Dossier introuvable."))
            log("Dossier introuvable.")
            page.update()
            return

        found = []
        for name in sorted(os.listdir(root)):
            if name.lower() in ("outputs", "__pycache__", "models", ".venv", "temp_napari", "results_auto"):
                continue

            p = os.path.join(root, name)
            files = []
            for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"):
                files += glob.glob(os.path.join(p, ext))
            files = sorted(set(os.path.normcase(f) for f in files))
            if os.path.isdir(p) and files:
                found.append((name, files))

        if not found:
            conditions_panel.controls.append(
                ft.Text("Aucune condition trouvée (attendu: data/condition/*.tif, *.png, etc.).")
            )
            log("❕ Aucune condition trouvée.")
            page.update()
            return

        for cond, files in found:
            cbs = [ft.Checkbox(label=os.path.basename(f), value=True, data=f) for f in files]
            condition_checkboxes[cond] = cbs

            def make_toggle_all(name, val):
                def _(_e=None):
                    for _cb in condition_checkboxes.get(name, []):
                        _cb.value = val
                    page.update()
                return _

            tile = ft.ExpansionTile(
                title=ft.Text(f"Condition : {cond}"),
                initially_expanded=False,
                controls=[
                    ft.Text(f"{len(cbs)} fichier(s)"),
                    ft.Row([
                        ft.TextButton("Tout sélectionner", on_click=make_toggle_all(cond, True)),
                        ft.TextButton("Tout désélectionner", on_click=make_toggle_all(cond, False)),
                    ]),
                    ft.Column(cbs),
                ],
            )
            conditions_panel.controls.append(ft.Container(content=tile, padding=8, border=ft.border.all(1), border_radius=6))
        page.update()

    def on_folder_picked(e: ft.FilePickerResultEvent):
        if e.path:
            data_root.value = e.path
            log_view.controls.clear()
            log(f"📂 Dossier choisi : {e.path}")
            detect_conditions() 
            page.update()

    folder_picker.on_result = on_folder_picked

    preview_img = ft.Image(src="", width=700, height=700, fit=ft.ImageFit.CONTAIN, border_radius=8)
    spinner = ft.ProgressRing(width=40, height=40, visible=False, color="blue")

    fp = ft.FilePicker()
    save_file_picker = ft.FilePicker()
    video_save_picker = ft.FilePicker()
    page.overlay.append(fp)
    page.overlay.append(save_file_picker)
    page.overlay.append(video_save_picker)
    picked_path_text = ft.Text("Aucun fichier chargé.")
    preview_file_path: str | None = None

    auto_preview_check = ft.Switch(label="Aperçu automatique (ON/OFF)", value=True)

    fast_mode_check = ft.Switch(label="Mode Rapide (1 frame sur 2)", value=False)

    pick_btn = ft.ElevatedButton(
        "Charger une image (TIF / PNG / JPG)",
        icon="folder_open",
        on_click=lambda _: fp.pick_files(
            allow_multiple=False,
            allowed_extensions=["tif", "tiff", "png", "jpg", "jpeg", "bmp"]
        ),
    )

    file_just_loaded = False

    def on_file_picked(e: ft.FilePickerResultEvent):
        nonlocal preview_file_path, file_just_loaded
        log_view.controls.clear()
        if e.files:
            preview_file_path = e.files[0].path or e.files[0].name
            picked_path_text.value = f"Fichier sélectionné : {os.path.basename(preview_file_path)}"
            
            file_just_loaded = True
            
            page.update()
            update_preview_final()
        else:
            preview_file_path = None
            picked_path_text.value = "Aucun fichier chargé."
            preview_img.src = None
            preview_img.src_base64 = None
        
            frame_slider.disabled = True
            frame_slider.label = "Frame"
            page.update()

    fp.on_result = on_file_picked

    _preview_timer = None
    def debounce_preview():
        nonlocal _preview_timer
        if not auto_preview_check.value:
            return
            
        try:
            if _preview_timer and _preview_timer.is_alive():
                _preview_timer.cancel()
        except Exception: pass
        
        _preview_timer = threading.Timer(0.6, update_preview_final)
        _preview_timer.start()

    def update_preview_final():
        nonlocal file_just_loaded
        if not preview_file_path or not os.path.exists(preview_file_path):
            return

        try:
            log_view.controls.clear()
            spinner.visible = True
            page.update()

            req_frame = None
            if not file_just_loaded:
                req_frame = int(frame_slider.value)

            img_b64, n_total, used_idx = generate_overlay_preview(
                preview_file_path,
                sigma=float(sigma_slider.value),
                min_size=int(min_size_slider.value),
                clahe_clip=float(clahe_slider.value),
                deep_enhance=bool(deep_check.value),
                frame_index=req_frame, 
                logger=log,
            )

            if img_b64:
                preview_img.src_base64 = img_b64
                preview_img.src = None 
                
                if n_total > 1:
                    frame_slider.min = 0
                    frame_slider.max = n_total - 1
                    frame_slider.divisions = max(1, n_total - 1)
                    frame_slider.value = used_idx
                    frame_slider.label = f"Frame {used_idx+1} / {n_total}"
                    frame_slider.disabled = False
                else:
                    frame_slider.min = 0
                    frame_slider.max = 0
                    frame_slider.value = 0
                    frame_slider.label = "1/1"
                    frame_slider.disabled = True

                file_just_loaded = False

                preview_img.update()
                frame_slider.update()
                
                log(f"Preview générée (Frame {used_idx+1}/{n_total})")
                log(f"Params: σ={sigma_slider.value:.2f}, min={min_size_slider.value:.0f}, CLAHE={clahe_slider.value:.3f}")
            else:
                log("Aucune image générée.")

        except Exception as ex:
            log(f"Erreur de preview: {ex}")
            import traceback
            log(traceback.format_exc())
        finally:
            spinner.visible = False
            page.update()

    auto_preview_switch = ft.Switch(
        label="Aperçu automatique (ON/OFF)",
        value=True,
        on_change=lambda e: log("Aperçu automatique : " + ("activé" if e.control.value else "désactivé"))
    )

    def on_slider_change(e):
        if isinstance(e.control.value, float):
            e.control.label = f"{e.control.value:.3f}" if e.control.max < 10 else f"{e.control.value:.2f}"
        else:
            e.control.label = f"{int(e.control.value)}"
        e.control.update()
        if auto_preview_switch.value:
            debounce_preview()

    sigma_text = ft.Text("Flou gaussien σ", width=180)
    sigma_slider = ft.Slider(
        min=0.5, max=3.0, divisions=25,
        value=ADV_PARAMS["sigma"],
        label=f"{ADV_PARAMS['sigma']:.2f}",
        on_change=on_slider_change,
        expand=True,
    )

    frame_slider = ft.Slider(
        min=0, max=1, value=0, divisions=1,
        label="Frame {value}",
        on_change=lambda e: debounce_preview(),
        disabled=True 
    )
    frame_text = ft.Text("Navigation Frame", width=180)

    min_text = ft.Text("Taille minimale (px)", width=180)
    min_size_slider = ft.Slider(
        min=5, 
        max=2000,       
        divisions=200, 
        value=float(ADV_PARAMS["min_size"]),
        label=f"{ADV_PARAMS['min_size']:.0f}",
        on_change=on_slider_change,
        expand=True,
    )

    clahe_text = ft.Text("Contraste CLAHE", width=180)
    clahe_slider = ft.Slider(
        min=0.005, max=0.08, divisions=15,
        value=ADV_PARAMS["clahe_clip"],
        label=f"{ADV_PARAMS['clahe_clip']:.3f}",
        on_change=on_slider_change,
        expand=True,
    )

    deep_check = ft.Switch(
        label="Amélioration profonde (Deep Enhance)",
        value=False,
        on_change=lambda e: debounce_preview()
    )

    controls_panel = ft.Card(
        ft.Container(
            content=ft.Column(
                [
                    ft.Text("Réglages interactifs (Ajuste en direct la preview)", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row([sigma_text, sigma_slider], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Row([min_text, min_size_slider], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Row([clahe_text, clahe_slider], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Divider(),
                    ft.Row([frame_text, frame_slider], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Divider(),
                    deep_check,
                    auto_preview_switch,
                    ft.Divider(),
                    ft.ElevatedButton(
                        "Appliquer ces paramètres à l'analyse",
                        icon="check_circle",
                        on_click=lambda _: (
                            log_view.controls.clear(),
                            set_advanced_params(
                                sigma=float(sigma_slider.value),
                                min_size=int(min_size_slider.value),
                                clahe_clip=float(clahe_slider.value),
                                deep_enhance=bool(deep_check.value),
                            ),
                            log("💾 Paramètres GAM appliqués à l'analyse."),
                            page.update()
                        )
                    ),
                ],
                spacing=8,
            ),
            padding=10,
        )
    )

    preview_controls = ft.Column(
        [
            ft.Text("Prévisualisation IA (Cellpose)", weight=ft.FontWeight.BOLD, size=16),
            ft.Row([pick_btn, picked_path_text]),
            ft.Divider(),
            ft.Row(
                [
                    ft.Column([controls_panel], width=380),
                    ft.Column([spinner, preview_img], alignment=ft.MainAxisAlignment.CENTER),
                ],
                spacing=16,
            ),
        ],
        spacing=8,
    )

    tracking_tab = ft.Column([], scroll=ft.ScrollMode.AUTO)

    run_btn = ft.ElevatedButton(
        "Lancer l'analyse",
        icon=ft.Icon(name="play_arrow"),
    )

    def toggle_ui(enabled: bool):
        for ctrl in [
            run_btn, detect_btn, seg_method, data_root,
            sigma_slider, min_size_slider, clahe_slider, deep_check, fast_mode_check
        ]:
            ctrl.disabled = not enabled
        page.update()


    def save_csv_callback(e, df):
        if e.path and df is not None:
             try:
                df.to_csv(e.path, index=False)
                log(f"[EXPORT] CSV sauvegardé : {e.path}")
                page.snack_bar = ft.SnackBar(ft.Text(f"Sauvegardé : {e.path}"))
                page.snack_bar.open = True
                page.update()
             except Exception as ex:
                log(f"[ERREUR] Export CSV : {ex}")

    def video_save_callback(e: ft.FilePickerResultEvent):
        if e.path and e.control.data:
            out_path = e.path
            data = e.control.data
            stack = data["stack"]
            df = data["df"]
            masks = data["masks"]

            log(f"Génération vidéo vers : {out_path} ...")
            page.snack_bar = ft.SnackBar(ft.Text(f"Génération vidéo en cours..."))
            page.snack_bar.open = True
            page.update()

            def _thread_target():
                try:
                    generate_video(stack, df, masks, out_path, fps=10, logger=print)
                    log(f"[EXPORT] Vidéo terminée : {out_path}")
                except Exception as ex:
                    log(f"[ERREUR] Vidéo : {ex}")

            threading.Thread(target=_thread_target, daemon=True).start()

    video_save_picker.on_result = video_save_callback

    def run_analysis(e=None):
        nonlocal GLOBAL_RESULTS, GLOBAL_FULL_DF, CURRENT_OUTPUT_DIR
        root = data_root.value.strip()
        import time

        GLOBAL_RESULTS = {}
        all_results_meta = []

        if not os.path.isdir(root):
            status.value = "Le dossier data/ est introuvable."
            page.update()
            return

        selection: Dict[str, List[str]] = {}
        for cond, cbs in condition_checkboxes.items():
            picked = [cb.data for cb in cbs if cb.value]
            if picked:
                selection[cond] = picked

        if not selection:
            status.value = "Sélectionnez au moins une image."
            page.update()
            return
        
        timestamp = time.strftime("%Y-%m-%d_%Hh%M")
        total_files = sum(len(v) for v in selection.values())
        
        if total_files == 1:
            first_file = list(selection.values())[0][0]
            base_name = os.path.splitext(os.path.basename(first_file))[0]
            run_folder_name = f"{timestamp}_{base_name}"
        else:
            run_folder_name = f"{timestamp}_Batch_{total_files}files"

        CURRENT_OUTPUT_DIR = os.path.join(root, "outputs", run_folder_name)
        os.makedirs(CURRENT_OUTPUT_DIR, exist_ok=True)
        
        log(f"📂 Résultats sauvegardés dans : outputs/{run_folder_name}")

        toggle_ui(False)
        progress = ft.ProgressBar(width=520)
        tabs.tabs[0].content.controls[:] = [ft.Text("Analyse en cours..."), progress, log_view]
        
        method = seg_method.value
        if deep_check.value and method not in ("cellpose",):
            method = "gam_gpu"

        use_fast = bool(fast_mode_check.value)
        log(f"Analyse — méthode: {method} — deep={deep_check.value} — fast={use_fast}")

        total = sum(len(v) for v in selection.values())
        done = 0

        for cond, files in selection.items():
            log(f"▶️ {cond}: {len(files)} fichier(s)")
            for pth in files:
                status.value = f"Traitement: {cond} / {os.path.basename(pth)}"
                page.update()

                try:
                    fname = os.path.basename(pth)
                    metrics, tracks, current_stack, current_masks, mitoses = process_file(
                        pth, seg_method=method, logger=log, debug=False, fast_mode=use_fast
                    )

                    GLOBAL_RESULTS[fname] = {
                        "tracks": tracks,
                        "stack": current_stack,
                        "masks": current_masks,
                        "metrics": metrics,
                        "mitoses": mitoses,
                        "condition": cond
                    }

                    img_dir = os.path.dirname(pth)
                    save_dir = os.path.join(img_dir, "results_auto")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    base_name = os.path.splitext(fname)[0]
                    if tracks is not None and not tracks.empty:
                        tracks.to_csv(os.path.join(save_dir, f"{base_name}_tracks.csv"), index=False)
                    
                    metrics.to_csv(os.path.join(save_dir, f"{base_name}_metrics.csv"), index=False)

                    metrics["condition"] = cond
                    all_results_meta.append(metrics)

                    if tracks is not None and not tracks.empty:
                        n_unique = tracks["track_id"].nunique()
                        log(f"✅ {fname} : {n_unique} cellules uniques suivies.")
                    else:
                        log(f"⚠️ {fname} : Aucune cellule suivie.")

                except Exception as ex:
                    log(f"Erreur {os.path.basename(pth)} : {ex}")
                    import traceback
                    log(traceback.format_exc())

                done += 1
                progress.value = done / max(1, total)
                page.update()

        toggle_ui(True)
        status.value = "Analyse terminée"
        page.update()

        if not all_results_meta:
            log("Aucun résultat.")
            return

        GLOBAL_FULL_DF = pd.concat(all_results_meta, ignore_index=True)
        log(f"Résultats globaux : {len(GLOBAL_FULL_DF)} lignes de métriques.")

        tracking_tab = tabs.tabs[2].content
        tracking_tab.controls.clear()

        file_options = [ft.dropdown.Option(k) for k in GLOBAL_RESULTS.keys()]
        if not file_options:
            tracking_tab.controls.append(ft.Text("Aucune donnée valide.", color="red300"))
            tracking_tab.update()
            return

        current_file_key = file_options[0].key

        file_dropdown = ft.Dropdown(
            label="Fichier à visualiser",
            options=file_options,
            value=current_file_key,
            width=400,
        )

        table_container = ft.Column(expand=True, scroll=ft.ScrollMode.AUTO)
        actions_container = ft.Row()

        def refresh_view(file_key):
            table_container.controls.clear()
            actions_container.controls.clear()

            if not file_key or file_key not in GLOBAL_RESULTS:
                return

            data = GLOBAL_RESULTS[file_key]
            df = data["tracks"]
            stack = data["stack"]
            masks = data.get("masks", None)

            if df is None or df.empty:
                table_container.controls.append(ft.Text("Pas de tracking pour ce fichier.", color="red300"))
                tracking_tab.update()
                return

            columns = [
                "track_id", "t", "x", "y",
                "speed_um_s", "cum_distance_um",
                "area_um2", "circularity", "eccentricity",
                "aspect_ratio", "solidity", "feret_max_um", "angle_deg", "straightness",
            ]

            rows_per_page = 50
            current_page = [0]
            total_rows = len(df)
            total_pages = (total_rows + rows_per_page - 1) // rows_per_page

            table = ft.DataTable(
                columns=[ft.DataColumn(ft.Text(col)) for col in columns],
                rows=[],
                horizontal_margin=10,
                column_spacing=20,
            )
            page_info = ft.Text(f"Page 1 / {max(1, total_pages)}")

            def update_table_rows():
                start = current_page[0] * rows_per_page
                end = start + rows_per_page
                subset = df.iloc[start:end]
                table.rows.clear()
                for _, r in subset.iterrows():
                    cells = []
                    for col in columns:
                        val = r.get(col, "")
                        txt = str(val) if isinstance(val, str) else f"{val:.2f}"
                        cells.append(ft.DataCell(ft.Text(txt)))
                    table.rows.append(ft.DataRow(cells=cells))
                page_info.value = f"Page {current_page[0] + 1} / {max(1, total_pages)}"
                table.update()
                page_info.update()

            def prev_page(e):
                if current_page[0] > 0:
                    current_page[0] -= 1
                    update_table_rows()

            def next_page(e):
                if current_page[0] < total_pages - 1:
                    current_page[0] += 1
                    update_table_rows()

            controls_row = ft.Row(
                [
                    ft.IconButton(icon="arrow_back", on_click=prev_page),
                    page_info,
                    ft.IconButton(icon="arrow_forward", on_click=next_page),
                ],
                alignment=ft.MainAxisAlignment.CENTER
            )

            table_container.controls.append(controls_row)
            table_container.controls.append(table)
            tracking_tab.update()

            update_table_rows()

            def open_viewer_click(e):
                if stack is None: return
                viewer = TrackViewer(stack, df, masks=masks)
                page.overlay.append(viewer)
                page.update()

            def export_current_csv(e):
                clean_columns = [
                    "track_id", "t", "x", "y",
                    "speed_um_s", "cum_distance_um",
                    "area_um2", "circularity", "eccentricity",
                    "aspect_ratio", "solidity", "feret_max_um", 
                    "angle_deg", "straightness"
                ]
                
                valid_cols = [c for c in clean_columns if c in df.columns]
                df_clean = df[valid_cols].copy()
                
                df_clean = df_clean.round(3)
                
                if "track_id" in df_clean.columns and "t" in df_clean.columns:
                    df_clean = df_clean.sort_values(["track_id", "t"])

                save_file_picker.data = df_clean 
                
                save_file_picker.save_file(
                    dialog_title=f"Sauvegarder {file_key}.csv",
                    file_name=f"{file_key}_tracking_clean.csv",
                    allowed_extensions=["csv"]
                )

            def export_video_click(e):
                video_save_picker.data = {
                    "stack": stack,
                    "df": df,
                    "masks": masks
                }
                video_save_picker.save_file(
                    dialog_title=f"Exporter Vidéo {file_key}.mp4",
                    file_name=f"{file_key}_tracking.mp4",
                    allowed_extensions=["mp4"]
                )

            actions_container.controls.extend([
                ft.ElevatedButton("👁 Visualiser Track", on_click=open_viewer_click, icon="remove_red_eye"),
                ft.ElevatedButton("Export CSV", icon="download", on_click=export_current_csv, bgcolor="blue700", color="white"),
                ft.ElevatedButton("Export Vidéo (.mp4)", icon="videocam", on_click=export_video_click, bgcolor="green700", color="white"),
            ])
            actions_container.update()

        def on_file_change(e):
            refresh_view(file_dropdown.value)

        file_dropdown.on_change = on_file_change

        out_dir = os.path.join(root, "outputs")

        def show_graphs(e):
            import importlib
            import pipeline
            import io
            import base64
            
            try:
                page.snack_bar = ft.SnackBar(ft.Text("Génération des graphiques..."))
                page.snack_bar.open = True
                page.update()

                if GLOBAL_FULL_DF is None or GLOBAL_FULL_DF.empty:
                    page.snack_bar = ft.SnackBar(ft.Text("Aucune donnée à afficher. Lancez une analyse d'abord !", color="red"))
                    page.snack_bar.open = True
                    page.update()
                    return

                importlib.reload(pipeline)
                from pipeline import plot_curves

                figs_data = plot_curves(GLOBAL_FULL_DF, out_dir=CURRENT_OUTPUT_DIR)

                dlg_content = ft.Column(scroll=ft.ScrollMode.AUTO, height=600, width=900)

                if not figs_data:
                    dlg_content.controls.append(
                        ft.Text("Erreur : La fonction plot_curves n'a rien renvoyé.", color="red")
                    )
                else:
                    for title, fig in figs_data:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                        buf.seek(0)
                        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                        
                        chart_img = ft.Image(
                            src_base64=img_b64,
                            fit=ft.ImageFit.CONTAIN,
                            expand=True
                        )

                        chart_container = ft.Container(
                            content=chart_img,
                            height=450, 
                            padding=10,
                            border=ft.border.all(1, "grey"),
                            border_radius=5,
                            margin=5
                        )
                        
                        dlg_content.controls.append(
                            ft.Column([
                                ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
                                chart_container
                            ])
                        )

                dlg = ft.AlertDialog(
                    title=ft.Text("Résultats Graphiques"),
                    content=dlg_content,
                    actions=[ft.TextButton("Fermer", on_click=lambda _: page.close_dialog())],
                )
                page.dialog = dlg
                dlg.open = True
                page.update()

            except Exception as ex:
                log(f"[CRASH] Affichage courbes : {ex}")
                import traceback
                log(traceback.format_exc())


        # --- COMPARAISON (Version Corrigée avec Sauvegarde) ---
        def show_comparison(e):
            try:
                files = list(GLOBAL_RESULTS.keys())
                if len(files) < 2:
                    page.snack_bar = ft.SnackBar(ft.Text("Il faut analyser au moins 2 fichiers pour comparer."))
                    page.snack_bar.open = True
                    page.update()
                    return

                dd1 = ft.Dropdown(options=[ft.dropdown.Option(f) for f in files], value=files[0], label="Condition A", expand=True)
                dd2 = ft.Dropdown(options=[ft.dropdown.Option(f) for f in files], value=files[1] if len(files)>1 else files[0], label="Condition B", expand=True)
                
                result_area = ft.Column(scroll=ft.ScrollMode.AUTO, height=350)
                
                # Variable pour garder le texte en mémoire
                report_data = {"text": ""}

                def save_report_click(e):
                    if not report_data["text"]:
                        page.snack_bar = ft.SnackBar(ft.Text("Rien à sauvegarder. Lancez la comparaison d'abord."))
                        page.snack_bar.open = True
                        page.update()
                        return
                    
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # On utilise le dossier de sortie actuel
                    out = CURRENT_OUTPUT_DIR if 'CURRENT_OUTPUT_DIR' in locals() else "outputs"
                    os.makedirs(out, exist_ok=True)
                    
                    filename = f"rapport_comparaison_{timestamp}.md"
                    path = os.path.join(out, filename)
                    
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(report_data["text"])
                        
                        page.snack_bar = ft.SnackBar(ft.Text(f"Rapport sauvegardé : {filename}"))
                        page.snack_bar.open = True
                        page.update()
                        log(f"💾 Rapport sauvegardé : {path}")
                    except Exception as ex:
                        log(f"Erreur sauvegarde : {ex}")

                def compute_comp(e):
                    try:
                        f1, f2 = dd1.value, dd2.value
                        if not f1 or not f2: return

                        res1 = GLOBAL_RESULTS.get(f1)
                        res2 = GLOBAL_RESULTS.get(f2)
                        
                        if not res1 or not res2: return

                        df1 = res1.get("tracks")
                        df2 = res2.get("tracks")

                        if df1 is None or df1.empty or df2 is None or df2.empty:
                            result_area.controls = [ft.Text("Données manquantes (tracking vide).", color="red")]
                            result_area.update()
                            return

                        # Helper moyenne
                        def get_mean(df, col):
                            return df[col].mean() if col in df.columns else 0.0

                        # Stats Mouvement
                        col_s = "speed_um_s" if "speed_um_s" in df1.columns else "speed"
                        s1, s2 = get_mean(df1, col_s), get_mean(df2, col_s)
                        
                        col_d = "cum_distance_um" if "cum_distance_um" in df1.columns else "cum_distance"
                        d1 = df1[col_d].max() if col_d in df1.columns else 0
                        d2 = df2[col_d].max() if col_d in df2.columns else 0

                        # Stats Morpho
                        area1, area2 = get_mean(df1, "area_um2"), get_mean(df2, "area_um2")
                        circ1, circ2 = get_mean(df1, "circularity"), get_mean(df2, "circularity")
                        ar1, ar2 = get_mean(df1, "aspect_ratio"), get_mean(df2, "aspect_ratio")

                        txt = (
                            f"### 🔬 Comparaison : {f1} vs {f2}\n\n"
                            f"#### 🚀 Dynamique\n"
                            f"| Métrique | {f1} | {f2} | Delta |\n"
                            f"| :--- | :--- | :--- | :--- |\n"
                            f"| **Vitesse** (µm/s) | `{s1:.3f}` | `{s2:.3f}` | `{s1-s2:+.3f}` |\n"
                            f"| **Dist. Max** (µm) | `{d1:.1f}` | `{d2:.1f}` | `{d1-d2:+.1f}` |\n\n"
                            
                            f"#### 🧬 Morphologie\n"
                            f"| Métrique | {f1} | {f2} | Delta |\n"
                            f"| :--- | :--- | :--- | :--- |\n"
                            f"| **Aire** (µm²) | `{area1:.1f}` | `{area2:.1f}` | `{area1-area2:+.1f}` |\n"
                            f"| **Circularité** | `{circ1:.3f}` | `{circ2:.3f}` | `{circ1-circ2:+.3f}` |\n"
                            f"| **Allongement** | `{ar1:.2f}` | `{ar2:.2f}` | `{ar1-ar2:+.2f}` |"
                        )
                        
                        report_data["text"] = txt
                        result_area.controls = [ft.Markdown(txt, selectable=True)]
                        result_area.update()

                    except Exception as ex_comp:
                         result_area.controls = [ft.Text(f"Erreur calcul : {ex_comp}", color="red")]
                         result_area.update()

                dlg = ft.AlertDialog(
                    title=ft.Text("⚖️ Comparateur"),
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([dd1, dd2]),
                            ft.Row([
                                ft.ElevatedButton("Lancer", on_click=compute_comp),
                                ft.ElevatedButton("💾 Sauvegarder", on_click=save_report_click, bgcolor="blue700", color="white"),
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Divider(),
                            result_area
                        ], height=500, width=700),
                        padding=10
                    ),
                    actions=[ft.TextButton("Fermer", on_click=lambda _: page.close_dialog())]
                )
                page.dialog = dlg
                dlg.open = True
                page.update()
                
            except Exception as ex:
                log(f"[CRASH] Comparaison : {ex}")
                import traceback
                log(traceback.format_exc())

        def open_napari_viewer(e):
            import numpy as np
            import tifffile
            import subprocess
            import sys
            import os

            files = list(GLOBAL_RESULTS.keys())
            if not files: return

            key = file_dropdown.value if file_dropdown.value else files[0]
            data = GLOBAL_RESULTS[key]
            stack = data["stack"]
            masks = data.get("masks", None)

            page.snack_bar = ft.SnackBar(ft.Text(f"Ouverture de Napari pour {key}..."))
            page.snack_bar.open = True
            page.update()

            temp_dir = os.path.join(root, "temp_napari")
            os.makedirs(temp_dir, exist_ok=True)

            img_path = os.path.join(temp_dir, "img.tif")
            lbl_path = os.path.join(temp_dir, "labels.tif")

            tifffile.imwrite(img_path, stack)
            if masks is not None:
                lbl_stack = np.array(masks, dtype=np.int32)
                tifffile.imwrite(lbl_path, lbl_stack)
            else:
                if os.path.exists(lbl_path): os.remove(lbl_path)

            script_code = f"""
import napari
import tifffile
import os

def view():
    viewer = napari.Viewer()
    
    # FORCE LE MODE 2D (Empêche le crash IndexError)
    viewer.dims.ndisplay = 2

    # Chargement Image
    if os.path.exists(r'{img_path}'):
        img = tifffile.imread(r'{img_path}')
        # On précise que c'est une Time-series (pas indispensable mais plus propre)
        viewer.add_image(img, name='Image')

    # Chargement Masques
    if os.path.exists(r'{lbl_path}'):
        lbl = tifffile.imread(r'{lbl_path}')
        viewer.add_labels(lbl, name='Masks')

    napari.run()

if __name__ == '__main__':
    view()
"""
            script_path = os.path.join(temp_dir, "launch_napari.py")
            with open(script_path, "w", encoding="utf-8") as f_script:
                f_script.write(script_code)

            subprocess.Popen([sys.executable, script_path])

        def import_napari_corrections(e):
            import tifffile
            import numpy as np
            from pipeline import recalculate_with_new_masks

            file_key = file_dropdown.value
            if not file_key or file_key not in GLOBAL_RESULTS:
                return

            lbl_path = os.path.join(root, "temp_napari", "labels.tif")

            if not os.path.exists(lbl_path):
                page.snack_bar = ft.SnackBar(ft.Text("Aucun fichier 'labels.tif' trouvé dans temp_napari. Avez-vous sauvegardé dans Napari ?"))
                page.snack_bar.open = True
                page.update()
                return

            page.snack_bar = ft.SnackBar(ft.Text(f"Importation des corrections pour {file_key}..."))
            page.snack_bar.open = True
            page.update()

            try:
                new_masks_array = tifffile.imread(lbl_path)
                new_masks_list = [new_masks_array[i] for i in range(new_masks_array.shape[0])]

                cond = GLOBAL_RESULTS[file_key]["condition"]
                full_path = os.path.join(root, cond, file_key) 
                if not os.path.exists(full_path):
                    for r, d, f in os.walk(root):
                        if file_key in f:
                            full_path = os.path.join(r, file_key)
                            break
                
                new_metrics, new_tracks = recalculate_with_new_masks(full_path, new_masks_list, logger=log)

                GLOBAL_RESULTS[file_key]["masks"] = new_masks_list
                GLOBAL_RESULTS[file_key]["tracks"] = new_tracks
                
                new_metrics["condition"] = cond
                GLOBAL_RESULTS[file_key]["metrics"] = new_metrics

                nonlocal GLOBAL_FULL_DF
                all_mets = [res["metrics"] for res in GLOBAL_RESULTS.values()]
                GLOBAL_FULL_DF = pd.concat(all_mets, ignore_index=True)

                log(f"✅ Corrections appliquées pour {file_key} !")
                
                refresh_view(file_key)

            except Exception as ex:
                log(f"[ERREUR] Import impossible : {ex}")
                import traceback
                log(traceback.format_exc())

        global_actions = ft.Row([
             ft.ElevatedButton("📈 Courbes Interactives", on_click=show_graphs, icon="show_chart"),
             ft.ElevatedButton("⚖️ Comparer Résultats", on_click=show_comparison, icon="compare_arrows"),
             
             ft.Container(
                 content=ft.Row([
                     ft.ElevatedButton("🧊 Ouvrir Napari", on_click=open_napari_viewer, icon="layers", bgcolor="teal700", color="white"),
                     ft.ElevatedButton("📥 Importer Corrections", on_click=import_napari_corrections, icon="file_download", bgcolor="orange700", color="white"),
                 ], spacing=0),
                 border=ft.border.all(1, "teal"),
                 border_radius=5,
             )
        ])

        tracking_tab.controls.append(ft.Container(content=file_dropdown, padding=10))
        tracking_tab.controls.append(ft.Container(content=global_actions, padding=5))
        tracking_tab.controls.append(actions_container)
        tracking_tab.controls.append(ft.Divider())
        tracking_tab.controls.append(table_container)
        tracking_tab.update()

        refresh_view(current_file_key)

    page.update()


    run_btn.on_click = run_analysis

    tabs = ft.Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            ft.Tab(
                text="Analyse",
                content=ft.Column([log_view], expand=True)
            ),
            ft.Tab(
                text="Prévisualisation",
                content=preview_controls
            ),
            ft.Tab(
                text="Tracking",
                content=ft.Column([], scroll=ft.ScrollMode.AUTO)
            ),
        ],
        expand=True,
    )

    page.add(
        ft.Text("Test Version 2.0", weight=ft.FontWeight.BOLD, size=18),
        ft.Row([data_root, pick_folder_btn, detect_btn, seg_method]),
        ft.Row([fast_mode_check]),
        ft.Divider(),
        ft.Text("Conditions expérimentales :"),
        conditions_panel,
        ft.Divider(),
        ft.Row([run_btn, status]),
        tabs,
    )

if __name__ == "__main__":
    ft.app(target=main)