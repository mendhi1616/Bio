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


class TrackViewer(ft.Container):
    def __init__(self, stack, tracking_df, masks=None):
        super().__init__(
            bgcolor="black",
            padding=10,
            expand=True
        )

        self.stack = stack
        self.df = tracking_df
        self.masks = masks
        self.t = 0
        self.playing = False
        
        self.img_display = ft.Image(border_radius=5)
        
        self.slider = ft.Slider(
            min=0,
            max=len(stack) - 1,
            value=0,
            on_change=self.on_seek
        )

        # ---- Boutons universels (pas d'icons) ----
        self.play_btn = ft.TextButton(
            "▶",
            on_click=self.toggle_play,
            style=ft.ButtonStyle(color="white")
        )

        self.close_btn = ft.TextButton(
            "✖",
            on_click=self.close,
            style=ft.ButtonStyle(color="red400")
        )

        self.content = ft.Column(
            [
                ft.Row([self.play_btn, self.slider, self.close_btn]),
                self.img_display,
            ],
            expand=True
        )

        self.update_frame()

    # ---------------- FRAME UPDATE ----------------
    def update_frame(self):
        img = self.stack[self.t].copy()
        # Convert normalized float [0,1] to uint8 [0,255] if necessary for OpenCV
        if img.dtype != 'uint8':
             img = (img * 255).astype('uint8')

        # Ensure image is RGB (or at least 3 channels) for colored drawing
        import cv2
        import numpy as np
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 1) DESSINER LES MASQUES (CONTOURS CELLPOSE) SI DISPONIBLES
        if self.masks is not None and len(self.masks) > self.t:
            mask = self.masks[self.t]
            if mask is not None:
                # Convert mask to contours
                # Mask is int32, labels 0..N
                # We want contours for each label
                u_labels = np.unique(mask)
                for lbl in u_labels:
                    if lbl == 0: continue

                    # Create binary mask for this label
                    # uint8 for findContours
                    bmask = (mask == lbl).astype(np.uint8)

                    # Find contours
                    contours, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Draw contours in Green (0, 255, 0)
                    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        # 2) DESSINER LES QUEUES (TAILS) : historique des positions jusqu'à t
        # On filtre tout ce qui est <= t
        history_df = self.df[self.df["t"] <= self.t]

        # Pour chaque track_id présent à l'instant t
        present_ids = self.df[self.df["t"] == self.t]["track_id"].unique()

        for tid in present_ids:
            # Récupérer le chemin complet de ce track jusqu'à t
            track_path = history_df[history_df["track_id"] == tid].sort_values("t")

            pts = []
            for _, r in track_path.iterrows():
                pts.append([int(r["x"]), int(r["y"])])

            if len(pts) > 1:
                # Dessiner la polyligne
                pts_arr = np.array(pts, np.int32)
                pts_arr = pts_arr.reshape((-1, 1, 2))
                # Couleur aléatoire stable par ID ou fixe (ici jaune/orange style TrackMate)
                # OpenCV utilise BGR -> (0, 255, 255) = Jaune
                cv2.polylines(img, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2)

        # 3) DESSINER LES POINTS COURANTS
        df_t = self.df[self.df["t"] == self.t]
        for _, row in df_t.iterrows():
            x, y = int(row["x"]), int(row["y"])
            tid = int(row["track_id"])
            
            # cercle sur la cellule (Rose/Magenta style TrackMate : BGR -> 255, 0, 255)
            cv2.circle(img, (x, y), 4, (255, 0, 255), 2)
            
            # ID
            cv2.putText(
                img,
                str(tid),
                (x+8, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
                cv2.LINE_AA
            )
        
        _, buf = cv2.imencode(".png", img)
        img_base64 = base64.b64encode(buf).decode()

        self.img_display.src_base64 = img_base64
        # Only call update() if control is on page
        if self.page:
            self.update()

    def on_seek(self, e):
        self.t = int(self.slider.value)
        self.update_frame()

    def toggle_play(self, e):
        self.playing = not self.playing
        self.play_btn.text = "⏸" if self.playing else "▶"
        self.update()
        if self.playing:
            asyncio.create_task(self.autoplay())

    async def autoplay(self):
        while self.playing:
            self.t = (self.t + 1) % len(self.stack)
            self.slider.value = self.t
            self.update_frame()
            await asyncio.sleep(0.05)

    def close(self, e):
        self.visible = False
        self.update()



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

    # --- GLOBAL DATA STORES ---
    # Stores for multiple files:
    # {filename: {"tracks": df, "stack": np_array, "masks": list_of_masks}}
    GLOBAL_RESULTS = {}
    # Full dataframe of all results (for global plots)
    GLOBAL_FULL_DF = None

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
            if name.lower() in ("outputs", "__pycache__", "models", ".venv"):
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
    page.overlay.append(fp)
    page.overlay.append(save_file_picker)
    picked_path_text = ft.Text("Aucun fichier chargé.")
    preview_file_path: str | None = None

    auto_preview_check = ft.Switch(label="Aperçu automatique (ON/OFF)", value=True)

    # Option "Mode Rapide" pour l'analyse
    fast_mode_check = ft.Switch(label="⚡ Mode Rapide (1 frame sur 2)", value=False)

    pick_btn = ft.ElevatedButton(
        "Charger une image (TIF / PNG / JPG)",
        icon="folder_open",
        on_click=lambda _: fp.pick_files(
            allow_multiple=False,
            allowed_extensions=["tif", "tiff", "png", "jpg", "jpeg", "bmp"]
        ),
    )

    def on_file_picked(e: ft.FilePickerResultEvent):
        nonlocal preview_file_path
        log_view.controls.clear()
        if e.files:
            preview_file_path = e.files[0].path or e.files[0].name
            picked_path_text.value = f"Fichier sélectionné : {os.path.basename(preview_file_path)}"
            page.update()
            update_preview_final()
        else:
            preview_file_path = None
            picked_path_text.value = "Aucun fichier chargé."
            preview_img.src = None
            preview_img.src_base64 = None
            page.update()

    fp.on_result = on_file_picked

    _preview_timer = None
    def debounce_preview():
        global _preview_timer
        if not auto_preview_check.value:
            return
        if _preview_timer and _preview_timer.is_alive():
            _preview_timer.cancel()
        _preview_timer = threading.Timer(0.6, update_preview_final)
        _preview_timer.start()

    def update_preview_final():
        if not preview_file_path or not os.path.exists(preview_file_path):
            return

        try:
            log_view.controls.clear()
            log(f"Nouvelle preview — σ={sigma_slider.value:.2f}, min={min_size_slider.value:.0f}, CLAHE={clahe_slider.value:.3f}")
            spinner.visible = True
            page.update()

            img_b64 = generate_overlay_preview(
                preview_file_path,
                sigma=float(sigma_slider.value),
                min_size=int(min_size_slider.value),
                clahe_clip=float(clahe_slider.value),
                logger=log,
                deep_enhance=bool(deep_check.value),
            )

            if img_b64:
                preview_img.src_base64 = img_b64
                preview_img.src = None 
                preview_img.update()
                log("Preview IA mise à jour.")
            else:
                log("Aucune image générée.")
        except Exception as ex:
            log(f"Erreur de preview: {ex}")
        finally:
            spinner.visible = False
            page.update()

    deep_check = ft.Switch(
        label="Amélioration profonde (Deep Enhance)",
        value=False,
        on_change=lambda e: debounce_preview()
    )

    import threading
    _preview_timer = None

    def debounce_preview():
        nonlocal _preview_timer
        try:
            if _preview_timer and _preview_timer.is_alive():
                _preview_timer.cancel()
        except Exception:
            pass
        _preview_timer = threading.Timer(0.8, update_preview_final)
        _preview_timer.start()

    def update_preview_final():
        if not preview_file_path or not os.path.exists(preview_file_path):
            return
        try:
            spinner.visible = True
            log_view.controls.clear()
            log(f"Nouvelle preview — σ={sigma_slider.value:.2f}, min={min_size_slider.value:.0f}, CLAHE={clahe_slider.value:.3f}")
            page.update()

            img_b64 = generate_overlay_preview(
                preview_file_path,
                sigma=float(sigma_slider.value),
                min_size=int(min_size_slider.value),
                clahe_clip=float(clahe_slider.value),
                logger=log,
                deep_enhance=bool(deep_check.value),
            )

            if img_b64:
                preview_img.src_base64 = img_b64
                preview_img.update()
                log("Preview mise à jour.")
            else:
                log("Aucune image générée.")
        except Exception as ex:
            log(f"Erreur de preview : {ex}")
        finally:
            spinner.visible = False
            page.update()

    def on_slider_change(e):
        if isinstance(e.control.value, float):
            e.control.label = f"{e.control.value:.3f}" if e.control.max < 10 else f"{e.control.value:.2f}"
        else:
            e.control.label = f"{int(e.control.value)}"
        e.control.update()
        debounce_preview()

    sigma_slider = ft.Slider(
        label=f"{ADV_PARAMS['sigma']:.2f}",
        min=0.5, max=3.0, divisions=25,
        value=ADV_PARAMS["sigma"],
        on_change=on_slider_change
    )
    min_size_slider = ft.Slider(
        label=f"{ADV_PARAMS['min_size']:.0f}",
        min=5, max=500, divisions=99,
        value=float(ADV_PARAMS["min_size"]),
        on_change=on_slider_change
    )
    clahe_slider = ft.Slider(
        label=f"{ADV_PARAMS['clahe_clip']:.3f}",
        min=0.005, max=0.08, divisions=15,
        value=ADV_PARAMS["clahe_clip"],
        on_change=on_slider_change
    )

    deep_check = ft.Switch(
        label="Amélioration profonde (Deep Enhance)",
        value=False,
        on_change=lambda e: debounce_preview()
    )

    auto_preview_switch = ft.Switch(
        label="Aperçu automatique (ON/OFF)",
        value=True,
        on_change=lambda e: log("🧩 Aperçu automatique : " + ("activé" if e.control.value else "désactivé"))
    )

    def on_slider_change(e):
        e.control.label = f"{e.control.value:.3f}" if isinstance(e.control.value, float) else f"{int(e.control.value)}"
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

    min_text = ft.Text("Taille minimale (px)", width=180)
    min_size_slider = ft.Slider(
        min=5, max=500, divisions=99,
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

    min_text = ft.Text("Taille minimale (px)", width=180)
    min_size_slider = ft.Slider(
        min=5, max=500, divisions=99,
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

    auto_preview_switch = ft.Switch(
        label="Aperçu automatique (ON/OFF)",
        value=True,
        on_change=lambda e: log("🧩 Aperçu automatique : " + ("activé" if e.control.value else "désactivé"))
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

    # ----------------------------------------------------------------------
    # TRACKING TAB container (vide au départ)
    # ----------------------------------------------------------------------
    tracking_tab = ft.Column([], scroll=ft.ScrollMode.AUTO)

    # ----------------------------------------------------------------------
    # BOUTON LANCER ANALYSE (icône Flet compatible)
    # ----------------------------------------------------------------------
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



    # ----------------------------------------------------------------------
    # UTILITAIRES UI (définis hors de run_analysis pour éviter scope issues)
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # LANCEMENT DE L'ANALYSE
    # ----------------------------------------------------------------------
    def run_analysis(e=None):
        nonlocal GLOBAL_RESULTS, GLOBAL_FULL_DF
        root = data_root.value.strip()

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
                    metrics, tracks, current_stack, current_masks = process_file(
                        pth, seg_method=method, logger=log, debug=True, fast_mode=use_fast
                    )

                    # Store in global cache
                    GLOBAL_RESULTS[fname] = {
                        "tracks": tracks,
                        "stack": current_stack,
                        "masks": current_masks,
                        "condition": cond
                    }

                    # Append metrcis for plotting
                    metrics["condition"] = cond
                    all_results_meta.append(metrics)

                    log(f"OK: {fname}")

                except Exception as ex:
                    log(f"Erreur {os.path.basename(pth)} : {ex}")

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

        # ----------------------------------------------------------------------
        # TRACKING — SETUP UI with File Selector
        # ----------------------------------------------------------------------
        tracking_tab = tabs.tabs[2].content
        tracking_tab.controls.clear()

        file_options = [ft.dropdown.Option(k) for k in GLOBAL_RESULTS.keys()]
        if not file_options:
            tracking_tab.controls.append(ft.Text("Aucune donnée valide.", color="red300"))
            tracking_tab.update()
            return

        # Default to first file
        current_file_key = file_options[0].key

        file_dropdown = ft.Dropdown(
            label="Fichier à visualiser",
            options=file_options,
            value=current_file_key,
            width=400,
        )

        # Containers for dynamic content
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

            # --- TABLE GENERATION (Reused logic) ---
            columns = [
                "track_id", "t", "x", "y",
                "speed_um_s", "cum_distance_um",
                "area_um2", "circularity", "eccentricity",
                "aspect_ratio", "solidity", "feret_max_um", "angle_deg", "straightness",
            ]

            # Pagination logic closure
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

            # Now render rows after table is on page
            update_table_rows()

            # --- ACTIONS ---
            def open_viewer_click(e):
                if stack is None: return
                viewer = TrackViewer(stack, df, masks=masks)
                page.overlay.append(viewer)
                page.update()

            def export_current_csv(e):
                save_file_picker.data = df # Pass df to picker via data or closure
                save_file_picker.save_file(
                    dialog_title=f"Sauvegarder {file_key}.csv",
                    file_name=f"{file_key}_tracking.csv",
                    allowed_extensions=["csv"]
                )

            # HACK: We attach the current DF to the picker in a closure or external ref
            # Better: define specific handler
            save_file_picker.on_result = lambda e: save_csv_callback(e, df)

            actions_container.controls.extend([
                ft.ElevatedButton("👁 Visualiser Track (TrackMate)", on_click=open_viewer_click, icon="remove_red_eye"),
                ft.ElevatedButton("Export CSV (Fichier)", icon="download", on_click=export_current_csv, bgcolor="blue700", color="white")
            ])
            actions_container.update()

        def on_file_change(e):
            refresh_view(file_dropdown.value)

        file_dropdown.on_change = on_file_change

        # --- GLOBAL ACTIONS DEFINITION (moved before render) ---
        out_dir = os.path.join(root, "outputs")
        # Plot curves using GLOBAL_FULL_DF (all files metrics)
        saved_plots = plot_curves(GLOBAL_FULL_DF, out_dir=out_dir)

        def show_graphs(e):
            dlg_content = ft.Column(scroll=ft.ScrollMode.AUTO, height=600)
            if not saved_plots:
                dlg_content.controls.append(ft.Text("Aucun graphique généré."))
            else:
                for p_path in saved_plots:
                    if os.path.exists(p_path):
                         with open(p_path, "rb") as f:
                             b64 = base64.b64encode(f.read()).decode("utf-8")
                         dlg_content.controls.append(
                             ft.Image(src_base64=b64, width=600, fit=ft.ImageFit.CONTAIN)
                         )
            dlg = ft.AlertDialog(
                title=ft.Text("Courbes Globales (Prolifération & Survie)"),
                content=dlg_content,
                actions=[ft.TextButton("Fermer", on_click=lambda _: page.close_dialog())],
            )
            page.dialog = dlg
            dlg.open = True
            page.update()

        # Comparison Dialog
        def show_comparison(e):
            files = list(GLOBAL_RESULTS.keys())
            if len(files) < 2:
                page.snack_bar = ft.SnackBar(ft.Text("Il faut au moins 2 fichiers pour comparer."))
                page.snack_bar.open = True
                page.update()
                return

            dd1 = ft.Dropdown(options=[ft.dropdown.Option(f) for f in files], value=files[0], label="Fichier A", expand=True)
            dd2 = ft.Dropdown(options=[ft.dropdown.Option(f) for f in files], value=files[1] if len(files)>1 else files[0], label="Fichier B", expand=True)
            result_area = ft.Column()

            def compute_comp(e):
                f1, f2 = dd1.value, dd2.value
                if not f1 or not f2: return

                df1 = GLOBAL_RESULTS[f1]["tracks"]
                df2 = GLOBAL_RESULTS[f2]["tracks"]

                # Simple Stats
                # Avoid crash if empty
                if df1.empty or df2.empty:
                    result_area.controls = [ft.Text("Données vides pour l'un des fichiers.")]
                    result_area.update()
                    return

                s1_speed = df1["speed_um_s"].mean()
                s2_speed = df2["speed_um_s"].mean()
                count1 = df1["track_id"].nunique()
                count2 = df2["track_id"].nunique()

                # Last time point survival (approx)
                t_max1 = df1["t"].max()
                t_max2 = df2["t"].max()

                txt = (
                    f"**Comparaison**\n\n"
                    f"**{f1}**:\n"
                    f" - Cellules (total tracks): {count1}\n"
                    f" - Vitesse moy: {s1_speed:.4f} µm/s\n"
                    f" - Durée (frames): {t_max1}\n\n"
                    f"**{f2}**:\n"
                    f" - Cellules (total tracks): {count2}\n"
                    f" - Vitesse moy: {s2_speed:.4f} µm/s\n"
                    f" - Durée (frames): {t_max2}\n\n"
                    f"**Delta** (A - B):\n"
                    f" - Speed: {s1_speed - s2_speed:.4f}\n"
                    f" - Cells: {count1 - count2}"
                )
                result_area.controls = [ft.Markdown(txt)]
                result_area.update()

            dlg = ft.AlertDialog(
                title=ft.Text("Comparer deux résultats"),
                content=ft.Container(
                    content=ft.Column([
                        ft.Row([dd1, dd2]),
                        ft.ElevatedButton("Comparer", on_click=compute_comp),
                        ft.Divider(),
                        result_area
                    ], height=400, width=500),
                    padding=10
                ),
                actions=[ft.TextButton("Fermer", on_click=lambda _: page.close_dialog())]
            )
            page.dialog = dlg
            dlg.open = True
            page.update()

        global_actions = ft.Row([
             ft.ElevatedButton("📈 Visualiser les Courbes Globales", on_click=show_graphs, icon="show_chart"),
             ft.ElevatedButton("⚖️ Comparer Résultats", on_click=show_comparison, icon="compare_arrows")
        ])

        tracking_tab.controls.append(ft.Container(content=file_dropdown, padding=10))
        tracking_tab.controls.append(ft.Container(content=global_actions, padding=5))
        tracking_tab.controls.append(actions_container)
        tracking_tab.controls.append(ft.Divider())
        tracking_tab.controls.append(table_container)
        tracking_tab.update()

        # Initial render - Now safe because containers are on the page
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