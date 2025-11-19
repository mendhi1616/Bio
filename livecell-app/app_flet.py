import asyncio
import warnings
import atexit
import gc
import torch
import sys
import verify_env
verify_env.check_and_install()
import os, glob, threading
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
    def __init__(self, stack, tracking_df):
        super().__init__(
            bgcolor=ft.colors.BLACK,
            padding=10,
            expand=True
        )

        self.stack = stack
        self.df = tracking_df
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
            style=ft.ButtonStyle(color=ft.colors.WHITE)
        )

        self.close_btn = ft.TextButton(
            "✖",
            on_click=self.close,
            style=ft.ButtonStyle(color=ft.colors.RED_400)
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
        
        import cv2
        df_t = self.df[self.df["t"] == self.t]
        
        for _, row in df_t.iterrows():
            x, y = int(row["x"]), int(row["y"])
            tid = int(row["track_id"])
            
            # cercle sur la cellule
            cv2.circle(img, (x, y), 6, (0,255,0), 2)
            
            # ID
            cv2.putText(
                img,
                str(tid),
                (x+8, y-8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1,
                cv2.LINE_AA
            )
        
        _, buf = cv2.imencode(".png", img)
        img_base64 = base64.b64encode(buf).decode()

        self.img_display.src_base64 = img_base64
        self.update()

    def on_seek(self, e):
        self.t = int(self.slider.value)
        self.update_frame()

    def toggle_play(self, e):
        self.playing = not self.playing
        self.play_btn.text = "⏸" if self.playing else "▶"
        self.update()
        if self.playing:
            self.autoplay()

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
    def on_folder_picked(e: ft.FilePickerResultEvent):
        pass
    pick_folder_btn = ft.ElevatedButton(
        "Choisir un dossier",
        icon=ft.Icon(name="folder_open"),
        on_click=on_folder_picked
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
    page.overlay.append(fp)
    picked_path_text = ft.Text("Aucun fichier chargé.")
    preview_file_path: str | None = None

    auto_preview_check = ft.Switch(label="Aperçu automatique (ON/OFF)", value=True)

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
            sigma_slider, min_size_slider, clahe_slider, deep_check
        ]:
            ctrl.disabled = not enabled
        page.update()



    # ----------------------------------------------------------------------
    # LANCEMENT DE L'ANALYSE
    # ----------------------------------------------------------------------
    def run_analysis(e=None):
        root = data_root.value.strip()

        all_results = []
        all_tracks = []

        # We'll store the last processed stack here for visualization
        last_stack = None

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
        tabs.tabs[0].content.controls[:] = [ft.Text("Analyse en cours..."), progress]
        log_view.controls.clear()

        method = seg_method.value
        if deep_check.value and method not in ("cellpose",):
            method = "gam_gpu"

        log(f"Analyse — méthode: {method} — deep={deep_check.value}")

        total = sum(len(v) for v in selection.values())
        done = 0

        for cond, files in selection.items():
            log(f"▶️ {cond}: {len(files)} fichier(s)")
            for pth in files:
                status.value = f"Traitement: {cond} / {os.path.basename(pth)}"
                page.update()

                try:
                    # Now extracting stack as well
                    metrics, tracks, current_stack = process_file(
                        pth, seg_method=method, logger=log, debug=True
                    )

                    last_stack = current_stack
                    all_tracks.append(tracks)
                    metrics["condition"] = cond
                    all_results.append(metrics)

                    log(f"OK: {os.path.basename(pth)}")

                except Exception as ex:
                    log(f"Erreur {os.path.basename(pth)} : {ex}")

                done += 1
                progress.value = done / max(1, total)
                page.update()

        toggle_ui(True)
        status.value = "Analyse terminée"
        page.update()

        if not all_results:
            log("Aucun résultat.")
            return

        results = pd.concat(all_results, ignore_index=True)
        log(f"Résultats : {len(results)} lignes")

        # ----------------------------------------------------------------------
        # TRACKING — AFFICHAGE TABLEAU
        # ----------------------------------------------------------------------
        tracking_tab = tabs.tabs[2].content
        tracking_tab.controls.clear()

        if not all_tracks or len(all_tracks) == 0:
            log("[WARN] Aucun tracking détecté.")
            tracking_tab.controls.append(
                ft.Text("Aucune cellule suivie dans cette vidéo.", color=ft.colors.RED_300)
            )
            tracking_tab.update()
            return

        tracking_df = pd.concat(all_tracks, ignore_index=True).fillna(0)

        if tracking_df.shape[0] == 0:
            log("[WARN] Tracking vide.")
            tracking_tab.controls.append(
                ft.Text("Aucune cellule suivie.", color=ft.colors.RED_300)
            )
            tracking_tab.update()
            return

        # Colonnes affichées
        columns = [
            "track_id", "t", "x", "y",
            "speed_um_s", "cum_distance_um",
            "area_um2", "circularity", "eccentricity",
            "aspect_ratio", "solidity",
            "feret_max_um", "angle_deg", "straightness",
        ]

        # --- HEADER FIXE ---
        header_table = ft.DataTable(
            columns=[ft.DataColumn(ft.Text(col)) for col in columns],
            rows=[],
            horizontal_margin=10,
            column_spacing=20,
        )


        # --- TABLE DES LIGNES ---
        rows_table = ft.DataTable(
            columns=[ft.DataColumn(ft.Container(width=0)) for col in columns],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                str(r[col])
                                if isinstance(r[col], str)
                                else f"{r[col]:.2f}"
                            )
                        )
                        for col in columns
                    ]
                )
                for _, r in tracking_df.iterrows()
            ],
            horizontal_margin=10,
            column_spacing=20,
        )

        # --- SCROLL FIX compatible Flet < 0.19 ---
        scroll_area = ft.Column(
            controls=[rows_table],
            expand=True,
            scroll=ft.ScrollMode.ALWAYS,
        )

        # Injection dans l’onglet Tracking
        tracking_tab.controls.append(header_table)
        tracking_tab.controls.append(scroll_area)
        tracking_tab.update()

        def open_track_viewer(e):
            if last_stack is None:
                log("[ERREUR] Pas d'image disponible pour la visualisation.")
                return
            viewer = TrackViewer(last_stack, tracking_df)
            page.overlay.append(viewer)
            page.update()

        view_btn = ft.TextButton("👁 Visualiser Track (Dernier Fichier)", on_click=open_track_viewer)
        tracking_tab.controls.append(view_btn)

        # Export CSV
        def export_csv(e):
            out_path = os.path.join(root, "tracking_results.csv")
            tracking_df.to_csv(out_path, index=False)
            log(f"[EXPORT] CSV sauvegardé : {out_path}")

        export_button = ft.ElevatedButton(
            "Export CSV",
            icon=ft.Icon(name="download"),
            on_click=export_csv,
            bgcolor=ft.colors.BLUE_700,
            color=ft.colors.WHITE,
        )


        tracking_tab.controls.append(ft.Container(height=10))
        tracking_tab.controls.append(export_button)
        tracking_tab.update()




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
        ft.Divider(),
        ft.Text("Conditions expérimentales :"),
        conditions_panel,
        ft.Divider(),
        ft.Row([run_btn, status]),
        tabs,
    )

if __name__ == "__main__":
    ft.app(target=main)