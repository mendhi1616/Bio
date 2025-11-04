
import os, glob
import streamlit as st
import pandas as pd
from pipeline import process_condition, compute_group_stats, plot_curves, CELLPOSE_OK

st.set_page_config(page_title="Live-Cell AI (Prolifération & Survie)", layout="wide")

st.title("🧬 Live-Cell — Prolifération & Survie (No‑Code)")
st.markdown("Cette application analyse automatiquement des images de microscopie (Hoechst) pour **compter**, **suivre** et **comparer** les cellules entre conditions (ex: WT vs KO), et calcule des **statistiques** (t-test, AUC).")

with st.expander("📂 Étape 1 — Choisir le dossier de données", expanded=True):
    data_root = st.text_input("Chemin du dossier `data` (ex: ./data)", value="data")
    st.info("Structure attendue : `data/condition/*.tif` (ex. `data/wild-type/wild-type_01.tif`, `data/KO/KO_01.tif`).")

    detected = []
    if os.path.isdir(data_root):
        for name in sorted(os.listdir(data_root)):
            p = os.path.join(data_root, name)
            if os.path.isdir(p) and glob.glob(os.path.join(p, '*.tif')):
                detected.append(name)

    conditions = st.multiselect("Conditions trouvées", detected, default=detected[:2])
    st.caption("Astuce : commencez par 2 conditions (ex. `wild-type`, `KO`).")

with st.expander("🧪 Étape 2 — Paramètres d'analyse", expanded=True):
    seg_method = st.selectbox("Méthode de segmentation", ["auto"] + (["cellpose"] if CELLPOSE_OK else []))
    st.caption("`auto` = pipeline classique (Otsu + morpho + watershed). `cellpose` = modèle noyaux si installé.")

run = st.button("▶️ Lancer l'analyse", type="primary")

if run:
    if not os.path.isdir(data_root) or len(conditions) < 1:
        st.error("Veuillez indiquer un dossier valide et au moins une condition.")
    else:
        all_results = []
        progress = st.progress(0.0, text="Analyse en cours...")
        for i, cond in enumerate(conditions):
            st.write(f"Traitement de **{cond}**...")
            res = process_condition(data_root, cond, seg_method=seg_method)
            all_results.append(res)
            progress.progress((i+1)/max(1,len(conditions)), text=f"Analyse {cond} terminée")
        if all_results:
            results = pd.concat(all_results, ignore_index=True)
            st.success("Analyse terminée ✅")
            st.subheader("📊 Résultats (table)")
            st.dataframe(results)

            auc_df, ttest = compute_group_stats(results)
            st.subheader("📐 AUC & Statistiques")
            st.write("**AUC par fichier** (prolifération & survie) :")
            st.dataframe(auc_df)

            if ttest:
                c1, c2 = ttest["conditions"]
                st.markdown(f"**T‑tests (Welch)** entre `{c1}` et `{c2}` :")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("p (n_cells au dernier temps)", f"{ttest['last_timepoint_ncells_p']:.3g}")
                with col2:
                    st.metric("p (AUC prolif)", f"{ttest['AUC_prolif_p']:.3g}")
                with col3:
                    st.metric("p (AUC survie)", f"{ttest['AUC_survival_p']:.3g}")
                st.caption("Convention : p < 0.05 → différence statistiquement significative.")
            else:
                st.info("Ajoutez deux conditions pour obtenir les tests statistiques.")

            st.subheader("📈 Courbes moyennes par condition")
            figs = plot_curves(results)
            for key, fig in figs.items():
                st.pyplot(fig, clear_figure=True)

            os.makedirs("outputs", exist_ok=True)
            results_path = os.path.join("outputs", "metrics_all.csv")
            auc_path = os.path.join("outputs", "AUC_by_file.csv")
            results.to_csv(results_path, index=False)
            auc_df.to_csv(auc_path, index=False)

            st.download_button("⬇️ Télécharger metrics_all.csv", data=results.to_csv(index=False).encode("utf-8"), file_name="metrics_all.csv", mime="text/csv")
            st.download_button("⬇️ Télécharger AUC_by_file.csv", data=auc_df.to_csv(index=False).encode("utf-8"), file_name="AUC_by_file.csv", mime="text/csv")

            st.success("Exports prêts (CSV). Vous pouvez les intégrer à vos figures/rapports.")
        else:
            st.warning("Aucun résultat.")
