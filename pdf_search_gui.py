#!/usr/bin/env python3

#################################################################
# Recherche intelligente dans les fichiers PDF.
#
# Fonctionnalités :
# • Recherche sémantique avec FAISS
# • Réponses générées avec Qwen
# • Réponses générées avec ChatGPT
# • Recherche textuelle
# • Résumés générés avec ChatGPT
# • Consultation des documents PDF
# • Recherche de doublons
#
# Développé en Python avec Tkinter
# en Vibe Coding avec l'IA.
#
# Auteur : Eric LAMETTE © 2026
# Contact : ericlmt@gmail.com
#
# ────────────────────────────────────
# LICENCE GNU GPL v3
# ────────────────────────────────────
#
# This program is free software: you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software
# Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General
# Public License along with this program.
# If not, see https://www.gnu.org/licenses/.
#
#################################################################

import os
import faiss
import torch
import pickle
import numpy as np
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox, Toplevel, Label, simpledialog
import time
from glob import glob
from pathlib import Path
import json
from tqdm import tqdm
import threading
from openai import OpenAI
import re
from collections import defaultdict
from collections import Counter
from datetime import datetime
import sys
import subprocess
import unicodedata
from functools import lru_cache
from queue import Empty, SimpleQueue
import tempfile
import webbrowser
from textwrap import dedent
from huggingface_hub import snapshot_download
import traceback
import logging


#https://docs.python.org/fr/3.15/library/index.html

#
# Type de document	                    Chunk	Overlap
# FAQ, définitions	                    256	    32
# Documents PDF généralistes	        512	    64
# Documents techniques détaillés	    1024	128
# Textes longs nécessitant du contexte	2048	256
# Répertoire de l'application

APP_NAME = "Recherche PDF"
APP_VERSION = "11/2026"

# ====== CONFIG ======
OPENAI_API_KEY = ""
MODEL_NAME = ""
OPENAI_API_ACCESS = False
client = None
 
MAX_CHARS_PER_CHUNK = 40000  # taille max de texte envoyé par appel pour résumer
MAX_LENGTH_TOKENIZER = 1024 #256
OVERLAP_TOKENS = 128

MAX_FILES = 20              # PDFs affichés
MAX_EXTRACTS_PER_FILE = 30 #15     # nombre d'extraits concaténés par PDF
MAX_CHARS_PER_FILE = 1500    # longueur max affichée par PDF
MAX_FILES_TO_SUMMERIZE = 10

MAX_QWEN_CONTEXT_DOCS = 4
MAX_QWEN_CONTEXT_LIMIT = 3000
MAX_QWEN_EXTRACTS_PER_FILE = 2
MAX_QWEN_EXTRACT_CHARS = 1200
MAX_QWEN_SUMMARY_CHARS = 500

MODEL_QWEN = "Qwen3-0.6B" #Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen3-4B
MODEL_QWEN_EMBEDDING = "Qwen3-Embedding-0.6B" #Qwen/Qwen3-Embedding-4B
device = "cuda" if torch.cuda.is_available() else "cpu"

#########################################################
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# Dossier principal des modèles
MODELS_DIR = Path(BASE_DIR) / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Dossier Models : {MODELS_DIR}")

MODEL_QWEN_EMBEDDING_PATH = BASE_DIR / MODELS_DIR / MODEL_QWEN_EMBEDDING
MODEL_QWEN_PATH = BASE_DIR / MODELS_DIR / MODEL_QWEN

# Dossier principal des données
PDFS_DIR = Path(BASE_DIR) / "PDFS"
PDFS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Dossier PDFS : {PDFS_DIR}")

DATAS_DIR = Path(BASE_DIR) / "datas"
DATAS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Dossier datas : {DATAS_DIR}")

TO_READ_FILE = BASE_DIR / DATAS_DIR / "to_read.jsonl"
SUMMARY_FILE = BASE_DIR / DATAS_DIR / "summarization_data.jsonl"

INDEX_PATH = BASE_DIR / DATAS_DIR / "index_qwen.idx"
EMBEDDINGS_PATH = BASE_DIR / DATAS_DIR / "chunk_embeddings.npy"
TEXTS_PATH = BASE_DIR / DATAS_DIR / "texts.pkl"
BAD_WORDS = BASE_DIR / DATAS_DIR / "bad_words.txt"
INDEXEDFILES_PATH = BASE_DIR / DATAS_DIR / "indexed_files.txt"

# Variables globales
embed_model = None
embed_tokenizer = None
summ_model = None
summ_tokenizer = None
models_ready = False
progress_running = False

# Log des erreurs
logging.basicConfig(
    filename=str(DATAS_DIR / "erreurs.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)


# Vérifie les fichiers d'un modèle Hugging Face et complète le dossier local si nécessaire.
def ensure_model(model_id, model_path):
    """Télécharge le modèle si nécessaire."""
    print(f"Vérification du modèle : {model_id}")
    results_text.insert(tk.END, f"⏳ Vérification du modèle {model_id}\n")
    results_text.see(tk.END)

    # Hugging Face vérifie les fichiers et complète
    # les téléchargements manquants ou interrompus.
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_path),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.tiktoken",
            "*.py",
        ],
    )

    print(f"Modèle disponible : {model_path}")
    return model_path


# Prépare les fichiers des deux modèles Qwen utilisés par l'application.
def ensure_qwen_models():
    """Prépare les deux modèles Qwen."""
    ensure_model(
        "Qwen/Qwen3-Embedding-0.6B",
        MODEL_QWEN_EMBEDDING_PATH
    )

    ensure_model(
        "Qwen/Qwen3-0.6B",
        MODEL_QWEN_PATH
    )


# Démarre l'animation de la barre de progression si elle est arrêtée.
def start_progress():
    global progress_running

    if progress_running:
        return

    progress.stop()
    progress.config(mode="indeterminate", maximum=100, value=0)
    progress_running = True
    progress.start(25)


# Arrête l'animation et remet la barre de progression en mode déterminé.
def stop_progress():
    global progress_running

    if not progress_running:
        return

    progress.stop()
    progress.configure(mode="determinate", maximum=100, value=0)
    progress_running = False


# Charge les modèles et l'index FAISS en arrière-plan, puis signale la fin de l'initialisation.
def load_models_worker():
    """Charge les modèles dans un thread secondaire."""
    global embed_model, embed_tokenizer, summ_model, summ_tokenizer
    global models_ready

    models_ready = False
    model_load_error[0] = None
    model_loading_step = ["Initialisation..."]

    try:
        results_text.insert(tk.END, f"⚠️ Cette opération peut prendre plusieurs minutes selon la puissance de votre ordinateur.\n\n")
        # 1. Vérifier et télécharger les modèles
        model_loading_step[0] = "Vérification des modèles Qwen..."
        ensure_qwen_models()

        # 2. Charger Qwen Embedding
        model_loading_step[0] = "Chargement de Qwen Embedding..."
        print(f"🔄 Chargement du modèle Qwen3-Embedding... {MODEL_QWEN_EMBEDDING}")
        print(f"🔄 {MODEL_QWEN_EMBEDDING_PATH}")
        results_text.insert(tk.END, f"🔄 Chargement du modèle Qwen3-Embedding... {MODEL_QWEN_EMBEDDING}\n")
        results_text.see(tk.END)

        embed_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_QWEN_EMBEDDING_PATH), local_files_only=True)
        embed_model = AutoModel.from_pretrained(str(MODEL_QWEN_EMBEDDING_PATH), local_files_only=True)
        embed_model.eval()
        embed_model = embed_model.to(device)
        torch.set_grad_enabled(False)  # pas d'apprentissage

        # 3. Charger Qwen génératif
        model_loading_step[0] = "Chargement de Qwen génératif..."
        print(f"🔄 Chargement du modèle Qwen3... {MODEL_QWEN}")
        print(f"🔄 {MODEL_QWEN_PATH}")
        results_text.insert(tk.END, f"🔄 Chargement du modèle Qwen3... {MODEL_QWEN}\n")
        results_text.see(tk.END)

        summ_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_QWEN_PATH), local_files_only=True)
        summ_model = AutoModelForCausalLM.from_pretrained(str(MODEL_QWEN_PATH), local_files_only=True)
        summ_model.eval()
        summ_model = summ_model.to(device)

        # 4. Charger FAISS
        model_loading_step[0] = "Chargement de l'index FAISS..."
        load_faiss()

        models_ready = True
        model_loading_step[0] = "RecherchePDF est prêt."

    except Exception as e:
        logging.exception(f"Erreur : {e}")
        model_load_error[0] = traceback.format_exc()
        print("ERREUR INITIALISATION :")
        print(model_load_error[0])

    finally:
        model_loading_finished.set()


# Surveille la fin du chargement et affiche son résultat dans Tkinter.
def check_models_loading():
    """Vérifie le chargement depuis le thread Tkinter."""
    if model_loading_finished.is_set():
        stop_progress()

        if models_ready:
            status_label.config(
                text=f"✅ Modèles Qwen chargés. {APP_NAME} est prêt."
            )
            btn_answer.config(state="normal")

        else:
            status_label.config(text="❌ Erreur de chargement des modèles")
            messagebox.showerror(
                "Erreur Qwen",
                model_load_error[0] or "Erreur inconnue",
                parent=root
            )
        return

    root.after(200, check_models_loading)


# Lance le thread d'initialisation après l'affichage de la fenêtre.
def start_models_loading():
    """Démarre le chargement après l'affichage de la fenêtre."""
    status_label.config(text="⏳ Chargement des modèles Qwen...")
    start_progress()

    threading.Thread(
        target=load_models_worker,
        daemon=True
    ).start()

    root.after(200, check_models_loading)


# Charge une valeur OpenAI ou propose de la renseigner. Une réponse vide ignore la configuration.
def load_or_ask_config(filename, title, message, secret=False):
    filepath = Path(filename)
    if filepath.is_file():
        value = filepath.read_text(encoding="utf-8").strip()
        if value:
            return value

    answer = simpledialog.askstring(
        title,
        message,
        show="*" if secret else None,
        parent=root
    )
    value = (answer or "").strip()
    if value:
        filepath.write_text(value, encoding="utf-8")
    return value


# Écrit les vecteurs dans un fichier temporaire avant de remplacer le fichier NumPy.
def save_embeddings_safe(vecs, path=EMBEDDINGS_PATH):
    temp_path = path + ".tmp"

    try:
        with open(temp_path, "wb") as f:
            np.save(f, np.asarray(vecs, dtype="float32"))

        os.replace(temp_path, path)

        print("✅ Embeddings sauvegardés.")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Charge l'index et les données associées, vérifie leur cohérence et initialise les mots à ignorer.
def load_faiss():
    global embed_model, index, indexed_files
    global chunk_embeddings, embedding_text_map
    global FRENCH_STOPWORDS

    # Chargement FAISS
    if embed_model is None:
        raise RuntimeError(
            "Impossible de charger FAISS : "
            "Qwen Embedding n'est pas initialisé."
        )

    print("Chargement de l'index FAISS...")
    # 1 Charger index (sûr)
    dimension = embed_model.config.hidden_size
    index, chunk_embeddings, embedding_text_map = (
        load_or_build_index(dimension)
    )
    # 2 Nettoyage OPTIONNEL
    cleanup_index_files()
    # 3 Vérification finale
    verify_index_integrity(
        embedding_text_map,
        chunk_embeddings,
        index
    )
    print("Nombre de vecteurs FAISS :", index.ntotal)
    print("Nombre de métadonnées :", len(embedding_text_map))
    # 4 Chargement des fichiers indexés
    indexed_files = fix_indexed_files(embedding_text_map)
    # 5 Chargement des bad_word
    FRENCH_STOPWORDS = load_stopwords(BAD_WORDS)
    print("Stopwords chargés :", len(FRENCH_STOPWORDS))
    print("Index FAISS chargé :", index.ntotal, "vecteurs")


# Charge les passages et leurs embeddings, puis réutilise ou reconstruit l'index FAISS.
def load_or_build_index(
    dim,
    index_path=INDEX_PATH,
    embeddings_path=EMBEDDINGS_PATH,
    texts_path=TEXTS_PATH,
):
    """
    Charge un index FAISS existant s'il est cohérent,
    sinon le reconstruit proprement à partir des embeddings.
    Retourne : index, chunk_embeddings, embedding_text_map
    """
    debut_load = time.time()

    # -----------------------------
    # Vérification de l'Index
    # -----------------------------
    if os.path.exists(INDEX_PATH):
        try:
            old_index = faiss.read_index(str(INDEX_PATH))
            print("Vecteurs FAISS :", old_index.ntotal)
            print("Dimension FAISS :", old_index.d)
            print("Type FAISS :", type(old_index))

        except Exception as e:
            logging.exception(f"Erreur : {e}")
            print(traceback.format_exc())
            print("Erreur lecture index FAISS :", e)


    index = None
    chunk_embeddings = []
    embedding_text_map = []

    # 1️⃣ Charger embedding_text_map (obligatoire)
    if os.path.exists(texts_path):
        with open(texts_path, "rb") as f:
            embedding_text_map = pickle.load(f)
            print("Type embedding_text_map :", type(embedding_text_map))
            print("Nombre d'éléments :", len(embedding_text_map))
            if len(embedding_text_map) > 0:
                if isinstance(embedding_text_map, dict):
                    key = next(iter(embedding_text_map))
                    print("Première clé :", repr(key))
                    print("Première valeur :", repr(embedding_text_map[key])[:250])
                else:
                    print("Premier élément :", repr(embedding_text_map[0])[:250])
    else:
        print("⚠️ texts.pkl absent → index vide")
        return faiss.IndexFlatIP(dim), [], []

    # 2️⃣ Charger embeddings
    if os.path.exists(embeddings_path):
        try:
            chunk_embeddings = np.load(embeddings_path)
        except Exception as e:
            logging.exception(f"Erreur : {e}")
            print(traceback.format_exc())
            print(f"⚠️ Erreur chargement embeddings : {e}")
            chunk_embeddings = []
    else:
        print("⚠️ chunk_embeddings.npy absent → rebuild requis")
        chunk_embeddings = []

    # 3️⃣ Vérifier cohérence embeddings ↔ textes
    if len(chunk_embeddings) != len(embedding_text_map):
        print("⚠️ Incohérence embeddings / textes → rebuild index")
        rebuild = True
    else:
        rebuild = False

    # 4️⃣ Charger FAISS si possible
    if os.path.exists(index_path) and not rebuild:
        try:
            index = faiss.read_index(str(index_path))

            # Vérifier le type d'index
            if not isinstance(index, faiss.IndexFlatIP):
                print("⚠️ Mauvais type d'index FAISS → rebuild")
                rebuild = True

            elif index.ntotal != len(chunk_embeddings):
                print("⚠️ FAISS ntotal incohérent → rebuild")
                rebuild = True

            else:
                print(f"✅ Index FAISS chargé ({index.ntotal} vecteurs)")
                return index, list(chunk_embeddings), embedding_text_map

        except Exception as e:
            logging.exception(f"Erreur : {e}")
            print(f"⚠️ Erreur lecture FAISS : {e}")
            rebuild = True

    # 5️⃣ Reconstruction propre
    # Vérifier si les embeddings sont disponibles
    if len(chunk_embeddings) == 0:
        print("⚠️ Aucun embedding disponible.")
        print("Recalcul des embeddings nécessaire.")
        return faiss.IndexFlatIP(dim), [], embedding_text_map

    print("🔄 Reconstruction de l’index FAISS...")
    if rebuild:
        index = faiss.IndexFlatIP(dim)

    if len(chunk_embeddings) > 0:
        vecs = np.array(chunk_embeddings, dtype="float32")
        if vecs.ndim != 2 or vecs.shape[1] != dim:
            raise ValueError(
                f"Dimensions incorrectes : {vecs.shape}"
            )
        faiss.normalize_L2(vecs)
        index.add(vecs)

        save_embeddings_safe(vecs, embeddings_path)
        faiss.write_index(index, str(index_path))

    fin_load = time.time()
    print(f"Temps d'exécution : {fin_load - debut_load:.2f} secondes")
    return index, list(chunk_embeddings), embedding_text_map


# Retire les passages des PDF disparus et sauvegarde l'index filtré avec ses données associées.
def cleanup_index_files():
    global embedding_text_map, chunk_embeddings, index

    print("🧹 Nettoyage des fichiers supprimés...")

    # Vérifier la cohérence avant nettoyage
    if len(embedding_text_map) != len(chunk_embeddings):
        print("⚠️ Nettoyage interrompu :")
        print("Incohérence entre textes et embeddings.")
        return

    # 1️⃣ indices à conserver
    keep_indices = [
        i for i, (fp, _) in enumerate(embedding_text_map)
        if os.path.exists(fp)
    ]

    removed = len(embedding_text_map) - len(keep_indices)

    if removed == 0:
        print("✅ Aucun fichier supprimé détecté")
        return

    # 2️⃣ filtrer embedding_text_map
    new_text_map = [
        embedding_text_map[i] for i in keep_indices
    ]

    # 3️⃣ filtrer chunk_embeddings
    new_embeddings = [
        chunk_embeddings[i] for i in keep_indices
    ]

    # Reconstruire FAISS
    if new_embeddings:
        vecs = np.vstack(new_embeddings).astype("float32")
        faiss.normalize_L2(vecs)

        dim = vecs.shape[1]
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(vecs)
    else:
        new_index = faiss.IndexFlatIP(index.d)
        vecs = np.empty((0, index.d), dtype="float32")

    # Sauvegarder les embeddings
    save_embeddings_safe(vecs, EMBEDDINGS_PATH)

    # Sauvegarder FAISS
    faiss.write_index(new_index, INDEX_PATH)


    # 4️⃣ reconstruire FAISS
    #dim = chunk_embeddings[0].shape[0]
    #index = faiss.IndexFlatIP(dim)

    #vecs = np.vstack(chunk_embeddings).astype("float32")
    #faiss.normalize_L2(vecs)
    #index.add(vecs)

    # 5️⃣ sauvegardes
    #np.save(EMBEDDINGS_PATH, vecs)
    #save_embeddings_safe(vecs, EMBEDDINGS_PATH)
    #faiss.write_index(index, INDEX_PATH)

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(new_text_map, f)

    # Mettre à jour les variables globales
    embedding_text_map = new_text_map
    chunk_embeddings = list(vecs)
    index = new_index

    print(f"🧹 Nettoyage terminé : {removed} chunks supprimés")


# Contrôle la concordance de l'index, des vecteurs et des passages, puis renvoie les anomalies.
def verify_index_integrity(
    embedding_text_map,
    chunk_embeddings,
    index,
    indexed_files_path=INDEXEDFILES_PATH
):
    errors = []
    warnings = []

    # 1️⃣ Vérification globale des tailles
    if len(embedding_text_map) != len(chunk_embeddings):
        errors.append(
            f"Incohérence chunks ({len(embedding_text_map)}) "
            f"≠ embeddings ({len(chunk_embeddings)})"
        )

    if index.ntotal != len(chunk_embeddings):
        errors.append(
            f"Incohérence FAISS ({index.ntotal}) "
            f"≠ embeddings ({len(chunk_embeddings)})"
        )

    # 2️⃣ Vérification des fichiers référencés
    file_to_chunks = {}
    for fp, _ in embedding_text_map:
        file_to_chunks.setdefault(fp, 0)
        file_to_chunks[fp] += 1

    for fp, count in file_to_chunks.items():
        if not os.path.exists(fp):
            errors.append(f"Fichier manquant : {fp}")
        elif count == 0:
            warnings.append(f"Aucun chunk pour : {fp}")

    # 3️⃣ Vérification indexed_files.txt (optionnelle)
    if os.path.exists(indexed_files_path):
        with open(indexed_files_path, "r", encoding="utf-8") as f:
            indexed = {line.strip() for line in f if line.strip()}

        indexed_map = set(file_to_chunks.keys())

        missing_in_index = indexed - indexed_map
        missing_in_file = indexed_map - indexed

        if missing_in_index:
            warnings.append(
                f"{len(missing_in_index)} fichiers listés mais absents de l’index"
            )

        if missing_in_file:
            warnings.append(
                f"{len(missing_in_file)} fichiers indexés mais absents de indexed_files.txt"
            )

    # 4️⃣ Résumé
    print("🔍 Vérification de l’index")
    if errors:
        print("❌ ERREURS CRITIQUES :")
        for e in errors:
            print("  -", e)
    else:
        print("✅ Aucune erreur critique")

    if warnings:
        print("⚠️ Avertissements :")
        for w in warnings:
            print("  -", w)

    return errors, warnings


# Aligne le registre des PDF indexés sur les chemins référencés par les passages.
def fix_indexed_files(
    embedding_text_map,
    indexed_files_path=INDEXEDFILES_PATH
):
    # Source de vérité
    indexed_from_map = set(fp for fp, _ in embedding_text_map)

    # Charger l’existant
    if os.path.exists(indexed_files_path):
        with open(indexed_files_path, "r", encoding="utf-8") as f:
            indexed_from_file = {line.strip() for line in f if line.strip()}
    else:
        indexed_from_file = set()

    # Différences
    missing_in_index = indexed_from_file - indexed_from_map
    missing_in_file = indexed_from_map - indexed_from_file

    # Correction automatique
    corrected = indexed_from_file - missing_in_index
    corrected |= missing_in_file

    # Réécriture propre
    with open(indexed_files_path, "w", encoding="utf-8") as f:
        for fp in sorted(corrected):
            f.write(fp + "\n")

    # Logs
    if missing_in_index:
        print(f"🧹 Supprimés de indexed_files.txt : {len(missing_in_index)}")

    if missing_in_file:
        print(f"➕ Ajoutés à indexed_files.txt : {len(missing_in_file)}")

    if not missing_in_index and not missing_in_file:
        print("✅ indexed_files.txt déjà cohérent")

    return corrected


# Charge la liste des mots à ignorer lors de la sélection des mots-clés.
def load_stopwords(filepath=BAD_WORDS):
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier stopwords introuvable : {filepath}")
        return set()

    with open(filepath, "r", encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }


# -----------------------------
# Fonction GPT
# -----------------------------
# Envoie les messages système et utilisateur au modèle OpenAI configuré et renvoie sa réponse.
def call_gpt(system_prompt, user_prompt, max_output_tokens=2000, temperature=0.3):
    if not OPENAI_API_ACCESS:
        raise RuntimeError("Les fonctions OpenAI nécessitent une clé API et un modèle.")

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_output_tokens=max_output_tokens
        #,temperature=temperature,
    )
    return response.output_text.strip()


# -----------------------------
# Fonction texte
# -----------------------------
# Copie le contenu de la zone de résumé dans le presse-papier.
def copier_resume():
    texte = summary_text.get("1.0", tk.END).strip()
    if not texte:
        return

    root.clipboard_clear()
    root.clipboard_append(texte)
    root.update()  # nécessaire sous Windows pour mettre à jour le presse-papier

    # petit message de confirmation dans la console ou dans l’UI
    print("📋 Résultat copié dans le presse-papier")
    status_label.config(text="Résultat copié dans le presse-papier")
    root.after(2000, lambda: status_label.config(text=""))


# Place le texte de la zone de résultats dans un fichier temporaire et demande son impression à Windows.
def print_resume():
    print("📋 Imprimer le résultat")
    status_label.config(text="Imprimer le résultat...")
    root.after(2000, lambda: status_label.config(text=""))
    texte = results_text.get("1.0", "end-1c").strip()

    if not texte:
        messagebox.showwarning("Impression", "Aucun texte à imprimer.")
        return

    try:
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            encoding="utf-8",
            delete=False
        ) as f:
            f.write(texte)
            fichier_temp = f.name

        # Envoyer à l'application d'impression par défaut
        os.startfile(fichier_temp, "print")

    except Exception as e:
        logging.exception(f"Erreur : {e}")
        print(traceback.format_exc())
        messagebox.showerror(
            "Erreur d'impression",
            str(e)
        )


# Raccourcit les noms de fichiers longs pour l'affichage en conservant leur extension.
def shorten_filename(filename, max_len=35):
    name, ext = os.path.splitext(filename)

    if len(name) <= max_len:
        return filename

    return name[:max_len] + "..." + ext


# Remplace les suites d'espaces et de retours à la ligne par un espace, puis ajoute un saut de ligne final.
def trim_empty_lines(content):
    cleaned = " ".join(content.split())
    return cleaned + "\n"


# Retrouve le premier résumé enregistré dans le fichier JSONL à partir du nom du PDF.
def read_dataset_summary(fname, datasetfile=SUMMARY_FILE):
    target_name = os.path.basename(fname)
    try:
        with open(datasetfile, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record.get("filename") == target_name:
                    return record.get("summary")
    except FileNotFoundError:
        print(f"⚠️ Fichier dataset introuvable : {datasetfile}")
        return None

    return None


# Aplatit les espaces et découpe le texte en blocs limités en nombre de caractères.
def split_text_by_chars(text, max_chars=MAX_CHARS_PER_CHUNK):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end

    return chunks

# Lit les chemins des PDF présents dans la liste « À lire » et les renvoie sous forme d'ensemble.
def read_to_read_set():
    files = set()
    if not os.path.exists(TO_READ_FILE):
        return files

    with open(TO_READ_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                record = json.loads(line)
                files.add(record["filepath"])
            except Exception as error:
                logging.exception(f"Erreur : {error}")
                print(traceback.format_exc())
                pass
    return files


# Ajoute le chemin, le nom et la date d'ajout d'un PDF dans la liste « À lire ».
def add_to_read(filepath):
    record = {
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "date": datetime.now().isoformat()
    }

    with open(TO_READ_FILE, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Fonction Qwen embedding
# -----------------------------
# Calcule un vecteur Qwen pour un texte en moyennant les représentations de ses tokens.
def get_embedding(text: str) -> np.ndarray:
    inputs = embed_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH_TOKENIZER,
        padding=True
    ).to(embed_model.device)

    with torch.no_grad():
        outputs = embed_model(**inputs)
        # pool simple : moyenne sur la séquence
        emb = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy().astype("float32")

    return emb


# -----------------------------
# Fonction résumé
# -----------------------------
# Indique si le chemin donné existe sur le disque.
def file_exists(filepath):
    return os.path.exists(filepath)


# Génère localement avec Qwen un résumé concis du texte fourni.
def resumer_texte(texte):
    prompt = f"Fais un résumé clair et concis du texte suivant :\n\n{texte}\n\nRésumé :"
    inputs = summ_tokenizer(prompt, return_tensors="pt").to(summ_model.device)
    with torch.no_grad():
        outputs = summ_model.generate(**inputs, max_new_tokens=2048)
    return summ_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Résumé :")[-1].strip()


# Prépare une consigne de synthèse et demande à OpenAI de résumer un bloc de texte.
def resumer_gpt_request_chunk(chunk_text: str):
    """
    Appelle GPT-4o pour générer des résumés
    """
    system_prompt = (
        "Tu es un générateur automatique de résumé de texte en français.\n"
        "À partir d'un TEXTE FOURNI, tu construis un résumé de haute qualité de style académique.\n"
        "\n"
        "Règles très importantes :\n"
        "- Tu dois te baser UNIQUEMENT sur le texte fourni. N'ajoute aucune connaissance extérieure.\n"
        "- Couvre les points importants du texte, évite de répéter plusieurs fois exactement la même information.\n"
        "- Respecte l’argumentation de l’auteur.\n"
        "- Identifie la problématique centrale.\n"
        "- Mentionne les concepts clés et cadres théoriques.\n"
        "- Ignore les exemples secondaires et détails techniques.\n"
        "- Rédige de manière neutre et académique.\n"
        "- Structure le résumé en points organisés.\n"
    )

    user_prompt = f"""
Texte :
\"\"\"{chunk_text}\"\"\"

Consignes :
- Génère un résumé pertinent basées sur ce texte.
- N'invente aucune information qui ne figure pas clairement dans le texte.
"""

    summary = call_gpt(system_prompt, user_prompt)

    if summary is None or len(summary) < 10 :
        summary = "Pas de résumé produit"
    return summary


# Demande à OpenAI de produire une synthèse finale à partir du texte fourni.
def resumer_gpt_request_text(text: str):
    system_prompt = (
        "Tu es un assistant académique spécialisé en sciences humaines.\n"

        "Voici plusieurs résumés partiels d’un même article scientifique.\n"

        "Ta tâche :\n"
        "- Fusionner ces résumés en un résumé final cohérent.\n"
        "- Ne pas répéter les idées.\n"
        "- Structurer le résumé par thèmes ou sections logiques.\n"
        "- Mettre en évidence :\n"
        "  • la problématique\n"
        "  • les hypothèses ou thèses\n"
        "  • l’approche théorique\n"
        "  • les résultats ou conclusions\n"
        "- Utiliser un style académique clair.\n"
        "- Termine toujours le résumé par une phrase complète.\n"
        "- Si tu arrives à la limite, conclus proprement.\n"
    )

    user_prompt = f"""
RÉSUMÉS :
\"\"\"{text}\"\"\"

Consignes :
- Génère un résumé pertinent basées sur ce texte.
- N'invente aucune information qui ne figure pas clairement dans le texte.
"""

    summary = call_gpt(system_prompt, user_prompt)

    if summary is None or len(summary) < 10 :
        summary = "Pas de résumé produit"
    return summary


# Résume les blocs d'un texte avec OpenAI, puis fusionne leurs résumés dans une synthèse.
def resumer_texte_gpt(text):
    chunks = split_text_by_chars(text, max_chars=MAX_CHARS_PER_CHUNK)
    print(f"📄 Article découpé en {len(chunks)} blocs")
    partial_summaries = []
    root.after(0, lambda:progress.config(maximum=len(chunks)))
    for i, chunk in enumerate(chunks, 1):
        print(f"✍️ Résumé du bloc {i}/{len(chunks)}")
        summary = resumer_gpt_request_chunk(chunk)
        partial_summaries.append(summary)
        root.after(0, lambda:progress.config(value=i))

    partial_summaries = "\n\n".join(partial_summaries)
    print("🧠 Génération du résumé final")
    final_summary = resumer_gpt_request_text(partial_summaries)

    return final_summary


# -----------------------------
# Fonctions PDF + découpage
# -----------------------------
# Extrait le texte de chaque page lisible par pypdf et le rassemble dans une chaîne.
def extract_text_from_pdf(path):
    try:
        reader = PdfReader(str(path), strict=False)

        texts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

        full_text = "\n".join(texts)
        print(f"Longueur du PDF : {len(full_text)}")
        return full_text

    except (PdfReadError, PdfStreamError, OSError, ValueError) as e:
        logging.warning("PDF illisible : %s - %s", pdf_file, e)
        print(f"⚠️ PDF ignoré : {pdf_file}")
        print(f"   Raison : {e}")

    except Exception as e:
        logging.exception(f"Erreur : {e}")
        print(traceback.format_exc())
        print(f"⚠️ Problème PDF : {path} → {e}")
        return ""


# Ancienne méthode de découpage qui accumule des fragments selon une estimation en tokens.
def chunk_text_old(text, tokenizer=embed_tokenizer, max_tokens=MAX_LENGTH_TOKENIZER): #256
    import re

    text = " ".join(text.split())
    paragraphs = text.split("\n")

    chunks = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sentence in sentences:
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_tokens + len(tokens) > max_tokens:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = len(tokens)
            else:
                current.append(sentence)
                current_tokens += len(tokens)

    if current:
        chunks.append(" ".join(current))

    return chunks


# Découpe le texte en passages limités en tokens, avec un chevauchement entre passages.
def chunk_text(
    text,
    tokenizer=embed_tokenizer,
    max_tokens=MAX_LENGTH_TOKENIZER,
    overlap_tokens=OVERLAP_TOKENS
):
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Overlap invalide")

    if not text or not text.strip():
        return []

    # Nettoyage sans supprimer les paragraphes
    paragraphs = [
        " ".join(p.split())
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    # Découpage en phrases
    sentences = []
    for para in paragraphs:
        sentences.extend(
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", para)
            if s.strip()
        )

    # Assemblage avec limite stricte et overlap
    chunks = []
    current = ""

    # Mesure la longueur d'un fragment avec le tokenizer utilisé pour l'indexation.
    def count_tokens(s):
        return len(tokenizer.encode(
            s, add_special_tokens=False
        ))

    # Ajoute un passage à la liste en appliquant la limite de tokens.
    def add_chunk(value):
        if value and (not chunks or chunks[-1] != value):
            chunks.append(value)

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()

        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        if current:
            add_chunk(current)

            # Conserver une partie du contexte précédent
            ids = tokenizer.encode(
                current, add_special_tokens=False
            )
            overlap_ids = ids[-overlap_tokens:] if overlap_tokens else []
            current = tokenizer.decode(
                overlap_ids, skip_special_tokens=True
            ).strip()

        candidate = f"{current} {sentence}".strip()

        if count_tokens(candidate) <= max_tokens:
            current = candidate
            continue

        # Phrase trop longue : découpage par tokens
        if current:
            add_chunk(current)

        ids = tokenizer.encode(
            sentence, add_special_tokens=False
        )

        step = max_tokens - overlap_tokens

        for start in range(0, len(ids), step):
            part = tokenizer.decode(
                ids[start:start + max_tokens],
                skip_special_tokens=True
            ).strip()

            if part:
                # Vérification après décodage
                if count_tokens(part) <= max_tokens:
                    add_chunk(part)
                else:
                    # Découpage de sécurité
                    words = part.split()
                    fragment = ""
                    for word in words:
                        test = f"{fragment} {word}".strip()
                        if count_tokens(test) > max_tokens:
                            if fragment:
                                add_chunk(fragment)
                            fragment = word
                        else:
                            fragment = test
                    if fragment:
                        add_chunk(fragment)

        current = ""

    if current:
        add_chunk(current)

    return chunks


# Vectorise les passages par lots avec Qwen et rassemble les vecteurs dans une matrice NumPy.
def embed_chunks(chunks, batch_size=16):
    all_vecs = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = embed_tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH_TOKENIZER, #512
            padding=True
        ).to(embed_model.device)

        with torch.no_grad():
            outputs = embed_model(**inputs)
            batch_embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
            all_vecs.append(batch_embs)

    if not all_vecs:
        return None

    return np.vstack(all_vecs)


# -----------------------------
# Fonction recherche
# -----------------------------
# Agrège les vecteurs des passages de chaque PDF en un vecteur moyen normalisé.
def build_document_embeddings(chunk_embeddings, embedding_text_map):
    doc_vecs = defaultdict(list)

    for emb, (filepath, _) in zip(chunk_embeddings, embedding_text_map):
        if emb is None:
            continue
        doc_vecs[filepath].append(emb)

    doc_embeddings = {}

    for filepath, vectors in doc_vecs.items():
        mat = np.vstack(vectors).astype("float32")
        mean_vec = np.mean(mat, axis=0)

        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm

        doc_embeddings[filepath] = mean_vec

    return doc_embeddings


# Calcule le nombre de groupes à demander selon la taille du corpus et la valeur souhaitée.
def safe_k(num_points, desired_k):
    return max(2, min(desired_k, num_points // 30))


# Recherche les voisins de chaque PDF et renvoie les paires uniques au-dessus du seuil.
def find_similar_documents(doc_embeddings, top_k=10, min_sim=0.88):
    files = list(doc_embeddings.keys())
    vectors = np.stack([doc_embeddings[f] for f in files]).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine
    index.add(vectors)

    D, I = index.search(vectors, top_k + 1)

    seen = set()
    results = []

    for i, f1 in enumerate(files):
        for sim, j in zip(D[i][1:], I[i][1:]):
            if sim < min_sim:
                continue

            f2 = files[j]
            pair = tuple(sorted((f1, f2)))
            if pair in seen:
                continue
            seen.add(pair)

            results.append((sim, f1, f2))

    # cosine similarity décroissante
    results.sort(key=lambda x: -x[0])
    return results


# Applique le k-means de FAISS et associe chaque groupe aux chemins de ses PDF.
def cluster_documents(doc_embeddings, k=6):
    files = list(doc_embeddings.keys())

    print("Nombre de fichiers :", len(files))
    #print("Nombre d'embeddings :", len(doc_embeddings))
    #print("Fichiers :", files)
    #print("Clés des embeddings :", list(doc_embeddings.keys())[:10])

    # Conserver uniquement les fichiers ayant un embedding
    files = [
        f for f in files
        if f in doc_embeddings
        and doc_embeddings[f] is not None
    ]

    if not files:
        print("Aucun document avec embedding valide.")
        return {}

    # Vérifier les dimensions des vecteurs
    dimensions = {
        np.asarray(doc_embeddings[f]).shape
        for f in files
    }

    if len(dimensions) != 1:
        print(f"Dimensions incompatibles : {dimensions}")
        return {}

    # Construire la matrice
    vectors = np.stack([doc_embeddings[f] for f in files]).astype("float32")
    faiss.normalize_L2(vectors)

    kmeans = faiss.Kmeans(vectors.shape[1], k, niter=30, verbose=False)
    kmeans.train(vectors)

    _, labels = kmeans.index.search(vectors, 1)

    clusters = defaultdict(list)
    for fp, label in zip(files, labels.flatten()):
        clusters[label].append(fp)

    return clusters


# Signale les paires de PDF dont les vecteurs sont assez proches pour être des quasi-doublons.
def detect_duplicates(doc_embeddings, similarity_threshold=0.95):
    """
    Détecte les documents quasi-identiques.
    Retourne une liste de paires (sim, doc1, doc2).
    """
    files = list(doc_embeddings.keys())
    vectors = np.stack([doc_embeddings[f] for f in files]).astype("float32")

    # cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    D, I = index.search(vectors, 2)  # top-1 voisin (hors soi-même)

    duplicates = []
    seen = set()

    for i, f1 in enumerate(files):
        sim = D[i][1]
        j = I[i][1]
        f2 = files[j]

        if sim >= similarity_threshold:
            pair = tuple(sorted((f1, f2)))
            if pair not in seen:
                seen.add(pair)
                duplicates.append((sim, f1, f2))

    # tri par similarité décroissante
    duplicates.sort(key=lambda x: -x[0])
    return duplicates


# Choisit pour chaque groupe le PDF le plus proche de son vecteur moyen.
def get_cluster_representatives(clusters, doc_embeddings):
    representatives = {}

    for cluster_id, files in clusters.items():
        vectors = np.stack([doc_embeddings[f] for f in files])
        center = np.mean(vectors, axis=0)

        sims = vectors @ center
        best_idx = np.argmax(sims)

        representatives[cluster_id] = files[best_idx]

    return representatives


# Classe les mots des PDF sélectionnés par fréquence après filtrage des mots à ignorer.
def extract_keywords(files, embedding_text_map, top_k=10):
    text = " ".join(
        chunk
        for fp, chunk in embedding_text_map
        if fp in files
    )

    words = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", text.lower())
    words = [
        w for w in words
        if w not in FRENCH_STOPWORDS
    ]

    common = Counter(words).most_common(top_k)
    return [w for w, _ in common]


# Associe à chaque groupe une étiquette de mots-clés et le nom de son PDF représentatif.
def label_clusters(clusters, doc_embeddings, embedding_text_map):
    if not clusters:
        return {}

    labels = {}

    reps = get_cluster_representatives(clusters, doc_embeddings)

    for cluster_id, files in clusters.items():
        keywords = extract_keywords(files, embedding_text_map)
        rep_file = os.path.basename(reps[cluster_id])

        label = f"{', '.join(keywords[:6])}"
        labels[cluster_id] = {
            "label": label,
            "representative": rep_file
        }

    return labels


# -----------------------------
# Indexation avec persistance
# -----------------------------
# Affiche une confirmation dont la réponse devient « Oui » à l'expiration du délai.
def askyesno_timeout(parent, title, message, timeout=5):
    result = {"value": True}

    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    remaining = timeout
    timer_id = None

    ttk.Label(dialog, text=message, padding=15).pack()

    countdown = ttk.Label(dialog)
    countdown.pack(pady=5)

    buttons = ttk.Frame(dialog)
    buttons.pack(pady=10)

    # Enregistre la réponse, annule le compte à rebours et ferme la boîte de dialogue.
    def close(value):
        nonlocal timer_id

        result["value"] = value

        if timer_id is not None:
            dialog.after_cancel(timer_id)
            timer_id = None

        dialog.destroy()

    # Actualise le compte à rebours et valide la réponse à la fin du délai.
    def tick():
        nonlocal remaining, timer_id

        countdown.config(
            text=f"Continuation automatique dans {remaining} s"
        )

        if remaining == 0:
            timer_id = None
            close(True)
        else:
            remaining -= 1
            timer_id = dialog.after(1000, tick)

    ttk.Button(
        buttons,
        text="Oui",
        command=lambda: close(True)
    ).pack(side="left", padx=10)

    ttk.Button(
        buttons,
        text="Non",
        command=lambda: close(False)
    ).pack(side="left", padx=10)

    dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))

    tick()
    parent.wait_window(dialog)

    return result["value"]


# Traite les PDF du dossier par lots et demande confirmation entre deux lots.
def index_pdfs_worker(folder):
    fichiers_pdf = sorted(
        str(Path(os.path.join(folder, f)).resolve())
        for f in os.listdir(folder)
        if f.endswith(".pdf")
    )

    total = len(fichiers_pdf)
    if total == 0:
        root.after(0, lambda: status_label.config(text="Aucun PDF trouvé."))
        root.after(0, lambda: messagebox.showinfo("Indexation", "Aucun PDF trouvé."))
        return

    taille_lot = MAX_FILES_TO_SUMMERIZE
    lots = [fichiers_pdf[i:i + taille_lot] for i in range(0, total, taille_lot)]

    root.after(0, lambda:summary_text.insert(tk.END, f"📂 Indexation par lot de {taille_lot}\n"))
    root.after(0, lambda:summary_text.see(tk.END))

    deja = 0

    for lot in lots:
        output = indexer_par_lots(lot)
        deja += len(lot)

        # Affiche le bilan du lot traité dans le thread principal de Tkinter.
        def update_ui(output=output, deja=deja):
            results_text.insert(tk.END, output)
            results_text.see(tk.END)
            status_label.config(text=f"⏳ Indexation en cours... Fichiers traités : {deja}/{total}")

        root.after(0, update_ui)

        if deja < total:
            answered = threading.Event()
            continuer = False

            # Demande à Tkinter si l'indexation doit continuer et transmet la réponse au thread d'indexation.
            def ask_to_continue():
                nonlocal continuer
                try:
                    continuer = askyesno_timeout(
                        root,
                        "Continuer ?",
                        f"{deja}/{total} fichiers traités.\nVoulez-vous continuer ?",
                        timeout=5
                    )
                finally:
                    answered.set()

            root.after(0, ask_to_continue)
            answered.wait()
            if not continuer:
                break

    # Affiche le bilan de l'indexation et remet la barre de progression à zéro.
    def finish_ui():
        status_label.config(text="✅ Indexation terminée")
        results_text.insert(
            tk.END,
            f"\n✅ Indexation terminée ({len(embedding_text_map)} segments au total).\n"
        )
        results_text.see(tk.END)
        progress.config(value=0)

    root.after(0, finish_ui)


# Demande le dossier à indexer et lance le traitement dans un thread dédié.
def index_pdfs():
    messagebox.showwarning("Indexation des PDF",  "Cette opération peut prendre du temps (de quelques minutes à plusieurs heures), selon le nombre de fichiers PDF à indexer.")
    
    folder = filedialog.askdirectory(title="Sélectionner un dossier contenant des PDF")
    if not folder:
        return

    root.after(0, lambda:status_label.config(text=""))
    clear_results()
    root.after(0, lambda:summary_text.delete(1.0, tk.END))

    root.after(0, lambda:results_text.insert(tk.END, f"\n📂 Indexation depuis : {folder}\n"))
    root.after(0, lambda:results_text.see(tk.END))

    root.after(0, lambda:summary_text.insert(tk.END, f"📂 Indexation depuis : {folder}\n"))
    root.after(0, lambda:summary_text.see(tk.END))

    root.after(0, lambda:status_label.config(text="⏳ Indexation en cours..."))
    root.after(0, lambda:progress.config(value=0))

    set_search_buttons(tk.DISABLED)

    # Exécute l'indexation et programme la réactivation des boutons à sa fin.
    def worker():
        try:
            index_pdfs_worker(folder)
        except Exception as error:
            logging.exception(f"Erreur : {error}")
            print(traceback.format_exc())
            root.after(0, status_label.config, {"text": "❌ Erreur d'indexation"})
            root.after(0, messagebox.showerror, "Erreur d'indexation", str(error))
        finally:
            root.after(0, set_search_buttons, tk.NORMAL)

    try:
        threading.Thread(target=worker, name="indexation-PDF", daemon=True).start()
    except RuntimeError as error:
        set_search_buttons(tk.NORMAL)
        status_label.config(text="❌ Erreur d'indexation")
        messagebox.showerror("Erreur d'indexation", str(error))


# Extrait et vectorise les PDF encore absents de l'index, puis sauvegarde les données du lot.
def indexer_par_lots(lot_paths):
    output = ""
    nouveaux_vecs = []
    nouveaux_texts = []
    total = len(lot_paths)
    index_debut = 0
    index_fin = total

    for path in lot_paths:
        filename = os.path.basename(path)

        if path in indexed_files:
            output += f"⏩ Déjà indexé : {filename}\n"
            index_fin = index_fin - 1
            continue

        try:
            root.after(0, lambda:progress.config(maximum=index_fin))
            root.after(0, lambda:progress.config(value=index_debut))
            root.after(0, lambda:results_text.insert(tk.END, f"📄 Traitement : {filename}\n"))
            root.after(0, lambda:results_text.see(tk.END))
            root.after(0, lambda:status_label.config(text=f"⏳ Indexation en cours... Fichiers traités : {index_debut}/{total}"))

            text = extract_text_from_pdf(path)
            chunks = chunk_text(text, embed_tokenizer)

            if not chunks:
                output += f"⚠️ Aucun texte extrait pour {filename}\n"
                continue

            short_name = shorten_filename(filename)
            msg = f"📄 Traitement : {short_name} -- {len(chunks)} chunks à indexer."
            print(msg)
            root.after(0, lambda:summary_text.insert(tk.END, msg + "\n"))
            root.after(0, lambda:summary_text.see(tk.END))

            # Embeddings pour tous les chunks de ce PDF
            embs = embed_chunks(chunks, batch_size=16)  # ou 8 si VRAM limitée
            if embs is None:
                output += f"⚠️ Pas d'embeddings pour {filename}\n"
                continue

            embs_np = np.asarray(embs, dtype="float32")
            if embs_np.ndim != 2 or len(embs_np) != len(chunks):
                raise ValueError("Nombre ou dimensions d'embeddings incorrects")

            faiss.normalize_L2(embs_np)

            for emb, chunk in zip(embs_np, chunks):
                chunk_embeddings.append(emb.copy())
                nouveaux_vecs.append(emb.copy())
                nouveaux_texts.append((path, chunk))

            # On ajoute au buffer global
            indexed_files.add(path)

            output += f"✅ Fichier {filename} indexé ({len(chunks)} chunks).\n"
            index_debut += 1
            root.after(0, lambda:progress.config(value=index_debut))

        except Exception as e:
            logging.exception(f"Erreur : {e}")
            print(traceback.format_exc())
            output += f"❌ Erreur avec {filename} : {e}\n"

    # ➜ Une seule fois : ajout dans FAISS + sauvegardes
    if nouveaux_vecs:
        all_vecs_np = np.vstack(nouveaux_vecs).astype("float32")

        # ✅ normalisation cosine
        faiss.normalize_L2(all_vecs_np)
        index.add(all_vecs_np)

        # update la map texte
        embedding_text_map.extend(nouveaux_texts)

        # sauvegarde disque
        #np.save(EMBEDDINGS_PATH, np.vstack(chunk_embeddings))
        save_embeddings_safe(np.vstack(chunk_embeddings).astype("float32"), EMBEDDINGS_PATH)
        faiss.write_index(index, INDEX_PATH)
        with open(TEXTS_PATH, "wb") as f:
            pickle.dump(embedding_text_map, f)

        # sauvegarde des fichiers déjà indexés
        with open(INDEXEDFILES_PATH, "a", encoding="utf-8") as f:
            for path in lot_paths:
                if path in indexed_files:
                    f.write(path + "\n")

    print(f"✅ {len(embedding_text_map)} chunks indexés.")

    return output


# -----------------------------
# Fonction de recherche
# -----------------------------
# Interroge FAISS avec le vecteur de la question et renvoie les scores et indices des passages.
def get_relevant_documents(query, index, embedding_text_map, top_k=30):
    vec = get_embedding(query)
    if vec is None:
        return []

    num_vectors = index.ntotal
    TOP_K = min(40, max(20, num_vectors // 100))
    if top_k > TOP_K:
        TOP_K = top_k

    D, I = index.search(np.expand_dims(vec, axis=0), TOP_K)
    results = sorted(zip(D[0], I[0]), key=lambda x: x[0], reverse=True)

    return results


# Calcule les vecteurs de synthèse pour les seuls PDF de la sélection.
def build_document_embeddings_subset(
    chunk_embeddings,
    embedding_text_map,
    selected_files
):
    from collections import defaultdict
    doc_vecs = defaultdict(list)

    for emb, (fp, _) in zip(chunk_embeddings, embedding_text_map):
        if fp in selected_files:
            doc_vecs[fp].append(emb)

    doc_embeddings = {
        fp: np.mean(vectors, axis=0)
        for fp, vectors in doc_vecs.items()
        if len(vectors) > 0
    }

    # normalisation cosine
    for fp in doc_embeddings:
        doc_embeddings[fp] /= np.linalg.norm(doc_embeddings[fp])

    return doc_embeddings


# Calcule les vecteurs documentaires et renvoie les paires similaires et les quasi-doublons.
def compute_similars_and_duplicates(
    chunk_embeddings,
    embedding_text_map,
    min_sim=0.90,
    dup_sim=0.95,
    max_pairs=10
):
    # Embeddings par document
    doc_embeddings = build_document_embeddings(
        chunk_embeddings,
        embedding_text_map
    )

    if len(doc_embeddings) < 2:
        return [], []

    # Documents similaires
    similar_docs = find_similar_documents(
        doc_embeddings,
        min_sim=min_sim
    )[:max_pairs]

    # Quasi-doublons
    duplicates = detect_duplicates(
        doc_embeddings,
        similarity_threshold=dup_sim
    )[:max_pairs]

    return similar_docs, duplicates


# Actualise les commandes de recherche et d'indexation en tenant compte de la configuration OpenAI.
def set_search_buttons(state):
    for button in (btn_search, btn_answer, btn_answer_brut, btn_similar, btn_index):
        button.config(state=state)
    btn_answer_gpt.config(state=state if OPENAI_API_ACCESS else tk.DISABLED)


# Lance un thread par recherche et transmet sa progression et son résultat via une file.
def run_search(work, display, update_progress=None):
    """Exécute une recherche sans appeler Tkinter depuis son thread."""
    updates = SimpleQueue()
    set_search_buttons(tk.DISABLED)

    start_progress()

    # Exécute le calcul et dépose le résultat ou l'exception dans la file.
    def worker():
        try:
            if update_progress is None:
                result = work()
            else:
                result = work(lambda *values: updates.put((update_progress, values)))
            updates.put((None, (result, None)))
        except Exception as error:
            logging.exception(f"Erreur : {error}")
            print(traceback.format_exc())
            updates.put((None, (None, error)))

    # Affiche dans Tkinter l'erreur remontée par la recherche.
    def show_error(error):
        status_label.config(text="❌ Erreur de recherche")
        messagebox.showerror("Erreur de recherche", str(error))

    # Récupère les messages sur le thread Tkinter et réactive les commandes à la fin du traitement.
    def poll():
        while True:
            try:
                callback, values = updates.get_nowait()
            except Empty:
                root.after(50, poll)
                return
            if callback is None:
                break
            callback(*values)

        result, error = values
        try:
            if error is not None:
                show_error(error)
            elif result is None:
                status_label.config(text="")
                messagebox.showwarning(
                    "Aucun contexte",
                    "Aucun extrait ou résumé pertinent n'a été trouvé pour cette question."
                )
            else:
                display(result)
        except Exception as error:
            logging.exception(f"Erreur : {error}")
            print(traceback.format_exc())
            show_error(error)
        finally:
            stop_progress()
            set_search_buttons(tk.NORMAL)

    try:
        threading.Thread(target=worker, name="recherche-PDF", daemon=True).start()
    except RuntimeError as error:
        set_search_buttons(tk.NORMAL)
        show_error(error)
        return

    root.after(50, poll)


# Lance la détection de documents similaires et de quasi-doublons en arrière-plan.
def show_similars_and_duplicates():
    summary_text.delete("1.0", tk.END)
    status_label.config(text="🔎 Analyse des documents similaires et doublons...")
    run_search(
        lambda: compute_similars_and_duplicates(chunk_embeddings, embedding_text_map),
        display_similars_and_duplicates
    )


# Affiche les paires détectées et leurs scores dans la zone de résumé.
def display_similars_and_duplicates(result):
    similar_docs, duplicates = result
    # 🔗 Documents similaires
    summary_text.insert(tk.END, "🔗 Documents similaires :\n\n")
    if similar_docs:
        for sim, f1, f2 in similar_docs:
            msg = (
                f"📄 {shorten_filename(os.path.basename(f1))} "
                f"↔ {shorten_filename(os.path.basename(f2))} "
                f"| sim={sim:.3f}\n"
            )
            summary_text.insert(tk.END, msg)
    else:
        summary_text.insert(tk.END, "Aucun document similaire détecté.\n")

    # ⚠️ Doublons
    summary_text.insert(tk.END, "\n⚠️ Quasi-doublons :\n\n")
    if duplicates:
        for sim, f1, f2 in duplicates:
            msg = (
                f"📄 {shorten_filename(os.path.basename(f1))} "
                f"↔ {shorten_filename(os.path.basename(f2))} "
                f"| sim={sim:.3f}\n"
            )
            summary_text.insert(tk.END, msg)
    else:
        summary_text.insert(tk.END, "Aucun quasi-doublon détecté.\n")

    summary_text.see(tk.END)
    status_label.config(text="✅ Analyse terminée")


################################################################ RECHERCHE ################################################
# Supprime les widgets intégrés à la zone de résultats, puis efface son texte.
def clear_results():
    results_text.config(state=tk.NORMAL)

    # Détruire les widgets intégrés
    for widget in results_text.window_names():
        results_text.nametowidget(widget).destroy()

    # Effacer le texte
    results_text.delete("1.0", tk.END)


# Insère le lien d'un PDF, son extrait et ses boutons de résumé et de liste de lecture.
def afficher_resultat(filepath, chunk):
    filename = os.path.basename(filepath)
    tag = f"link_{filename}_{hash(chunk)}"
    results_text.insert(tk.END, f"📄 {filename}\n", (tag,))
    results_text.tag_config(tag, foreground="blue", underline=True)

    # Ouvre le PDF cliqué avec l'application par défaut de Windows.
    def open_result(event, path=filepath):
        try:
            os.startfile(path)
        except Exception as error:
            logging.exception(f"Erreur : {error}")
            print(traceback.format_exc())
            results_text.insert(tk.END, f"❌ Impossible d'ouvrir {path} : {error}\n")
            results_text.see(tk.END)

    results_text.tag_bind(tag, "<Button-1>", open_result)
    results_text.tag_config("extrait", foreground="black", underline=False)
    excerpt = trim_empty_lines(chunk.strip()[:MAX_CHARS_PER_FILE])
    results_text.insert(tk.END, excerpt, "extrait")

    # Affiche le résumé enregistré pour ce PDF s'il est disponible.
    def resumer_json():
        root.after(0, lambda: summary_text.delete("1.0", tk.END))
        summary = read_dataset_summary(filename)  # on passe filename
        if summary:
            print(f"📄 Résumé existant trouvé pour : {filename}")
            root.after(0, lambda: summary_text.insert(tk.END, f"Résumé existant :\n{summary}"))
        else:
            print(f"❌ Aucun résumé existant pour : {filename}")
            root.after(0, lambda: summary_text.insert(tk.END, "Aucun résumé existant dans le dataset."))
        root.after(0, lambda: summary_text.see(tk.END))

    # 🔹 Bouton Résumé du passage
    # Génère un résumé local de l'extrait et l'affiche dans la zone de résumé.
    def resumer_text(chunk):
        print(f"✅ Résumé un extrait de : {filename}")
        root.after(0, lambda: summary_text.delete("1.0", tk.END))
        summary = resumer_texte(chunk)
        root.after(0, lambda: summary_text.insert(tk.END, f"Résumé du passage :\n{summary}"))
        root.after(0, lambda: summary_text.see(tk.END))
        root.update_idletasks()

    # Extrait le PDF et le résume avec OpenAI avant de proposer l'enregistrement du résumé.
    def resumer_pdf_worker(filepath, filename):
        try:
            texte = extract_text_from_pdf(filepath)
            summary = resumer_texte_gpt(texte)  # Limite pour éviter surcharge
            root.after(0, lambda:progress.config(value=0))
            root.after(0, lambda: summary_text.insert(tk.END, f"Résumé complet du PDF :\n{summary}"))
            root.after(0, lambda: summary_text.see(tk.END))
            root.after(0, lambda: status_label.config(text=f"✅ Résumé terminé : {filename}"))

            # Bouton Enregistrer le résumé
            btn_dataset_pdf = tk.Button(
                btn_frame,
                text="Enregistrer le résumé ?"
            )
            root.after(0, lambda: btn_dataset_pdf.pack(side=tk.LEFT, padx=2))

            btn_dataset_pdf.config(
                command=lambda f=filepath, t=texte, s=summary:
                    build_dataset(
                        f, t, s,
                        btn_resume_passage,
                        btn_resume_pdf,
                        btn_dataset_pdf
                    )
            )
        except Exception as e:
            logging.exception(f"Erreur : {e}")
            print(traceback.format_exc())
            root.after(0, lambda: status_label.config(text=f"❌ Erreur : {e}"))

    # 🔸 Bouton Résumé complet du PDF
    # Demande confirmation pour utiliser l'API OpenAI, puis lance le résumé complet en arrière-plan.
    def resumer_pdf():
        if not OPENAI_API_ACCESS:
            return

         # Demander confirmation AVANT l'appel OpenAI
        confirmation = messagebox.askyesno(
            "Confirmation — Utilisation OpenAI",
            "⚠️ ATTENTION\n\n"
            "Cette opération utilise l'API OpenAI et "
            "entraînera une consommation de vos crédits API.\n\n"
            f"Document : {filename}\n\n"
            "Voulez-vous continuer ?",
            icon="warning",
            parent=root
        )

        # L'utilisateur a cliqué sur Non
        if not confirmation:
            print("❌ Résumé annulé par l'utilisateur.")
            return

        print(f"✅ Résumé complet de : {filename}")
        root.after(0, lambda: status_label.config(text=f"⏳ Résumé en cours : {filename}"))
        root.after(0, lambda: summary_text.delete("1.0", tk.END))

         # Lancer le travail lourd dans un thread
        threading.Thread(
            target=resumer_pdf_worker,
            args=(filepath, filename),
            daemon=True
        ).start()


    # 🔸 Bouton Enregistre Résumé du PDF
    # Ajoute le résumé, un extrait et les métadonnées du PDF au fichier JSONL.
    def build_dataset(fname, text, summary,
                  btn_resume_passage,
                  btn_resume_pdf,
                  btn_dataset_pdf):

        output_file = SUMMARY_FILE

        if len(text) <= 100:
            print("Texte trop court !!")
            return

        with open(output_file, "a", encoding="utf-8") as fout:
            text_short = text[:3000]
            record = {
                "filename": os.path.basename(fname),
                "text": text_short,
                "summary": summary,
                "filepath": filepath,
                "date": datetime.now().isoformat()
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Confirme l'enregistrement et actualise les boutons associés au résumé.
        def update_ui():
            label_save = f"Résumé enregistré dans : {output_file}"
            summary_text.insert(tk.END, "\n"+label_save+"\n")
            status_label.config(text=label_save)

            # vérification réelle
            existing_summary = read_dataset_summary(fname)
            if existing_summary:
                print(f"Résumé existant pour : {fname}")
                btn_resume_passage.config(bg="#90EE90")
                btn_resume_pdf.config(state="disabled")
                btn_dataset_pdf.config(state="disabled")

        root.after(0, update_ui)


    # 🔘 Crée un conteneur pour les boutons
    btn_frame = tk.Frame(results_text)
    btn_resume_passage = tk.Button(btn_frame,
        text="Résumé existant",
        command=resumer_json
    )
    print(f"Vérification résumé existant pour : {filename}")
    existing_summary = read_dataset_summary(filename)
    btn_resume_pdf = tk.Button(
        btn_frame, text="Résumé complet du PDF", command=resumer_pdf,
        state=tk.NORMAL if OPENAI_API_ACCESS else tk.DISABLED
    )
    if existing_summary is not None:
        print(f"Résumé existant pour : {filename}")
        btn_resume_passage.config(bg="#90EE90")
        btn_resume_pdf.config(state="disabled")

    # 📌 Bouton "À lire"
    # Ajoute le document à la liste « À lire » et met à jour son bouton.
    def mark_to_read():
        add_to_read(filepath)
        btn_to_read.config(text="📌 À lire ✔", bg="#90EE90", state="disabled")
        status_label.config(
            text=f"📌 Article ajouté à la liste 'À lire' : {os.path.basename(filepath)}"
        )

    to_read_set = read_to_read_set()
    btn_to_read = tk.Button(btn_frame, text="📌 À lire", width=12)
    if filepath in to_read_set:
        btn_to_read.config(text="📌 À lire ✔", bg="#90EE90", state="disabled")
    else:
        btn_to_read.config(command=mark_to_read)

    btn_resume_passage.pack(side=tk.LEFT, padx=2)
    btn_resume_pdf.pack(side=tk.LEFT, padx=2)
    btn_to_read.pack(side=tk.LEFT, padx=2)
    results_text.window_create(tk.END, window=btn_frame)
    results_text.insert(tk.END, "\n\n")


# -----------------------------
# Recherche
# -----------------------------
# Lit la question dans Tkinter et lance la recherche dans l'index en arrière-plan.
def search_query():
    question = entry.get()
    clear_results()
    summary_text.delete("1.0", tk.END)
    if index.ntotal == 0 or not embedding_text_map:
        messagebox.showwarning("Recherche impossible",  "Index ou textes vides.")
        return

    if index.ntotal != len(chunk_embeddings) or index.ntotal != len(embedding_text_map):
        messagebox.showwarning("Recherche impossible", "Index et données incohérents.")
        return
        
    if not question.strip():
        messagebox.showwarning("Question vide", "Veuillez saisir une question.")
        return

    status_label.config(text="⏳ Recherche dans l'index...")
    results_text.insert(tk.END, f"⚠️ Cette recherche peut prendre du temps selon la taille de votre fichier d'indexation.\n\n")
    results_text.insert(tk.END, f"⏳ Recherche en cours... : {question}\n")
    
    run_search(lambda: find_index_results(question), lambda result: display_index_results(question, result))


# Prépare les correspondances sémantiques et textuelles, puis regroupe les PDF pertinents par thème.
def find_index_results(question):
    if index.ntotal == 0 or not embedding_text_map:
        raise ValueError("Recherche impossible : index ou textes vides.")
    if index.ntotal != len(chunk_embeddings) or index.ntotal != len(embedding_text_map):
        raise ValueError("Index et données incohérents.")

    results = get_relevant_documents(question, index, embedding_text_map, top_k=30)
    files_hits = defaultdict(list)
    for score, idx in results:
        idx = int(idx)
        if not (0 <= idx < len(embedding_text_map)):
            print(
                f"⚠️ Indice FAISS invalide : {idx} "
                f"(mapping : {len(embedding_text_map)} éléments)"
            )
            continue
        filepath, chunk = embedding_text_map[idx]

        if os.path.exists(filepath):
            files_hits[filepath].append((score, chunk))
        else:
            print(f"⚠️ Fichier supprimé ignoré : {filepath}")

    relevant_files = set(files_hits.keys())

    # 2️⃣ embeddings documents ciblés
    doc_embeddings_q = build_document_embeddings_subset(
        chunk_embeddings,
        embedding_text_map,
        relevant_files
    )

    # 3️⃣ clustering adapté
    k = safe_k(len(doc_embeddings_q), desired_k=6)
    clusters = cluster_documents(doc_embeddings_q, k=k)

    semantic_results = []
    for filepath, hits in list(files_hits.items())[:MAX_FILES]:
        hits_sorted = sorted(hits, key=lambda hit: hit[0], reverse=True)
        selected_chunks = []
        for _, chunk in hits_sorted:
            clean_chunk = " ".join(chunk.split())
            if clean_chunk in selected_chunks:
                continue
            selected_chunks.append(clean_chunk)
            if len(selected_chunks) >= MAX_EXTRACTS_PER_FILE:
                break
        semantic_results.append((filepath, "\n\n".join(selected_chunks)))

    exact_results = []
    already_displayed = set(files_hits)
    keyword = question.lower()
    for filepath, chunk in embedding_text_map:
        if filepath not in already_displayed and keyword in chunk.lower():
            exact_results.append((filepath, chunk[:MAX_CHARS_PER_FILE]))
            already_displayed.add(filepath)

    doc_embeddings = build_document_embeddings(chunk_embeddings, embedding_text_map)
    labels = label_clusters(clusters, doc_embeddings, embedding_text_map)
    return semantic_results, exact_results, clusters, labels


# Affiche les extraits, les liens PDF et les groupes thématiques renvoyés par la recherche.
def display_index_results(question, result):
    summary_text.delete("1.0", tk.END)
    clear_results()   
    semantic_results, exact_results, clusters, labels = result
    results_text.insert(tk.END, "\n--- RÉSULTATS DE RECHERCHE ---\n\n")
    results_text.insert(tk.END, f'🔍 Résultats triés pour : "{question}"\n')
    for filepath, chunk in semantic_results:
        afficher_resultat(filepath, chunk)

    results_text.insert(tk.END, f'\n\n🔍 Résultats contenant exactement "{question}" :\n')
    for filepath, chunk in exact_results:
        afficher_resultat(filepath, chunk)

    summary_text.insert(tk.END, f'\n\n🔍 Résultats par thèmes pour : "{question}"')
    for cid, files in clusters.items():
        info = labels.get(cid, {"label": "—", "representative": "—"})
        summary_text.insert(tk.END, f"\n🧠 Cluster {cid} — {info['label']}\n")
        representative = info["representative"]
        summary_text.insert(tk.END, f"   📄 Document central : {representative}\n")
        for filepath in sorted(set(files)):
            name = shorten_filename(os.path.basename(filepath))
            summary_text.insert(tk.END, f" - {name}\n")

    results_text.see("1.0")
    summary_text.see(tk.END)
    count = len(semantic_results) + len(exact_results)
    status_label.config(text=f"✅ Recherche terminée : {count} PDF(s) affiché(s)")


# Uniformise la casse, retire les accents et réduit les espaces pour la comparaison de texte.
def normalize_search_text(text):
    text = unicodedata.normalize("NFKD", text.casefold())

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    return re.sub(r"\s+", " ", text).strip()


# Cherche un terme en exigeant une frontière de mot à ses deux extrémités.
def contains_term(text, term):
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text) is not None


# Uniformise la casse et retire les accents en conservant les espacements.
def normalize_search_text_(text):
    """Normalisation pour une recherche insensible à la casse et aux accents."""
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        c for c in text
        if not unicodedata.combining(c)
    )


# Classe les passages avec les termes proposés par Qwen et la correspondance exacte de la question.
def search_text_chunks_qwen(
    question,
    query_analysis,
    embedding_text_map,
    max_results=30,
    min_score=5
):
    normalized_question = normalize_search_text(question)
    results = []

    for filepath, chunk in embedding_text_map:

        if not os.path.isfile(filepath):
            continue

        score = score_chunk_with_qwen_analysis(
            chunk,
            query_analysis
        )

        if normalized_question and contains_term(normalize_search_text(chunk), normalized_question):
            score += 12

        if score >= min_score:
            results.append((score, filepath, chunk))

    results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return results[:max_results]


# Cache
# Met en cache l'analyse Qwen d'une question jusqu'à éviction ou vidage du cache.
@lru_cache(maxsize=256)
def cached_query_analysis(question):
    return analyze_query_with_qwen(question)


# Tronque le contexte à la limite de tokens demandée, puis le reconvertit en texte.
def limit_qwen_context(context, tokenizer, max_tokens=3000):
    """Limite le contexte documentaire au nombre de tokens indiqué."""

    tokens = tokenizer(
        context,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=False
    )["input_ids"]

    limited_context = tokenizer.decode(
        tokens,
        skip_special_tokens=True
    )

    print(f"📏 Contexte Qwen : {len(tokens)} tokens")

    return limited_context


# Fusionne les passages retrouvés par FAISS et par recherche textuelle, puis ajoute les résumés disponibles.
def build_qwen_context(question):
    # Étape 1 : analyse de la question par Qwen
    t0 = time.perf_counter()
    cached_query_analysis.cache_clear()
    query_analysis = cached_query_analysis(question)
    print("🔎 Analyse de la question :")
    print("Sujet :", query_analysis["subject"])
    print("Intentions :", query_analysis["intent"])
    print("Mots-clés :", query_analysis["keywords"])
    print("Synonymes :", query_analysis["synonyms"])
    print("Expressions :", query_analysis["expressions"])
    print(f"⏱️ Pré-analyse : {time.perf_counter() - t0:.2f} s", flush=True)

    # Étape 2 : recherche sémantique FAISS
    t1 = time.perf_counter()
    semantic_results = []

    if index.ntotal > 0 and embedding_text_map:
        semantic_results = get_relevant_documents(
            question,
            index,
            embedding_text_map,
            top_k=MAX_QWEN_CONTEXT_DOCS * MAX_QWEN_EXTRACTS_PER_FILE * 2
        )
    print(f"⏱️ FAISS : {time.perf_counter() - t1:.2f} s", flush=True)

    # Étape 3 : recherche textuelle enrichie par Qwen
    t2 = time.perf_counter()
    text_results = search_text_chunks_qwen(
        question,
        query_analysis,
        embedding_text_map,
        max_results=MAX_QWEN_CONTEXT_DOCS * MAX_QWEN_EXTRACTS_PER_FILE * 2
    )
    print(f"⏱️ Recherche textuelle : {time.perf_counter() - t2:.2f} s", flush=True)

    # 3. Fusionner les résultats
    t3 = time.perf_counter()
    files_hits = defaultdict(list)
    seen = set()

    # Vérifie le PDF et déduplique le passage avant de l'ajouter aux résultats de ce document.
    def add_hit(filepath, chunk, source, score=None):
        if not os.path.isfile(filepath):
            return

        clean_chunk = trim_empty_lines(chunk).strip()
        if not clean_chunk:
            return

        key = (os.path.normcase(os.path.abspath(filepath)), clean_chunk)

        if key in seen:
            return

        seen.add(key)
        files_hits[filepath].append({
            "score": score,
            "chunk": clean_chunk,
            "source": source
        })

    # Priorité aux résultats textuels exacts
    #for filepath, chunk in text_results:
    #    add_hit(filepath, chunk, "text")

    for score, filepath, chunk in text_results:
        add_hit(filepath, chunk, "text", score)

    # Ajouter les résultats sémantiques
    for score, idx in semantic_results:
        idx = int(idx)
        if not (0 <= idx < len(embedding_text_map)):
            continue

        filepath, chunk = embedding_text_map[idx]
        add_hit(filepath, chunk, "semantic", float(score))

    if not files_hits:
        return ("", [])

    print(f"⏱️ Fusionne des résultats : {time.perf_counter() - t3:.2f} s", flush=True)

    # 4. Construire le contexte
    t4 = time.perf_counter()
    relevant_documents = []
    context_parts = []

    for filepath, hits in list(files_hits.items())[:MAX_QWEN_CONTEXT_DOCS]:

        filename = os.path.basename(filepath)
        doc_parts = [f"Document : {filename}"]

        # Résumé existant
        summary = read_dataset_summary(filepath)
        if summary:
            clean_summary = trim_empty_lines(summary)[:MAX_QWEN_SUMMARY_CHARS].strip()
            if clean_summary: doc_parts.append(f"Résumé disponible :\n{clean_summary}")

        # Trier : recherche textuelle d'abord,
        # puis similarité sémantique décroissante
        hits_sorted = sorted(
            hits,
            key=lambda h: (
                0 if h["source"] == "text" else 1,
                -(h["score"] if h["score"] is not None else 0)
            )
        )

        selected_chunks = []
        for hit in hits_sorted:
            clean_chunk = hit["chunk"][:MAX_QWEN_EXTRACT_CHARS].strip()
            if not clean_chunk:
                continue

            selected_chunks.append((hit["source"], clean_chunk))
            if len(selected_chunks) >= MAX_QWEN_EXTRACTS_PER_FILE:
                break

        # Ajouter les extraits au contexte
        for i, (source, chunk) in enumerate(selected_chunks, 1):
            source_label = (
                "Correspondance textuelle"
                if source == "text"
                else "Correspondance sémantique"
            )

            doc_parts.append(f"Extrait pertinent {i} " f"({source_label}) :\n{chunk}")

        context_parts.append("\n\n".join(doc_parts))

        # Score du meilleur extrait de ce document
        best_score = max(
            (hit["score"] for hit in hits
            if hit["source"] == "text"
            and hit["score"] is not None),
            default=0
        )

        relevant_documents.append({
            "filepath": filepath,
            "score": best_score,
            "page": None
        })

    print(f"⏱️ Construction du contexte : {time.perf_counter() - t4:.2f} s", flush=True)
    context = "\n\n---\n\n".join(context_parts)
    return context, relevant_documents


# Demande à Qwen d'extraire le sujet et les termes de recherche, avec la question comme solution de repli.
def analyze_query_with_qwen(question):
    """
    Utilise Qwen pour générer des termes de recherche.
    Retourne toujours un dictionnaire utilisable,
    même si la génération échoue.
    """

    fallback = {
        "subject": "",
        "intent": [],
        "keywords": [question],
        "synonyms": [],
        "expressions": []
    }

    system_prompt = """
Tu es un assistant spécialisé dans la recherche documentaire.

Analyse la question pour améliorer une recherche dans des PDF.

Tu dois identifier :

1. subject :
   Le sujet central précis de la question.
   Il doit être court et ne pas contenir la demande.

2. intent :
   Les notions qui décrivent ce que l'utilisateur
   souhaite connaître concernant le sujet.
   Propose aussi des termes équivalents.

3. keywords :
   Les principaux mots-clés de recherche.

4. synonyms :
   Des formulations alternatives qui conservent
   le même sujet et la même intention.

5. expressions :
   Des expressions susceptibles d'apparaître
   dans les documents et de répondre à la question.


Règles :
- Identifie le sujet central de la question.
- Conserve toujours ce sujet dans les expressions proposées.
- Conserve les noms propres et termes techniques.
- Ne remplace pas le sujet de la question.
- Ne génère pas de réponses à la question.
- N'invente pas de faits.
- Propose des synonymes du concept recherché.
- Ne remplace jamais un concept spécifique par un concept plus large.
- Propose au maximum 8 termes par catégorie.
- Réponds uniquement en français avec un objet JSON valide.

Format :
{
  "subject": "",
  "intent": [],
  "keywords": [],
  "synonyms": [],
  "expressions": []
}
"""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        inputs = summ_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt"
        )
        # Envoyer les tenseurs sur le périphérique du modèle
        inputs = {
            key: value.to(summ_model.device)
            for key, value in inputs.items()
        }

        # Génération déterministe
        with torch.inference_mode():
            outputs = summ_model.generate(
                **inputs,
                max_new_tokens=MAX_LENGTH_TOKENIZER,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=summ_tokenizer.eos_token_id
            )

        # Ne récupérer que la réponse générée
        generated = outputs[0][inputs["input_ids"].shape[-1]:]


        response = summ_tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        print("🤖 Réponse brute Qwen :", repr(response))

        # Supprimer les blocs de raisonnement terminés
        response = re.sub(
            r"<think>.*?</think>",
            "",
            response,
            flags=re.DOTALL
        ).strip()

        # Extraire l'objet JSON
        match = re.search(r"\{.*\}", response, re.DOTALL)

        if not match:
            print("❌ Aucun JSON détecté dans la réponse Qwen")
            return fallback

        data = json.loads(match.group(0))

        # Après json.loads(...)
        result = {
            "subject": "",
            "intent": [],
            "keywords": [],
            "synonyms": [],
            "expressions": []
        }

        # Sujet central
        subject = data.get("subject", "")
        if isinstance(subject, str):
            result["subject"] = subject.strip()

        # Listes
        for key in ("intent", "keywords", "synonyms", "expressions"):
            values = data.get(key, [])

            if not isinstance(values, list):
                values = []

            result[key] = [
                str(value).strip()
                for value in values
                if isinstance(value, str) and value.strip()
            ][:8]

        return result

    except Exception as error:
        logging.exception(f"Erreur : {error}")
        print(traceback.format_exc())
        print(f"⚠️ Analyse Qwen impossible : {error}")
        return fallback


# Note un passage selon le sujet, le vocabulaire et les intentions extraits de la question.
def score_chunk_with_qwen_analysis(chunk, analysis):
    text = normalize_search_text(chunk)

    subject = normalize_search_text(
        analysis.get("subject", "")
    )

    intents = [
        normalize_search_text(term)
        for term in analysis.get("intent", [])
        if term.strip()
    ]

    keywords = [
        normalize_search_text(term)
        for term in analysis.get("keywords", [])
        if term.strip()
    ]

    synonyms = [
        normalize_search_text(term)
        for term in analysis.get("synonyms", [])
        if term.strip()
    ]

    expressions = [
        normalize_search_text(term)
        for term in analysis.get("expressions", [])
        if term.strip()
    ]

    score = 0

    # Sujet central : priorité absolue
    subject_found = bool(
        subject and contains_term(text, subject)
    )

    if subject_found:
        score += 10

    # Intentions de recherche
    intent_matches = sum(
        contains_term(text, term)
        for term in set(intents)
    )

    score += min(intent_matches, 4) * 3

    # Bonus sujet + intention
    if subject_found and intent_matches:
        score += 10

    # Mots-clés
    for term in set(keywords):
        if contains_term(text, term):
            score += 4

    # Synonymes
    for term in set(synonyms):
        if contains_term(text, term):
            score += 3

    # Expressions complémentaires
    for term in set(expressions):
        if contains_term(text, term):
            score += 2

    # Réduire les résultats qui ne mentionnent
    # pas le sujet central
    if subject and not subject_found:
        score *= 0.15

    return round(score, 2)


# Assemble le prompt commun à Qwen et ChatGPT avec les extraits FAISS et le contexte par document.
def build_form_query(question):
    chunks = []
    if index.ntotal > 0 and embedding_text_map:
        vec = get_embedding(question)
        if vec is not None:
            _, indices = index.search(np.expand_dims(vec, axis=0), 5)
            chunks = [
                embedding_text_map[i][1]
                for i in indices[0]
                if 0 <= i < len(embedding_text_map)
            ]
    contexte_index = trim_empty_lines("\n\n".join(chunks)).strip()

    contexte_files, relevant_documents = build_qwen_context(question)
    if contexte_files:
        contexte_files = limit_qwen_context(
            contexte_files, summ_tokenizer, max_tokens=MAX_QWEN_CONTEXT_LIMIT
        ).strip()

    context_parts = []
    if contexte_index:
        context_parts.append(
            "Extraits les plus proches dans l'index :\n"
            f"{contexte_index}\n"
        )
    if contexte_files:
        context_parts.append(
            "Documents pertinents avec résumés disponibles :\n"
            f"{contexte_files}\n"
        )

    contexte = trim_empty_lines("\n\n---\n\n".join(context_parts)) if context_parts else ""

    if not contexte:
        return None

    prompt_msg = (
        f"Voici un contexte issu de documents PDF. "
        f"Chaque document peut contenir un résumé existant et des extraits retrouvés par l'index :\n\n"
        f"{contexte}\n\n"
        f"En te basant uniquement sur ce contexte, réponds en français à la question suivante. "
        f"Si le contexte ne permet pas de répondre, dis-le clairement :\n"
        f"{question}"
    )

    return prompt_msg, relevant_documents


# Lit la question et lance la recherche documentaire suivie d'une réponse locale dans un thread.
def repondre_avec_qwen():
    question = entry.get()
    clear_results()
    summary_text.delete("1.0", tk.END)
    if not question.strip():
        messagebox.showwarning("Question vide", "Veuillez saisir une question.")
        return

    status_label.config(text=f"⏳ Recherche avec Qwen... : {question}")
    results_text.insert(tk.END, f"⚠️ Cette recherche peut prendre plusieurs minutes selon la puissance de votre ordinateur.\n\n")
    results_text.insert(tk.END, f"🖥️ Périphérique : {summ_model.device}\n")

    run_search(lambda: generate_qwen_answer(question), lambda result: display_qwen_answer(*result))


# Prépare le contexte, génère la réponse avec Qwen et renvoie le texte accompagné de ses documents.
def generate_qwen_answer(question):
    prepared_query = build_form_query(question)
    if prepared_query is None:
        return None

    prompt_text, relevant_documents = prepared_query
    messages = [{"role": "user", "content": prompt_text}]
    t0 = time.perf_counter()
    prompt = summ_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = summ_tokenizer(prompt, return_tensors="pt", truncation=True).to(summ_model.device)

    nb_input_tokens = inputs["input_ids"].shape[-1]
    print(f"📥 Tokens en entrée : {nb_input_tokens}")
    print(f"🖥️ Périphérique : {summ_model.device}")

    with torch.inference_mode():
        outputs = summ_model.generate(
            **inputs,
            max_new_tokens=MAX_LENGTH_TOKENIZER,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=summ_tokenizer.eos_token_id
        )

    # Nombre de tokens du prompt envoyé à Qwen
    input_length = inputs["input_ids"].shape[-1]
    # Ne conserver que les tokens générés
    generated_tokens = outputs[0][input_length:]
    # Décoder uniquement la réponse
    reponse = summ_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    nb_generated = outputs.shape[-1] - nb_input_tokens
    print(f"📤 Tokens générés : {nb_generated}", flush=True)
    print(f"Temps de génération Qwen : {time.perf_counter() - t0:.2f} s", flush=True)
    return reponse, relevant_documents


# Affiche la réponse et crée des liens vers les PDF associés.
def display_qwen_answer(answer, relevant_documents):
    """Affiche la réponse Qwen et les PDF cliquables."""
    status_label.config(text=f"")
    summary_text.config(state=tk.NORMAL)
    summary_text.delete("1.0", tk.END)
    results_text.config(state=tk.NORMAL)
    results_text.delete("1.0", tk.END)
    root.update_idletasks()

    # Réponse Qwen
    results_text.insert(tk.END, "🤖 RÉPONSE DE QWEN\n\n")
    results_text.insert(tk.END, answer.strip() + "\n\n")
    summary_text.insert(tk.END, "🤖 RÉPONSE DE QWEN\n\n")
    summary_text.insert(tk.END, answer.strip() + "\n\n")


    # Documents
    results_text.insert(tk.END, "📚 DOCUMENTS LES PLUS PERTINENTS\n\n")

    if not relevant_documents:
        results_text.insert(tk.END, "Aucun document associé.\n")

    for i, doc in enumerate(relevant_documents, 1):
        filepath = doc["filepath"]
        page = doc.get("page")
        filename = Path(filepath).name
        label = f"📄 {i}. {filename}"
        if page is not None:
            label += f" — page {page}"
        tag = f"qwen_pdf_link_{i}"

        # Insérer le lien avec un tag propre
        results_text.insert(tk.END, label, (tag,))
        results_text.insert(tk.END, "\n")

        # Apparence du lien
        results_text.tag_configure(tag, foreground="#1976D2", underline=True)

        # Ouvrir le PDF au clic
        results_text.tag_bind(tag, "<Button-1>", lambda event, fp=filepath, p=page: open_pdf(fp, p))

        # Curseur au survol
        results_text.tag_bind(tag, "<Enter>", lambda event: results_text.config(cursor="hand2"))

        results_text.tag_bind(tag, "<Leave>", lambda event: results_text.config(cursor=""))

    status_label.config(text=f"✅ Recherche terminée")
    results_text.config(state=tk.DISABLED)
    root.update_idletasks()


# Demande confirmation avant de lancer la recherche et l'appel OpenAI en arrière-plan.
def repondre_avec_gpt():
    question = entry.get()
    summary_text.delete("1.0", tk.END)
    clear_results()
    
    if not OPENAI_API_ACCESS:
        results_text.insert(tk.END, f"⚠️ Cette fonctionnalité n'est pas disponible. Vérifiez vos fichiers 'openai_key.txt' et 'openai_model.txt'.\n\n")
        return
    
    if not question.strip():
        messagebox.showwarning("Question vide", "Veuillez saisir une question.")
        return
    # Demander confirmation AVANT l'appel OpenAI
    confirmation = messagebox.askyesno(
        "Confirmation — Utilisation OpenAI",
        "⚠️ ATTENTION\n\n"
        "Cette opération utilise l'API OpenAI et "
        "entraînera une consommation de vos crédits API.\n\n"
        "Voulez-vous continuer ?",
        icon="warning",
        parent=root
    )

    # L'utilisateur a cliqué sur Non
    if not confirmation:
        print("❌ Recherche annulée par l'utilisateur.")
        return

    status_label.config(text="⏳ Recherche avec ChatGPT...")
    results_text.insert(tk.END, f"⚠️ Cette recherche consomme votre crédit ChatGPT.\n\n")
    results_text.insert(tk.END, f"⏳ Recherche en cours... : {question}\n")

    run_search(lambda: generate_gpt_answer(question), display_gpt_answer)


# Construit le prompt documentaire et demande la réponse au modèle OpenAI configuré.
def generate_gpt_answer(question):
    system_prompt = (
        "Tu es un assistant académique spécialisé en sciences humaines.\n"
        "Voici plusieurs résumés partiels d’articles scientifiques.\n"
    )
    prepared_query = build_form_query(question)
    if prepared_query is None:
        return None

    user_prompt, _ = prepared_query

    response = call_gpt(system_prompt, user_prompt)
    if response is None or len(response) < 10:
        response = "Pas de réponse produite"
    return trim_empty_lines(response)


# Insère la réponse ChatGPT dans la zone de résumé et actualise le statut.
def display_gpt_answer(response):
    status_label.config(text="✅ Recherche terminée")
    summary_text.insert(tk.END, response)
    summary_text.see(tk.END)


# Ouvre un PDF sous Windows, à la page demandée si Adobe Acrobat est disponible.
def open_pdf_acrobat(pdf_path, page):
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        print(f"❌ PDF introuvable : {pdf_path}")
        return

    acrobat_paths = [
        r"C:\Program Files\Adobe\Acrobat DC\Acrobat\Acrobat.exe",
        r"C:\Program Files\Adobe\Acrobat DC\Acrobat\AcroRd32.exe",
        r"C:\Program Files (x86)\Adobe\Acrobat Reader DC\Reader\AcroRd32.exe",
    ]

    for reader_path in acrobat_paths:
        if Path(reader_path).is_file():
            command = [reader_path]
            if page is not None:
                command.extend(["/A", f"page={int(page)}"])

            command.append(str(pdf_path))
            subprocess.Popen(command)
            return

    # Si aucun lecteur Adobe n'est trouvé
    os.startfile(str(pdf_path))


# Ouvre un PDF avec la commande adaptée au système d'exploitation.
def open_pdf(pdf_path, page):
    # Accepte aussi bien str que Path
    pdf_path = Path(pdf_path).resolve()
    #pdf_path = str(pdf_path.resolve())

    if not pdf_path.is_file():
        print(f"❌ Fichier introuvable : {pdf_path}")
        return

    if sys.platform == "win32":
        open_pdf_acrobat(pdf_path, page)

    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(pdf_path)])

    else:
        subprocess.Popen(["xdg-open", str(pdf_path)])


# Demande le dossier et lance la recherche dans le texte brut des PDF en arrière-plan.
def search_text_pdfs():
    keyword = entry.get().strip().lower()
    clear_results()
    summary_text.delete("1.0", tk.END)
    if not keyword.strip():
        messagebox.showwarning("Mots clés vides", "Veuillez saisir un ou des mots clés.")
        return

    selected_folder = filedialog.askdirectory(title="Sélectionner un dossier contenant des PDF")
    if not selected_folder:
        return

    folder = Path(selected_folder)
    status_label.config(text="⏳ Recherche dans les PDF...")
    results_text.insert(tk.END, f"⚠️ Cette recherche peut prendre du temps selon le nombre de fichiers traités.\n\n")
    results_text.insert(tk.END, f"⏳ Recherche en cours... : {keyword}\n")
    results_text.insert(tk.END, f"📂 Recherche depuis : {folder}\n")

    run_search(
        lambda report: find_text_in_pdfs(folder, keyword, report),
        display_text_pdf_results,
        update_text_pdf_progress
    )


# Affiche le nombre de PDF parcourus et met à jour la barre de progression.
def update_text_pdf_progress(processed, total):
    if progress["mode"] != "determinate":
        progress.stop()

    progress.config(maximum=max(total, 1), value=processed, mode="determinate")
    status_label.config(text=f"⏳ Recherche : {processed}/{total} PDF")


# Parcourt les PDF du dossier et de ses sous-dossiers, puis regroupe les pages correspondantes par fichier.
def find_text_in_pdfs(folder, keyword, report_progress):
    grouped_results = defaultdict(list)
    folder = Path(folder)

    pdf_files = sorted(folder.rglob("*.pdf"))
    nb_files = len(pdf_files)
    report_progress(0, nb_files)
    keyword = keyword.lower()

    for nb_f, pdf_file in enumerate(pdf_files, 1):
        name = pdf_file.relative_to(folder)
        try:
            reader = PdfReader(str(pdf_file), strict=False)

            for number, page in enumerate(reader.pages, 1):
                try:
                    text = (page.extract_text() or "").lower()
                    if keyword in text:
                        grouped_results[pdf_file].append(number)

                except (PdfReadError, PdfStreamError, OSError, ValueError) as e:
                    logging.warning("Page %d illisible dans %s : %s", page_num, pdf_file, e)
                    continue

        except (PdfReadError, PdfStreamError, OSError, ValueError) as e:
            logging.warning("PDF illisible : %s - %s", pdf_file, e)
            print(f"⚠️ PDF ignoré : {pdf_file}")
            print(f"   Raison : {e}")

        except Exception as error:
            logging.exception(f"Erreur : {error}")
            print(traceback.format_exc())
            print(f"Erreur avec {name} : {error}")

        finally:
            report_progress(nb_f, nb_files)

    return grouped_results


# Affiche les pages trouvées sous forme de liens et indique le nombre de résultats.
def display_text_pdf_results(grouped_results):
    occurrence_count = sum(len(pages) for pages in grouped_results.values())
    # Effacer les anciens résultats
    results_text.delete("1.0", tk.END)
    results_text.insert(tk.END, "\n--- RÉSULTATS DE RECHERCHE ---\n\n")

    # Afficher chaque fichier une seule fois
    for i, (pdf_file, pages) in enumerate(grouped_results.items()):
        filename = shorten_filename(pdf_file.name)
        results_text.insert(tk.END, f"📄 {filename} | ")
        for j, page in enumerate(pages):
            tag = f"link_{i}_{j}"
            # Numéro de page cliquable
            results_text.insert(tk.END, f"Page {page}", (tag,))
            results_text.tag_config(tag, foreground="blue", underline=True)
            # Clic sur le numéro de page
            results_text.tag_bind(
                tag,
                "<Button-1>",
                lambda event, path=pdf_file, p=page:
                    open_pdf(path, p)
            )
            # Curseur main
            results_text.tag_bind(
                tag,
                "<Enter>",
                lambda event:
                    results_text.config(cursor="hand2")
            )
            results_text.tag_bind(
                tag,
                "<Leave>",
                lambda event:
                    results_text.config(cursor="xterm")
            )
            # Séparateur entre les pages
            if j < len(pages) - 1:
                results_text.insert(tk.END, " | ")

        results_text.insert(tk.END, "\n")
    # Statistiques
    results_text.insert(
        tk.END,
        f"\nOccurrences : {occurrence_count} page(s)\n"
        f"Fichiers concernés : {len(grouped_results)}\n"
    )
    status_label.config(text=f"✅ Occurrences : {occurrence_count} page(s)")
    results_text.see("1.0")


# Text de l'about
ABOUT_TEXT = dedent(f"""\
{APP_NAME} avec {MODEL_QWEN}
Version : {APP_VERSION}

Recherche intelligente dans les fichiers PDF.

Fonctionnalités :
• Recherche sémantique avec FAISS
• Réponses générées avec Qwen
• Réponses générées avec ChatGPT
• Recherche textuelle
• Résumés générés avec ChatGPT
• Consultation des documents PDF
• Recherche de doublons

Développé en Python avec Tkinter
en Vibe Coding avec l'IA.

Auteur : Eric LAMETTE © 2026
Contact : ericlmt@gmail.com

────────────────────────────────────
LICENCE GNU GPL v3
────────────────────────────────────

This program is free software: you can redistribute it
and/or modify it under the terms of the GNU General
Public License as published by the Free Software
Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public
License for more details.

You should have received a copy of the GNU General
Public License along with this program.
If not, see https://www.gnu.org/licenses/.
""")


# Ouvre la fenêtre « À propos » avec les informations sur l'application et ses liens.
def about_pdf_search():
    about = tk.Toplevel(root)
    about.title("À propos — Recherche PDF")
    #about.geometry("480x340")
    #about.resizable(False, False)
    about.transient(root)

    frame = tk.Frame(about, padx=25, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        frame,
        text="Recherche PDF avec Qwen",
        font=("Arial", 15, "bold")
    ).pack(pady=(0, 5))

    tk.Label(
        frame,
        text=ABOUT_TEXT,
        font=("Arial", 9),
        justify=tk.LEFT,
        anchor="w",
        wraplength=600
    ).pack(pady=(0, 5))

    tk.Label(
        frame,
        text="Crédits — Intelligence artificielle",
        font=("Arial", 11, "bold")
    ).pack(pady=(0, 8))

    tk.Label(
        frame,
        text=("Cette application utilise un modèle Qwen,\ndéveloppé par l'équipe Qwen d'Alibaba Cloud,\npour analyser et interroger des documents PDF."),
        justify=tk.LEFT,
    ).pack(pady=(0, 5))

    # Lien officiel Qwen
    link = tk.Label(
        frame,
        text="🌐 Qwen sur Hugging Face",
        fg="#1976D2",
        cursor="hand2",
        font=("Arial", 10, "underline")
    )
    link.pack(pady=(12, 5))

    link.bind(
        "<Button-1>",
        lambda event: webbrowser.open(
            "https://huggingface.co/Qwen"
        )
    )

    tk.Label(
        frame,
        text="Licence : consulter la fiche du modèle utilisé",
        font=("Arial", 9),
        fg="gray"
    ).pack(pady=5)

    tk.Button(
        frame,
        text="Fermer",
        command=about.destroy,
        width=12
    ).pack(pady=(15, 0))


# Affiche dans la fenêtre principale l'aide sur les fonctions du logiciel.
def use_pdf_search():
    use_message = """⚠️ N'effacez jamais les fichiers créés par l'application dans son répertoire. ⚠️
⚠️ Regroupez UNE COPIE de vos fichers Pdf dans le dossier PDFS, en créant éventuellement des sous dossiers, ⚠️
⚠️ ce sera votre répertoire de travail centralisé. ⚠️
La configuration OpenAI est facultative. Si vous renseignez une clé, elle est enregistrée en clair dans 'openai_key.txt'.

📚 FONCTIONNALITÉS :
--------------------

✅ Recherche sémantique avec FAISS
    - Pour pouvoir effectuer une recherche par Index, vous devez au préalable avoir indexé le dossier qui contient vos fichiers Pdf.
      Cette indexation créera les fichiers nommés 'indexed_files.txt', 'index_qwen.idx' et 'texts.pkl' dans le répertoire 'datas' de l'application, ne les effacés pas après leur création, c'est la base de données de votre Index.
    - Après votre recherche, les résultats vous permettront de demander des résumés via ChatGPT, ils seront enregistrés dans le fichier summarization_data.jsonl.
    - Vous aurez également la possibilité de marquer vos Pdf comme étant 'À lire', ils seront répertoriés dans le fichier 'to_read.jsonl' dans le répertoire 'datas'.

✅ Réponse générée avec Qwen
    - Cette recherche est particulièrement gourmande en ressources mais peut-être très utile car elle combine à la fois la recherche dans l'Index, la recherche textuelle et fournis des résultats plus précis.

✅ Réponse générée avec ChatGPT
    - Cette fonction permet d'interroger directement ChatGPT, en lui fournissant comme contexte les extraits de vos Pdfs les plus pertinents.

✅ Recherche textuelle
    - Cette recherche est une recherche textuelle brute dans les Pdf du répertoire que vous séléctionné.

✅ Résumés générées avec ChatGPT
    - Lors de vos recherche dans l'Index, vous aurez la possibilité de demander des résumés de chaque Pdf isolément à ChatGPT, ils seront conservés dans le fichier 'summarization_data.jsonl' dans le répertoire 'datas'.

✅ Consultation des documents PDF
    - Dans les resultats, si le nom du fichier ou le numéro de la page apparait sous forme de lien hypertexte, vous avez la possibilité de l'ouvrir et de le ou la consulter.

✅ Recherche de doublons
    - Cette fonction recherche les fichiers les plus proches dans leur contenu et en affiche la liste.
\n\n"""

    summary_text.delete("1.0", tk.END)
    results_text.delete("1.0", tk.END)
    results_text.insert(tk.END, use_message)


# Affiche un message de fermeture et détruit la fenêtre principale.
def exit_pdf_search():
    for thread in threading.enumerate():
        print(thread.name, thread.daemon)

    messagebox.showinfo("Quitter...",f"Merci d'avoir utilisé {APP_NAME}", parent=root)
    root.destroy()

# ----------------------------------
# Composition de l'Interface console
# ----------------------------------
if __name__ == "__main__":
    # -----------------------------
    # Interface console
    # -----------------------------

    root = tk.Tk()

    root.protocol("WM_DELETE_WINDOW", exit_pdf_search)

    # Les fonctions OpenAI sont disponibles lorsque la clé et le modèle sont renseignés.
    OPENAI_API_KEY = load_or_ask_config(
        "openai_key.txt",
        "Configuration OpenAI (facultative)",
        "Entrez votre clé API OpenAI. Elle sera enregistrée en clair dans openai_key.txt.\n"
        "Pour continuer sans OpenAI, cliquez sur Annuler ou laissez le champ vide.",
        secret=True
    )

    if OPENAI_API_KEY:
        MODEL_NAME = load_or_ask_config(
            "openai_model.txt",
            "Modèle OpenAI (facultatif)",
            "Entrez le nom du modèle OpenAI, par exemple gpt-4.1-mini.\n"
            "Pour continuer sans OpenAI, cliquez sur Annuler ou laissez le champ vide."
        )

    OPENAI_API_ACCESS = bool(OPENAI_API_KEY and MODEL_NAME)
    if OPENAI_API_ACCESS:
        client = OpenAI(api_key=OPENAI_API_KEY)

    root.title(f"{APP_NAME} avec {MODEL_QWEN} v. {APP_VERSION}")
    root.geometry("1200x800")

    # Barre de menu
    menu_bar = tk.Menu(root)

    # Menu Quitter
    exit_menu = tk.Menu(menu_bar, tearoff=0)
    exit_menu.add_command(label="Quitter...", command=exit_pdf_search)
    menu_bar.add_cascade(label="Quitter", menu=exit_menu)

    # Menu Aide
    help_menu = tk.Menu(menu_bar, tearoff=0, name="help")
    help_menu.add_command(label="Àide...", command=use_pdf_search)
    help_menu.add_command(label="À propos...", command=about_pdf_search)
    menu_bar.add_cascade(label="Aide", menu=help_menu)

    root.config(menu=menu_bar)

    # Titre de l'application
    root_label = tk.Label(root, text="Recherche PDF avec Qwen...", font=("Arial", 10, "bold"))
    root_label.pack(pady=(5, 0))

    # 🔍 Zone de recherche
    search_frame = tk.Frame(root)
    search_frame.pack(pady=5)

    entry = tk.Entry(search_frame, width=80)
    entry.pack(side=tk.LEFT, padx=5)

    btn_search = tk.Button(search_frame, text="🔍 dans l'Index", command=search_query)
    btn_search.pack(side=tk.LEFT)
    btn_answer = tk.Button(search_frame, text="🔍 avec Qwen", command=repondre_avec_qwen, state="disabled")
    btn_answer.pack(side=tk.LEFT, padx=5)
    btn_answer_gpt = tk.Button(
        search_frame, text="🔍 avec ChatGPT", command=repondre_avec_gpt,
        state=tk.NORMAL if OPENAI_API_ACCESS else tk.DISABLED
    )
    btn_answer_gpt.pack(side=tk.LEFT)
    #btn_answer_gpt.pack(side=tk.LEFT, padx=5)
    btn_answer_brut = tk.Button(search_frame, text="🔍 texte Brut", command=search_text_pdfs)
    #btn_answer_brut.pack(side=tk.LEFT)
    btn_answer_brut.pack(side=tk.LEFT, padx=5)

    # 📂 Bouton d’indexation
    index_frame = tk.Frame(root)
    index_frame.pack(pady=5)

    btn_index = tk.Button(
        index_frame,
        text="📂 Sélectionner dossier PDF et indexer",
        command=index_pdfs
    )
    btn_index.pack(side=tk.LEFT, padx=5)

    btn_similar = tk.Button(
        index_frame,
        text="🧠 Similaires & doublons",
        command=show_similars_and_duplicates
    )
    btn_similar.pack(side=tk.LEFT, padx=5)

    # 📊 Barre de progression
    progress = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
    progress.pack(pady=5)
    status_label = tk.Label(root, text="", font=("Arial", 10, "italic"))
    status_label.pack()

    # Fenêtre texte
    paned = tk.PanedWindow(root, orient=tk.VERTICAL)
    paned.pack(fill="both", expand=True, padx=10, pady=10)

    # Partie haute
    frame_top = tk.Frame(paned)
    results_label = tk.Label(frame_top, text="Documents trouvés :", font=("Arial", 10, "bold"))
    results_label.pack()
    results_text = scrolledtext.ScrolledText(frame_top, wrap=tk.WORD, height=20)
    results_text.pack(fill="both", expand=True)
    paned.add(frame_top)

    # Partie basse
    frame_bottom = tk.Frame(paned)
    summary_label = tk.Label(frame_bottom, text="Document sélectionné :", font=("Arial", 10, "bold"))
    summary_label.pack(pady=5)
    # Frame pour les boutons
    frame_buttons = tk.Frame(frame_bottom)
    frame_buttons.pack(pady=5)
    btn_copier = tk.Button(frame_buttons, text="📋 Copier le texte", command=copier_resume)
    btn_copier.pack(side=tk.LEFT, padx=5)
    btn_print = tk.Button(frame_buttons, text="📋 Imprimer le texte", command=print_resume)
    btn_print.pack(side=tk.LEFT, padx=5)

    summary_text = scrolledtext.ScrolledText(frame_bottom, wrap=tk.WORD, height=20, bg="#f5f5f5")
    summary_text.pack(fill="both", expand=True)
    paned.add(frame_bottom)

    # Synchronisation entre les threads
    model_loading_finished = threading.Event()
    model_load_error = [None]

    # Lancement du chargement après initialisation de Tkinter
    root.after(100, start_models_loading)

    # Lancement de l’interface
    root.mainloop()
