# Recherche PDF
Application de bureau en Python et Tkinter pour explorer une collection de PDF. Elle retrouve des passages par similarité avec FAISS ou par recherche textuelle, puis peut s'appuyer sur ces passages et les résumés enregistrés pour répondre à une question avec Qwen ou ChatGPT.

## Démarrer sous Windows
### Depuis l'exécutable
Le programme `RecherchePDF.exe` est accessible dans `releases`.
Lors de la première exécution, les fichiers des modèles `Qwen3-Embedding-0.6B` et `Qwen3-0.6B` seront téléchargés dans le dossier `models/`. 

### Depuis le code source
Installez Python avec Tkinter. Depuis la racine du projet, créez un environnement virtuel et installez les dépendances.

```powershell
py -m venv .venv
.\.venv\[Dossier courant]\python.exe -m pip install faiss-cpu torch numpy pypdf transformers tqdm openai huggingface-hub
.\.venv\[Dossier courant]\python.exe .\pdf_search_gui.py
```

### Premier lancement
Copiez vos fichiers PDFs dans le dossier `PDFS/` pour les centraliser. Si vous créez des sous-dossiers, vous devrez les indexer individuellement.
1. Pour utiliser ChatGPT ou générer de nouveaux résumés complets avec OpenAI, renseignez une clé API et un nom de modèle. La configuration est facultative. Cliquez sur **Annuler** ou laissez le champ vide pour continuer avec Qwen, FAISS et la recherche textuelle. Vous pourrez toujours consulter les résumés déjà enregistrés.
2. Attendez la vérification des modèles `Qwen3-Embedding-0.6B` et `Qwen3-0.6B`. Une connexion Internet permet de récupérer les fichiers manquants depuis Hugging Face. Les calculs Qwen s'effectuent sur la machine.
3. Cliquez sur **Sélectionner dossier PDF et indexer**. Le programme indexe les PDF situés directement dans ce dossier. L'index est conservé pour les prochaines sessions.

## Utilisation
| Commande | Résultat |
| --- | --- |
| **dans l'Index** | Retrouve des extraits pertinents avec FAISS, complète les résultats avec une recherche exacte et présente les PDF par thèmes. |
| **avec Qwen** | Génère une réponse locale à partir des extraits de l'index et des résumés disponibles. |
| **avec ChatGPT** | Envoie le contexte documentaire et la question à l'API OpenAI après confirmation. La commande est disponible lorsque la clé et le modèle sont renseignés. |
| **texte Brut** | Cherche les mots saisis dans les PDF d'un dossier et de ses sous-dossiers sans indexation préalable. Les numéros de page des résultats sont cliquables. |
| **Similaires & doublons** | Repère les PDF proches et les quasi-doublons à partir de l'index. |

Depuis les résultats de l'index, vous pouvez ouvrir un PDF, consulter un résumé existant, demander un résumé complet via OpenAI, enregistrer ce résumé ou ajouter le document à la liste **À lire**. La recherche et l'indexation s'exécutent en arrière-plan, une opération à la fois. Les PDF composés uniquement d'images doivent disposer d'une couche de texte pour être recherchés.

## Données locales
L'application crée ces dossiers à côté du script ou, dans la version compilée, à côté de l'exécutable.
- `models/` contient les fichiers Qwen téléchargés.
- `datas/` contient l'index FAISS, les embeddings, les passages extraits, les résumés éventuels, la liste **À lire** et le journal d'erreurs. Le fichier facultatif `datas/bad_words.txt` sert à filtrer les mots-clés des thèmes.
- `PDFS/` sert à ranger des documents. La recherche et l'indexation peuvent aussi utiliser un autre dossier choisi dans l'interface.

Les fichiers `openai_key.txt` et `openai_model.txt` sont stockés dans le dossier de travail depuis lequel le programme est lancé. La clé API y est enregistrée en clair. Les opérations ChatGPT peuvent consommer des crédits API et transmettent à OpenAI le contexte de la question ou le texte extrait du PDF à résumer.

## Licence
Copyright © 2026 Eric Lamette. Le code est distribué sous licence GNU GPL version 3 ou ultérieure. Consultez le [texte de la licence](LICENCE.TXT).
