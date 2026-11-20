---
title: Got Chatbot Frontend
short_description: Frontend du chatbot FAQ-RAG
emoji: 🐨
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
python_version: '3.12'
app_file: app.py
pinned: false
---

# Chatbot FAQ + RAG - Interface Gradio

Application Gradio permettant de tester le chatbot de Guild Open Tech. Elle interroge l'API backend [`faq-rag-backend-api`](https://huggingface.co/spaces/Loren/got-chatbot-backend), qui répond depuis la FAQ quand la question y correspond, et sinon génère une réponse par RAG. L'interface affiche la réponse, sa source, les candidats FAQ et les extraits documentaires, et permet de régler tous les paramètres de l'appel.

## Caractéristiques

- 💬 **Interface de chat** : saisie d'une question, historique des échanges dans un `Chatbot`
- 🏷️ **Source de la réponse** : FAQ, FAQ validée par le juge LLM, ou RAG, avec le meilleur score FAQ et l'avis du juge
- 🔎 **Candidats FAQ** : formulations candidates avec leurs scores (similarité, cross-encoder), classées par score cross-encoder
- 📚 **Extraits RAG** : extraits documentaires utilisés pour la génération (titre, auteur, score, aperçu)
- ⚙️ **Paramètres avancés** : les 8 paramètres de l'appel, dans un accordéon replié par défaut
- 📃 **Contenu de la FAQ** : FAQ complète, groupée par catégorie, dans un accordéon (via `GET /api/v1/faq`)
- 👍👎 **Feedback** : retour utilisateur (OK / KO + commentaire) envoyé au backend
- 📈 **Observabilité** : gérée entièrement par le backend (Langfuse) ; le frontend ne fait que transmettre le `trace_id`
- ⏳ **Réveil du backend** : si le Space backend est endormi, l'application le réveille, affiche un bandeau d'attente et bloque la saisie jusqu'à ce qu'il réponde (voir [Réveil du Space backend](#réveil-du-space-backend))

## Architecture

```
app.py                  # Application Gradio (UI, appels à l'API, feedback)
├── models.py           # Modèles Pydantic, copie du contrat de l'API backend
├── config.py           # Valeurs par défaut des paramètres, URL de l'API, port
└── requirements.txt    # Dépendances Python
```

Flux d'une requête :

1. L'utilisateur saisit une question et valide
2. `submit_query` vérifie la cohérence des seuils (seuil du juge ≤ seuil d'acceptation), puis envoie un `POST /api/v1/ask` avec la question, les 8 paramètres et l'identifiant de session (si le backend dort, il est d'abord réveillé, voir [Réveil du Space backend](#réveil-du-space-backend))
3. Le backend cherche dans la FAQ (bi-encoder puis cross-encoder), consulte éventuellement le juge LLM, ou bascule sur le RAG ; il trace l'ensemble dans Langfuse et renvoie le `trace_id`
4. L'interface affiche la réponse avec sa source, les candidats FAQ et les extraits RAG
5. Les boutons 👍 / 👎 envoient un `POST /api/v1/feedback` rattaché au `trace_id`

## Installation locale

### Prérequis
- Python 3.12
- pip ou conda

### Étapes

1. **Cloner le repository**
```bash
git clone <url>
cd chatbot-faq-rag-frontend
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement** : créer un fichier `.env` (voir [Configuration](#configuration))

5. **Lancer l'application**
```bash
python app.py
```

L'application sera accessible à `http://localhost:7861` (le port 7860 est celui du backend en local ; il est modifiable avec `GRADIO_SERVER_PORT`).

## Utilisation

1. (Optionnel) Ouvrir l'accordéon **📃 Contenu de la FAQ** pour consulter les questions disponibles
2. Saisir une question dans le champ **Question** et valider avec Entrée (si un bandeau ⏳ indique que le backend démarre, le champ est désactivé : attendre qu'il disparaisse)
3. Lire la réponse dans le chat ; la source (FAQ, FAQ validée par le juge, RAG) et le meilleur score FAQ sont indiqués sous la réponse
4. Consulter les accordéons **🔎 Candidats FAQ** et **📚 Extraits RAG**
5. Donner son avis avec **👍 Correct** / **👎 Incorrect** (commentaire optionnel)
6. **Clear** réinitialise la conversation, la session et les paramètres

### Paramètres (accordéon ⚙️ Paramètres avancés)

| Paramètre | Variable | Description | Défaut |
|---|---|---|---|
| FAQ Top K | `FAQ_TOP_K` | Nombre de candidats FAQ (1 par thème), 1 à 20 | `3` |
| Seuil de similarité FAQ | `FAQ_SIMILARITY_THRESHOLD` | Similarité cosinus minimale du bi-encoder | `0.85` |
| Seuil d'acceptation FAQ | `FAQ_ACCEPT_THRESHOLD` | Score cross-encoder à partir duquel la réponse FAQ est renvoyée directement | `0.96` |
| Seuil du juge LLM | `FAQ_JUDGE_THRESHOLD` | Score à partir duquel le juge LLM est consulté ; en dessous, RAG. Doit être ≤ seuil d'acceptation | `0.60` |
| Top K | `TOP_K` | Nombre d'extraits documentaires RAG, 1 à 20 | `3` |
| Seuil de similarité | `SIMILARITY_THRESHOLD` | Score minimal d'un extrait documentaire | `0.4` |
| Use MMR | `USE_MMR` | Diversification des extraits (Maximal Marginal Relevance) | `true` |
| Lambda MMR | `LAMBDA_MULT` | 0 = diversité, 1 = pertinence ; ignoré (envoyé à `null`) si MMR est désactivé | `0.55` |

Le bouton **↺ Valeurs par défaut** remet les paramètres aux valeurs de `config.py`.

## Configuration

### Variables d'environnement

Créer un fichier `.env` (toutes les variables sont optionnelles) :

```env
# URL de l'API backend (en local : http://127.0.0.1:7860)
API_URL=https://loren-got-chatbot-backend.hf.space
# Délai maximal d'une requête /ask, en secondes
REQUEST_TIMEOUT=90
# Nom de la trace Langfuse créée par le backend
TRACE_NAME=chatbot-frontend
# Port Gradio en local
GRADIO_SERVER_PORT=7861

# Valeurs par défaut des paramètres (voir le tableau ci-dessus)
FAQ_TOP_K=3
FAQ_SIMILARITY_THRESHOLD=0.85
FAQ_ACCEPT_THRESHOLD=0.96
FAQ_JUDGE_THRESHOLD=0.60
TOP_K=3
USE_MMR=true
LAMBDA_MULT=0.55
SIMILARITY_THRESHOLD=0.4

# Réveil du Space backend endormi (optionnel)
HF_SPACE_ID=Loren/got-chatbot-backend
HF_TOKEN=hf_...
```

Aucune clé Langfuse n'est nécessaire côté frontend.

## API consommée

| Endpoint | Usage |
|---|---|
| `GET /api/v1/faq` | Contenu de la FAQ affiché dans l'accordéon, chargé à l'ouverture de la page |
| `POST /api/v1/ask` | Envoie la question et les paramètres ; retourne la réponse, sa source, les candidats FAQ, les extraits RAG et le `trace_id` |
| `POST /api/v1/feedback` | Enregistre le feedback utilisateur sur la trace |

### Réveil du Space backend

Sur Hugging Face, le Space backend s'endort après une période d'inactivité et répond alors HTTP 503. Les appels `GET /api/v1/faq` (à l'ouverture de la page) et `POST /api/v1/ask` (à chaque question) passent par `_with_wakeup`, qui :

1. demande un redémarrage du Space via l'API Hugging Face (`_restart_space`, une seule fois par appel ; nécessite `HF_SPACE_ID` et `HF_TOKEN` avec droit d'écriture, sinon un avertissement est loggé et l'application se contente d'attendre)
2. réessaie toutes les 10 s, pendant 5 minutes au maximum
3. pendant l'attente, affiche sous le titre un bandeau « ⏳ Le backend est en veille, démarrage en cours… » avec le temps écoulé, et désactive le champ **Question**
4. une fois le backend disponible, masque le bandeau et réactive le champ ; la question en attente est alors envoyée automatiquement

Cas d'arrêt et de reprise :

- **FAQ** : en cas d'échec, un message d'erreur remplace le contenu de la FAQ.
- **`/ask`** : seuls les cas où la requête n'a pas été traitée (HTTP 503, erreur de connexion) sont réessayés. Un dépassement de `REQUEST_TIMEOUT` n'est pas réessayé, pour ne pas relancer une génération déjà en cours. En cas d'échec, le champ est réactivé avec la question conservée et l'erreur est affichée.

## Hugging Face

L'application est déployée sur Hugging Face Spaces (SDK Gradio 5.49.1, Python 3.12, point d'entrée `app.py`).

Dans l'interface du Space, "Settings" → "Variables and secrets" permet de définir les variables d'environnement (`API_URL`, paramètres par défaut, `HF_TOKEN`). Les logs de build et d'exécution sont consultables depuis l'onglet "Logs".

## Logs

- 🤖 Démarrage (URL de l'API)
- 📃 Chargement du contenu de la FAQ
- 🔍 Questions envoyées à l'API, avec les paramètres
- ✅ Réponses (source, meilleur score FAQ, `trace_id`) et feedbacks enregistrés
- ⏳🔄 Réveil du Space backend (API injoignable, redémarrage demandé, nouvel essai)
- ⚠️/❌ Avertissements et erreurs (API indisponible, échec du redémarrage, paramètres refusés)
