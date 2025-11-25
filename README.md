# Volatility & RL Dashboards

Une app Streamlit qui regroupe trois modules :
- **Volatility Tools** : analyse IV, pricing straddles et earnings crush (Yahoo Finance).
- **RL & Finance Lab** : notebooks interactifs et démos RL/quant (pricing, hedging, signals, stratégies, NLP sentiment).
- **AI Trading Bot** : tableau de bord trading/options (Alpaca, pricing avancé, persistance JSON locale).

## Arborescence
- `streamlit_app.py` : point d’entrée qui assemble les modules en onglets.
- `scripts/` : app Volatility Tools.
- `streamlit_app3.py` et `scripts/sources/` : app RL & Finance Lab + datasets.
- `streamlit_appGPT.py` et `scripts/scriptsGPT/` : app AI Trading Bot + scripts/pricing.
- `database/` : JSON locaux (portefeuille, options, forwards, etc.).
- `tests/` : scripts de vérification/fixtures.

## Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement
```bash
streamlit run streamlit_app.py
```
Onglets disponibles : Volatility Tools / RL & Finance Lab / AI Trading Bot.

## Variables d’environnement utiles
- `OPENAI_API_KEY` : pour les appels ChatGPT dans l’onglet AI Trading Bot.
- `APCA_API_KEY_ID` et `APCA_API_SECRET_KEY` : clés Alpaca (paper ou live) si tu actives les calls API.
- `HTTP_PROXY` / `HTTPS_PROXY` : si nécessaire derrière un proxy.

## Données locales
- Les portefeuilles/options/forwards sont stockés dans `database/*.json`. Ils sont créés/écrasés par l’UI.
- Les jeux de données RL/quant sont dans `scripts/sources/` (sous-dossiers data).

## Notes
- L’app charge des dépendances lourdes (torch, tensorflow, plotly…). Prévois quelques minutes d’installation.
- Si tu ne veux pas installer Alpaca/OpenAI, garde les variables non définies : l’UI gère les erreurs mais les fonctions liées ne marcheront pas.
