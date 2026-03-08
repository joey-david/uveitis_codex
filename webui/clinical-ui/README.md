# UI Clinique (Statique)

Client web statique pour l'API d'inférence UWF.

## Configuration

Éditer `config.js`:

```js
window.UVEITIS_UI_CONFIG = {
  apiBaseUrl: "http://127.0.0.1:18080",
  apiToken: ""
};
```

- `apiBaseUrl`: endpoint API (local, tunnel SSH, ou domaine)
- `apiToken`: laisser vide si auth désactivée côté API

## Lancer en local

```bash
cd webui/clinical-ui
python3 -m http.server 5173
```

Ouvrir `http://127.0.0.1:5173`.

## Fonctionnalités UI

- interface française orientée clinique,
- statut API (online/offline),
- progression par étapes (upload -> preprocessing -> inférence -> rendu),
- indicateurs de transfert et de temps,
- vue image prédite + image prétraitée,
- zoom plein écran au clic sur image (fermeture bouton/clic extérieur/Echap),
- table des prédictions triée par score.

## Dépannage rapide

- `304` sur `index.html`/`config.js`: normal (cache navigateur).
- hard refresh: `Ctrl+Shift+R`.
- si l'API ne répond pas: vérifier `apiBaseUrl` et le tunnel SSH.
