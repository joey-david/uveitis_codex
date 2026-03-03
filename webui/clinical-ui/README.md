# Clinical UI

Static web client for the inference API.

## Configure

Edit `config.js`:

```js
window.UVEITIS_UI_CONFIG = {
  apiBaseUrl: "https://your-api-host.example",
  apiToken: ""
};
```

## Serve locally

```bash
cd webui/clinical-ui
python3 -m http.server 5173
```

Open `http://127.0.0.1:5173`.
