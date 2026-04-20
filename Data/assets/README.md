# Branding assets

Drop a `logo.png` here and it will automatically appear as a subtle
watermark behind the chat window (see `App/dashboard.py`). Any reasonable
PNG works; the dashboard fades it to ~5% opacity and centers it, so a
flat transparent PNG at ~800×800 gives the cleanest result.

If `logo.png` is missing the dashboard renders without a watermark — no
error, no placeholder. That's deliberate so this directory stays empty in
fresh clones.
