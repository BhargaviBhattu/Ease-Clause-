## Purpose
Provide concise, actionable guidance for AI coding agents working in this repository so they can be immediately productive.

## Quick start (what to run)
- **Install deps:** `pip install -r requirements.txt` (uses `requirements.txt` in repo root).
- **Run locally:** `streamlit run main.py` — the app is a Streamlit multi-page app; `main.py` is the landing page.
- **Initialize DB / Admin:** call `backend.setup()` or run `create_admin.py` to initialize `users.db` and create the default admin.

## High-level architecture (big picture)
- **UI / pages:** the Streamlit pages live in the `pages/` directory (examples: [pages/Auth.py](pages/Auth.py), [pages/Main_App.py](pages/Main_App.py), [pages/_Create_Account.py](pages/_Create_Account.py)). Navigation uses `st.switch_page("pages/....py")` and relies on `st.session_state.user` for auth state.
- **Backend / logic:** `backend.py` contains database schema, user management, document storage helpers, and a simple model-based `simplify_text` function using `transformers` (T5).
- **Utilities / models:** `utils/simplifier.py` is the more fully-featured simplifier: rule-based preprocessing, chunking, model-loading with `lru_cache`, and model fallback behavior. Callers in the UI may call either `backend.simplify_text` or `utils.simplifier.simplify_text` (note: both exist).
- **Data storage:** there are two DB patterns:
  - `users.db` (user accounts, documents, simplifications) — created/used by `backend.py` and several pages.
  - `data/simplification_logs.db` (analytics logs) — used by `utils/db_helper.py`.

## Important, discoverable conventions & gotchas
- **Streamlit navigation:** pages rely on `st.session_state.user` (a dict with at least `email`) to gate access. If `user` not set, pages call `st.stop()`.
- **Password hashing & storage:** code uses `bcrypt` and stores hashed passwords as BLOB. Column names sometimes differ between modules (`password` in `backend.py` vs `password_hash` in `pages/_Create_Account.py`) — look at [backend.py](backend.py) and [pages/_Create_Account.py](pages/_Create_Account.py) when modifying auth flows.
- **Multiple model loaders:** `backend.py` loads a small `t5-small` model at import time; `utils/simplifier.py` prefers `google/flan-t5-large` (environment override via `SIMPLIFIER_MODEL`) and includes a fallback to `facebook/bart-large-cnn`. Be careful to unify model usage to avoid duplicate large downloads in memory.
- **DB path differences:** some modules use `DB_PATH = "users.db"` while `utils/db_helper.py` uses `data/simplification_logs.db` — update paths consistently if migrating or containerizing.
- **File uploads parsing:** `pages/Main_App.py` supports `.txt`, `.docx` (requires `python-docx`), and `.pdf` (requires `PyPDF2`) with explicit runtime checks that raise helpful runtime errors if optional libs are missing.

## Key entry points and examples
- Simplify text (high-level):

```py
from utils.simplifier import simplify_text
out = simplify_text(long_text, level="Intermediate")
```

- Create/init DB and admin (example):

```py
import backend
backend.setup()  # creates users.db and default admin
```

- Run Streamlit app (development):

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Files to inspect first when debugging/ changing behavior
- Authentication and user schema: [backend.py](backend.py), [pages/_Create_Account.py](pages/_Create_Account.py), [pages/Auth.py](pages/Auth.py)
- Document ingestion and storage: [pages/Main_App.py](pages/Main_App.py)
- Model logic and configuration: [utils/simplifier.py](utils/simplifier.py) and [backend.py](backend.py)
- Lightweight analytics/logging: [utils/db_helper.py](utils/db_helper.py)

## Debugging tips specific to this project
- If simplification returns raw or empty output, check model selection and device (GPU vs CPU). `utils/simplifier.py` logs model-load attempts and falls back on BART.
- For file parsing errors, the UI raises human-readable runtime errors (e.g., missing `python-docx`), so reproduce the upload locally and inspect the stack trace in console.
- To inspect DB contents quickly, use `sqlite3 users.db` or a GUI SQLite browser; schema is defined in `backend.init_db()`.

## When editing code, follow these minimal heuristics
- Prefer `utils/simplifier.py` for model-related work (it contains chunking, fallback, and caching). If you change model behavior, update both `backend.py` and `utils/simplifier.py` or consolidate to one place.
- Preserve Streamlit session keys: `st.session_state.user` is the single source of truth for logged-in users.
- Avoid heavy model downloads during import time in web-facing modules — prefer lazy loading or cached loaders (`lru_cache`) as in `utils/simplifier.py`.

## Next steps / TODO for maintainers (non-aspirational, concrete)
- Consider unifying model loading into `utils/simplifier.py` and removing the eager model load from `backend.py` to reduce memory and startup time.
- Normalize the user table schema (pick `password` or `password_hash`) and update places that assume different column names.

---
If anything here is unclear or you want the file to include more/less detail (examples, exact environment variables, or CI instructions), tell me which areas to expand. 
