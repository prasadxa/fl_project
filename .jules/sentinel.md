## 2025-02-14 - Admin Endpoints Missing Authentication
**Vulnerability:** The backend administrative endpoints (`/api/admin/*`) were completely unauthenticated, exposing aggregate statistics, user feedback logs, and raw AI parameters.
**Learning:** The frontend SPA routed `/admin` pages locally, but relied on the backend API without sending credentials. A simple `fetch` wrapper update and server-side HTTPBasic authentication cleanly patched the vulnerability without architectural overhauls.
**Prevention:** All administrative or sensitive API routes must explicitly include a dependency (like `Depends(verify_admin)`) enforcing authentication upon implementation.
