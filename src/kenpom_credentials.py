"""Load KenPom credentials from env or repo credentials file."""
from __future__ import annotations

import json
import os
from typing import Tuple


def load_kenpom_credentials() -> Tuple[str, str]:
    email = os.getenv("KENPOM_EMAIL", "").strip()
    password = os.getenv("KENPOM_PASSWORD", "").strip()
    if email and password:
        return email, password

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(root, "credentials.json")
    if os.path.exists(creds_path):
        with open(creds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        email = str(data.get("kenpom_email", "")).strip()
        password = str(data.get("kenpom_password", "")).strip()
        if email and password:
            return email, password

    raise ValueError(
        "Missing KenPom credentials. Set KENPOM_EMAIL/KENPOM_PASSWORD or fill credentials.json."
    )
