import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data/cycletwin.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            probability REAL NOT NULL,
            prediction INTEGER NOT NULL,
            risk_band TEXT NOT NULL,
            input_json TEXT NOT NULL,
            shap_json TEXT,
            recommendations_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """)
        con.commit()

def get_or_create_user(name):
    with get_connection() as con:
        row = con.execute(
            "SELECT user_id FROM users WHERE name = ? ORDER BY user_id LIMIT 1",
            (name,)
        ).fetchone()
        if row:
            return row[0]
        cur = con.execute(
            "INSERT INTO users(name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat(timespec="seconds"))
        )
        con.commit()
        return cur.lastrowid

def save_assessment(user_id, result):
    p = result["prediction"]
    shap_rows = result.get("shap_rows", [])
    recs = result.get("recommendations", [])
    with get_connection() as con:
        cur = con.execute("""
        INSERT INTO assessments
        (user_id, created_at, probability, prediction, risk_band,
         input_json, shap_json, recommendations_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            datetime.now().isoformat(timespec="seconds"),
            float(p["probability"]),
            int(p["label"]),
            p["risk_band"],
            json.dumps(result["input"], default=str),
            json.dumps(shap_rows, default=str),
            json.dumps(recs, default=str)
        ))
        con.commit()
        return cur.lastrowid

def get_history(user_id, limit=30):
    with get_connection() as con:
        rows = con.execute("""
        SELECT assessment_id, created_at, probability, prediction, risk_band
        FROM assessments
        WHERE user_id = ?
        ORDER BY assessment_id DESC
        LIMIT ?
        """, (user_id, limit)).fetchall()
    cols = ["assessment_id", "created_at", "probability", "prediction", "risk_band"]
    return pd.DataFrame(rows, columns=cols).sort_values("created_at")

def get_latest(user_id):
    with get_connection() as con:
        row = con.execute("""
        SELECT assessment_id, created_at, probability, prediction, risk_band,
               input_json, shap_json, recommendations_json
        FROM assessments
        WHERE user_id = ?
        ORDER BY assessment_id DESC LIMIT 1
        """, (user_id,)).fetchone()
    return row
