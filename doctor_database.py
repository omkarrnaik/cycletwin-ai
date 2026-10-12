
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/cycletwin_doctor.db")

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = connect()
    con.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS doctors (
        doctor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor_id INTEGER NOT NULL,
        patient_code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        age INTEGER,
        height REAL,
        weight REAL,
        source TEXT NOT NULL DEFAULT 'Doctor Entry',
        created_at TEXT NOT NULL,
        FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
    );

    CREATE TABLE IF NOT EXISTS assessments (
        assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        assessment_type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        probability REAL NOT NULL,
        risk_band TEXT NOT NULL,
        input_json TEXT NOT NULL,
        shap_json TEXT,
        recommendations_json TEXT,
        trace_json TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    );
    """)
    con.commit()
    con.close()

def ensure_demo_doctor():
    con = connect()
    row = con.execute(
        "SELECT * FROM doctors WHERE email=?",
        ("doctor@cycletwin.demo",)
    ).fetchone()
    if row is None:
        con.execute(
            """INSERT INTO doctors(name,email,password,created_at)
               VALUES(?,?,?,?)""",
            ("Dr. Sharma", "doctor@cycletwin.demo", "1234",
             datetime.now().isoformat(timespec="seconds"))
        )
        con.commit()
    row = con.execute(
        "SELECT * FROM doctors WHERE email=?",
        ("doctor@cycletwin.demo",)
    ).fetchone()
    con.close()
    return dict(row)

def authenticate(email, password):
    con = connect()
    row = con.execute(
        "SELECT * FROM doctors WHERE email=? AND password=?",
        (email.strip(), password)
    ).fetchone()
    con.close()
    return dict(row) if row else None

def add_patient(doctor_id, name, age, height=None, weight=None,
                source="Doctor Entry"):
    con = connect()
    n = con.execute(
        "SELECT COUNT(*) FROM patients WHERE doctor_id=?",
        (doctor_id,)
    ).fetchone()[0] + 1
    code = f"CT-{n:03d}"
    while con.execute(
        "SELECT 1 FROM patients WHERE patient_code=?", (code,)
    ).fetchone():
        n += 1
        code = f"CT-{n:03d}"

    cur = con.execute(
        """INSERT INTO patients
           (doctor_id,patient_code,name,age,height,weight,source,created_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (doctor_id, code, name.strip(), int(age),
         float(height) if height is not None else None,
         float(weight) if weight is not None else None,
         source, datetime.now().isoformat(timespec="seconds"))
    )
    con.commit()
    pid = cur.lastrowid
    con.close()
    return pid

def get_patients(doctor_id):
    con = connect()
    rows = con.execute(
        "SELECT * FROM patients WHERE doctor_id=? ORDER BY patient_id DESC",
        (doctor_id,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]

def get_patient(patient_id, doctor_id):
    con = connect()
    row = con.execute(
        """SELECT * FROM patients
           WHERE patient_id=? AND doctor_id=?""",
        (patient_id, doctor_id)
    ).fetchone()
    con.close()
    return dict(row) if row else None

def save_assessment(patient_id, result, assessment_type):
    p = result["prediction"]
    con = connect()
    cur = con.execute(
        """INSERT INTO assessments
           (patient_id,assessment_type,created_at,probability,risk_band,
            input_json,shap_json,recommendations_json,trace_json)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (patient_id, assessment_type,
         datetime.now().isoformat(timespec="seconds"),
         float(p["probability"]), p["risk_band"],
         json.dumps(result.get("input", {}), default=str),
         json.dumps(result.get("shap_rows", []), default=str),
         json.dumps(result.get("recommendations", []), default=str),
         json.dumps(result.get("trace", []), default=str))
    )
    con.commit()
    aid = cur.lastrowid
    con.close()
    return aid

def get_assessments(patient_id):
    con = connect()
    rows = con.execute(
        """SELECT * FROM assessments
           WHERE patient_id=?
           ORDER BY datetime(created_at), assessment_id""",
        (patient_id,)
    ).fetchall()
    con.close()

    result = []
    for row in rows:
        d = dict(row)
        d["input"] = json.loads(d.pop("input_json") or "{}")
        d["shap_rows"] = json.loads(d.pop("shap_json") or "[]")
        d["recommendations"] = json.loads(
            d.pop("recommendations_json") or "[]"
        )
        d["trace"] = json.loads(d.pop("trace_json") or "[]")
        result.append(d)
    return result

def get_latest_assessment(patient_id):
    rows = get_assessments(patient_id)
    return rows[-1] if rows else None

def assessment_dataframe(patient_id):
    rows = get_assessments(patient_id)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "assessment_id": r["assessment_id"],
            "type": r["assessment_type"],
            "date": r["created_at"],
            "probability": r["probability"],
            "risk_band": r["risk_band"],
        }
        for r in rows
    ])
