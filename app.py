import hashlib
import sqlite3
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="IoT-Driven Aquatic Airbag Safety Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "airbag_platform.db"


# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 0.9rem;
    }
    .soft-box {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
    }
    .auth-card {
        background: white;
        padding: 1.4rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .role-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 0.5rem;
        color: white;
    }
    .role-public { background: linear-gradient(135deg, #0284c7, #2563eb); }
    .role-supervisor { background: linear-gradient(135deg, #7c3aed, #9333ea); }
    .role-rescue { background: linear-gradient(135deg, #dc2626, #f97316); }
    .role-research { background: linear-gradient(135deg, #059669, #10b981); }
    .small-note {
        color: #64748b;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hash_password(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def role_badge(role: str) -> str:
    cls_map = {
        "Public User": "role-public",
        "Supervisor": "role-supervisor",
        "Rescue Team": "role-rescue",
        "Researcher": "role-research",
    }
    cls = cls_map.get(role, "role-public")
    return f'<span class="role-badge {cls}">{role}</span>'


def ema(series: pd.Series, alpha: float = 0.25) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def safe_auc(y_true, y_prob):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def safe_metric_delta(current, previous):
    if previous is None or pd.isna(previous):
        return None
    return round(float(current) - float(previous), 3)


def create_ticket_id():
    return f"ABG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def flotation_liters(user_mass_kg: float, safety_factor: float = 1.1) -> float:
    return float(user_mass_kg) * float(safety_factor)


# ============================================================
# DATABASE WITH MIGRATION
# ============================================================
class DB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_db()

    def execute(self, query, params=()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        self.conn.commit()
        return cur

    def query_df(self, query, params=()):
        return pd.read_sql_query(query, self.conn, params=params)

    def table_exists(self, table):
        q = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return q is not None

    def get_columns(self, table):
        if not self.table_exists(table):
            return []
        cur = self.execute(f"PRAGMA table_info({table})")
        return [row["name"] for row in cur.fetchall()]

    def add_column_if_missing(self, table, column, col_type):
        if column not in self.get_columns(table):
            self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def init_db(self):
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password_hash TEXT,
                display_name TEXT,
                role TEXT,
                created_at TEXT
            )
            """
        )

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                session_token TEXT UNIQUE,
                created_at TEXT,
                expires_at TEXT
            )
            """
        )

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                username TEXT,
                role TEXT,
                event_type TEXT,
                status TEXT,
                message TEXT
            )
            """
        )

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                scenario_name TEXT,
                severity TEXT,
                depth REAL,
                heart_rate REAL,
                body_temp REAL,
                spo2 REAL,
                motion_index REAL,
                fusion_risk REAL,
                prediction_prob REAL,
                health_score REAL,
                location_name TEXT,
                latitude REAL,
                longitude REAL,
                assigned_team TEXT,
                notes TEXT
            )
            """
        )

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT UNIQUE,
                created_time TEXT,
                created_by TEXT,
                incident_time TEXT,
                severity TEXT,
                issue_type TEXT,
                description TEXT,
                status TEXT,
                assigned_to TEXT,
                action_note TEXT
            )
            """
        )

        self.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                username TEXT,
                title TEXT,
                note TEXT
            )
            """
        )

        for col, typ in [
            ("prediction_prob", "REAL"),
            ("fusion_risk", "REAL"),
            ("health_score", "REAL"),
            ("latitude", "REAL"),
            ("longitude", "REAL"),
            ("scenario_name", "TEXT"),
        ]:
            self.add_column_if_missing("event_logs", col, typ)

        self.seed_users()

    def seed_users(self):
        users = [
            ("public1", hash_password("public123"), "Public User", "Public User"),
            ("supervisor1", hash_password("supervisor123"), "Operations Supervisor", "Supervisor"),
            ("rescue1", hash_password("rescue123"), "Rescue Team Alpha", "Rescue Team"),
            ("research1", hash_password("research123"), "Research Analyst", "Researcher"),
        ]
        for username, pwd, display_name, role in users:
            self.execute(
                """
                INSERT OR IGNORE INTO users (username, password_hash, display_name, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, pwd, display_name, role, now_str()),
            )

    def authenticate(self, username, password):
        cur = self.execute(
            "SELECT * FROM users WHERE username=? AND password_hash=?",
            (username, hash_password(password)),
        )
        return cur.fetchone()

    def create_session(self, username, hours=24):
        token = hash_password(f"{username}_{uuid.uuid4()}_{time.time()}")
        created = datetime.now()
        expires = created + timedelta(hours=hours)
        self.execute(
            """
            INSERT INTO sessions (username, session_token, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, token, created.isoformat(), expires.isoformat()),
        )
        return token

    def get_session(self, token):
        cur = self.execute("SELECT * FROM sessions WHERE session_token=?", (token,))
        row = cur.fetchone()
        if row is None:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now():
            self.execute("DELETE FROM sessions WHERE session_token=?", (token,))
            return None
        return row

    def delete_session(self, token):
        self.execute("DELETE FROM sessions WHERE session_token=?", (token,))

    def log_event(
        self,
        username,
        role,
        event_type,
        status,
        message,
        prediction_prob=None,
        fusion_risk=None,
        health_score=None,
        latitude=None,
        longitude=None,
        scenario_name=None,
    ):
        self.execute(
            """
            INSERT INTO event_logs (
                ts, username, role, event_type, status, message,
                prediction_prob, fusion_risk, health_score,
                latitude, longitude, scenario_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str(),
                username,
                role,
                event_type,
                status,
                message,
                prediction_prob,
                fusion_risk,
                health_score,
                latitude,
                longitude,
                scenario_name,
            ),
        )

    def add_incident(
        self,
        scenario_name,
        severity,
        depth,
        heart_rate,
        body_temp,
        spo2,
        motion_index,
        fusion_risk,
        prediction_prob,
        health_score,
        location_name,
        latitude,
        longitude,
        assigned_team="Team Alpha",
        notes="",
    ):
        self.execute(
            """
            INSERT INTO incidents (
                ts, scenario_name, severity, depth, heart_rate, body_temp, spo2,
                motion_index, fusion_risk, prediction_prob, health_score,
                location_name, latitude, longitude, assigned_team, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str(),
                scenario_name,
                severity,
                depth,
                heart_rate,
                body_temp,
                spo2,
                motion_index,
                fusion_risk,
                prediction_prob,
                health_score,
                location_name,
                latitude,
                longitude,
                assigned_team,
                notes,
            ),
        )


db = DB(DB_PATH)


# ============================================================
# SESSION / LOGIN
# ============================================================
def init_session():
    defaults = {
        "authenticated": False,
        "username": None,
        "display_name": None,
        "role": None,
        "token": None,
        "sim_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def restore_session():
    params = st.query_params
    token = params.get("token", None)
    if token:
        row = db.get_session(token)
        if row is not None:
            user_df = db.query_df("SELECT * FROM users WHERE username=?", (row["username"],))
            if not user_df.empty:
                st.session_state.authenticated = True
                st.session_state.username = row["username"]
                st.session_state.display_name = user_df.iloc[0]["display_name"]
                st.session_state.role = user_df.iloc[0]["role"]
                st.session_state.token = token
                return True
    return False


def login_user(username, password):
    row = db.authenticate(username, password)
    if row is None:
        return False
    token = db.create_session(username)
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.display_name = row["display_name"]
    st.session_state.role = row["role"]
    st.session_state.token = token
    st.query_params["token"] = token
    db.log_event(username, row["role"], "login", "success", "User logged in")
    return True


def logout_user():
    token = st.session_state.get("token")
    if token:
        db.delete_session(token)
    db.log_event(
        st.session_state.get("username", "unknown"),
        st.session_state.get("role", "unknown"),
        "logout",
        "success",
        "User logged out",
    )
    for key in ["authenticated", "username", "display_name", "role", "token"]:
        st.session_state[key] = None if key != "authenticated" else False
    st.query_params.clear()
    st.rerun()


init_session()
if not st.session_state.authenticated:
    restore_session()

if not st.session_state.authenticated:
    st.markdown('<div class="main-title">🌊 IoT-Driven Aquatic Airbag Safety Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Advanced research-grade dashboard for multi-sensor distress detection, airbag deployment decision support, and aquatic rescue operations.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.1, 1.6, 1.1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("### Secure Role-Based Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if login_user(username.strip(), password):
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown("#### Demo Credentials")
        st.code(
            """Public User
username: public1
password: public123

Supervisor
username: supervisor1
password: supervisor123

Rescue Team
username: rescue1
password: rescue123

Researcher
username: research1
password: research123"""
        )
        st.markdown(
            """
            <div class="small-note">
            This prototype uses SQLite-backed demo authentication and persistent URL-session tokens.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# DATA GENERATION
# ============================================================
def generate_airbag_demo_data(
    scenario_name="Normal Swim",
    n=300,
    turbulence=0.35,
    dropout=0.05,
    water_temp_c=24.0,
):
    rng = np.random.default_rng()
    t = np.arange(n)

    depth = 0.30 + 0.12 * np.sin(2 * np.pi * t / 40.0) + 0.05 * rng.normal(size=n)
    heart_rate = 88 + 4 * np.sin(2 * np.pi * t / 55.0) + 3 * rng.normal(size=n)
    spo2 = 98 + 0.35 * rng.normal(size=n)
    motion = 0.22 + 0.12 * np.abs(np.sin(2 * np.pi * t / 18.0)) + 0.08 * rng.normal(size=n)
    body_temp = np.full(n, 36.7)
    pressure_kpa = 101.3 + depth * 9.81 + turbulence * rng.normal(0, 0.8, size=n)
    water_contact = np.zeros(n)
    panic_button = np.zeros(n)
    child_mode = np.zeros(n)

    if scenario_name == "Vigorous Swim":
        motion += 0.45 + 0.18 * np.abs(np.sin(2 * np.pi * t / 7.0))
        heart_rate += 18 + 7 * np.sin(2 * np.pi * t / 20.0)
        depth += 0.10 * np.sin(2 * np.pi * t / 8.0)
        water_contact[:] = 1

    elif scenario_name == "Panic Struggle":
        motion += 0.70 + 0.30 * np.abs(np.sin(2 * np.pi * t / 5.0))
        heart_rate += 28 + 12 * np.abs(np.sin(2 * np.pi * t / 11.0))
        depth += 0.20 + 0.20 * np.sin(2 * np.pi * t / 10.0)
        water_contact[:] = 1
        panic_button[-50:] = rng.integers(0, 2, size=50)

    elif scenario_name == "Silent Distress":
        motion += 0.03 * rng.normal(size=n)
        heart_rate += np.linspace(10, -20, n)
        spo2 -= np.linspace(0.0, 5.5, n)
        depth += np.linspace(0.1, 1.5, n)
        water_contact[:] = 1

    elif scenario_name == "Cold Shock":
        heart_rate += 34 * np.exp(-t / 60.0) + 8 * rng.normal(size=n)
        body_temp = 36.7 - 0.010 * t - 0.04 * rng.normal(size=n)
        depth += 0.10 + 0.08 * np.sin(2 * np.pi * t / 21.0)
        motion += 0.25 + 0.12 * np.abs(np.sin(2 * np.pi * t / 9.0))
        water_contact[:] = 1

    elif scenario_name == "Near Drowning":
        depth += np.linspace(0.2, 2.4, n)
        heart_rate += np.piecewise(
            t,
            [t < 80, (t >= 80) & (t < 180), t >= 180],
            [
                lambda x: 18 + 4 * np.sin(x / 10.0),
                lambda x: 32 - 0.05 * (x - 80),
                lambda x: -5 - 0.10 * (x - 180),
            ],
        )
        spo2 -= np.linspace(0.0, 9.5, n)
        motion += np.concatenate(
            [
                0.20 + 0.30 * np.abs(np.sin(2 * np.pi * np.arange(80) / 6.0)),
                0.55 + 0.20 * np.abs(np.sin(2 * np.pi * np.arange(100) / 4.0)),
                0.10 + 0.05 * rng.normal(size=n - 180),
            ]
        )
        body_temp = 36.6 - 0.006 * t
        water_contact[:] = 1
        child_mode[:] = 1

    elif scenario_name == "Child Fall":
        depth += np.linspace(0.3, 1.8, n)
        heart_rate += 20 + 10 * np.abs(np.sin(2 * np.pi * t / 9.0))
        motion += 0.85 + 0.25 * np.abs(np.sin(2 * np.pi * t / 4.0))
        spo2 -= np.linspace(0.0, 6.0, n)
        water_contact[:] = 1
        child_mode[:] = 1
        panic_button[-40:] = 1

    depth += turbulence * 0.15 * rng.normal(size=n)
    pressure_kpa += turbulence * 1.2 * rng.normal(size=n)
    motion += turbulence * 0.10 * rng.normal(size=n)

    depth = np.clip(depth, 0.0, None)
    heart_rate = np.clip(heart_rate, 35, 190)
    spo2 = np.clip(spo2, 70, 100)
    motion = np.clip(motion, 0, 2.2)
    body_temp = np.clip(body_temp, 24, 37.2)
    water_contact = np.clip(water_contact, 0, 1)
    panic_button = np.clip(panic_button, 0, 1)
    child_mode = np.clip(child_mode, 0, 1)

    heart_rate = heart_rate.astype(float)
    spo2 = spo2.astype(float)
    heart_rate[rng.random(n) < dropout] = np.nan
    spo2[rng.random(n) < dropout / 2] = np.nan

    df = pd.DataFrame(
        {
            "t": t,
            "depth_m": depth,
            "pressure_kpa": pressure_kpa,
            "heart_rate": heart_rate,
            "spo2": spo2,
            "motion_index": motion,
            "body_temp_c": body_temp,
            "water_temp_c": np.full(n, water_temp_c),
            "water_contact": water_contact,
            "panic_button": panic_button,
            "child_mode": child_mode,
        }
    )

    df["heart_rate"] = df["heart_rate"].interpolate().bfill().ffill()
    df["spo2"] = df["spo2"].interpolate().bfill().ffill()

    df["depth_s"] = ema(df["depth_m"], 0.25)
    df["pressure_s"] = ema(df["pressure_kpa"], 0.25)
    df["hr_s"] = ema(df["heart_rate"], 0.20)
    df["spo2_s"] = ema(df["spo2"], 0.20)
    df["motion_s"] = ema(df["motion_index"], 0.25)

    df["d_depth"] = df["depth_s"].diff().fillna(0)
    df["d_hr"] = df["hr_s"].diff().fillna(0)
    df["depth_var_10"] = df["depth_m"].rolling(10, min_periods=1).var().fillna(0)
    df["motion_energy_10"] = df["motion_index"].rolling(10, min_periods=1).mean().fillna(0)
    df["spo2_drop"] = 100 - df["spo2_s"]
    df["thermal_risk"] = np.clip((35.5 - df["body_temp_c"]) / 8.0, 0, 1.0)
    df["cold_shock_index"] = np.clip((df["hr_s"] - 115) / 50.0, 0, 1.0) * np.clip((24 - df["water_temp_c"]) / 10.0, 0, 1.0)
    df["submerge_ratio_20"] = (df["depth_s"] > 0.8).rolling(20, min_periods=1).mean()

    distress_score = (
        (df["depth_s"] > 1.0).astype(int)
        + (df["spo2_s"] < 93).astype(int)
        + (df["motion_s"] > 0.95).astype(int)
        + (df["hr_s"] > 130).astype(int)
        + (df["hr_s"] < 55).astype(int)
        + (df["thermal_risk"] > 0.2).astype(int)
        + (df["panic_button"] > 0).astype(int)
    )
    if scenario_name in ["Near Drowning", "Child Fall", "Silent Distress", "Panic Struggle", "Cold Shock"]:
        distress_score += (df["water_contact"] > 0).astype(int)

    df["distress_label"] = (distress_score >= 2).astype(int)

    if df["distress_label"].nunique() < 2:
        if scenario_name == "Normal Swim":
            idx = np.linspace(int(0.80 * len(df)), len(df) - 1, max(3, len(df) // 25), dtype=int)
            df.loc[idx, "distress_label"] = 1
        else:
            idx = np.linspace(0, int(0.20 * len(df)), max(3, len(df) // 25), dtype=int)
            df.loc[idx, "distress_label"] = 0

    horizon = 20
    fut = np.zeros(len(df), dtype=int)
    labels = df["distress_label"].to_numpy()
    for i in range(len(df)):
        fut[i] = int(labels[i : min(len(df), i + horizon)].max() > 0)
    df["future_distress"] = fut

    if df["future_distress"].nunique() < 2:
        df.loc[: max(2, len(df) // 15), "future_distress"] = 0
        df.loc[len(df) // 2 :, "future_distress"] = 1

    return df


# ============================================================
# MODELING
# ============================================================
def feature_columns():
    return [
        "depth_s",
        "pressure_s",
        "hr_s",
        "spo2_s",
        "motion_s",
        "d_depth",
        "d_hr",
        "depth_var_10",
        "motion_energy_10",
        "thermal_risk",
        "spo2_drop",
        "cold_shock_index",
        "submerge_ratio_20",
        "water_contact",
        "panic_button",
        "child_mode",
    ]


def dummy_prob_from_signal(X_df):
    p_depth = np.clip(X_df["depth_s"] / 1.5, 0, 1)
    p_hr_high = np.clip((X_df["hr_s"] - 115) / 45.0, 0, 1)
    p_hr_low = np.clip((60 - X_df["hr_s"]) / 25.0, 0, 1)
    p_spo2 = np.clip((96 - X_df["spo2_s"]) / 12.0, 0, 1)
    p_motion = np.clip((X_df["motion_s"] - 0.55) / 0.80, 0, 1)
    p_temp = np.clip(X_df["thermal_risk"], 0, 1)
    return np.clip(
        0.24 * p_depth
        + 0.18 * np.maximum(p_hr_high, p_hr_low)
        + 0.18 * p_spo2
        + 0.16 * p_motion
        + 0.12 * p_temp
        + 0.06 * X_df["panic_button"]
        + 0.06 * X_df["water_contact"],
        0,
        1,
    )


class DummyBinaryModel:
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            p = dummy_prob_from_signal(X)
        else:
            p = np.full(len(X), 0.5)
        p = np.asarray(p, dtype=float)
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_binary_model(X, y):
    if len(np.unique(y)) < 2:
        return DummyBinaryModel()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X, y)
    return model


def fit_models(df):
    feats = feature_columns()
    X = df[feats].copy()
    y_now = df["distress_label"].astype(int)
    y_future = df["future_distress"].astype(int)

    model_now = train_binary_model(X, y_now)
    model_future = train_binary_model(X, y_future)

    rf = None
    rf_acc = np.nan
    if len(np.unique(y_now)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_now, test_size=0.30, random_state=42, stratify=y_now
        )
        rf = RandomForestClassifier(
            n_estimators=140,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))

    scaler_iso = StandardScaler()
    X_iso = scaler_iso.fit_transform(X)
    iso = IsolationForest(n_estimators=120, contamination=0.08, random_state=42)
    iso.fit(X_iso)

    try:
        if len(np.unique(y_now)) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_now, test_size=0.30, random_state=42, stratify=y_now
            )
        else:
            X_test, y_test = X, y_now
        y_prob = model_now.predict_proba(X_test)[:, 1]
        auc = safe_auc(y_test, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred) if len(y_test) > 0 else np.nan
    except Exception:
        X_test, y_test = X, y_now
        y_prob = model_now.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = safe_auc(y_now, y_prob)
        acc = accuracy_score(y_now, y_pred)

    return {
        "features": feats,
        "model_now": model_now,
        "model_future": model_future,
        "rf": rf,
        "rf_acc": rf_acc,
        "iso": iso,
        "iso_scaler": scaler_iso,
        "eval_X": X_test,
        "eval_y": y_test,
        "eval_prob": y_prob,
        "eval_auc": auc,
        "eval_acc": acc,
    }


def apply_models(df, models):
    X = df[models["features"]].copy()
    out = df.copy()

    out["distress_prob"] = models["model_now"].predict_proba(X)[:, 1]
    out["future_distress_prob"] = models["model_future"].predict_proba(X)[:, 1]

    if models["rf"] is not None:
        out["state_pred"] = models["rf"].predict(X)
    else:
        out["state_pred"] = (out["distress_prob"] >= 0.5).astype(int)

    X_iso = models["iso_scaler"].transform(X)
    iso_pred = models["iso"].predict(X_iso)
    out["anomaly_flag"] = (iso_pred == -1).astype(int)
    out["anomaly_score"] = models["iso"].decision_function(X_iso)

    p_depth = np.clip(out["depth_s"] / 1.6, 0, 1)
    p_hr_high = np.clip((out["hr_s"] - 110) / 45.0, 0, 1)
    p_hr_low = np.clip((60 - out["hr_s"]) / 25.0, 0, 1)
    p_spo2 = np.clip((96 - out["spo2_s"]) / 12.0, 0, 1)
    p_motion = np.clip((out["motion_s"] - 0.55) / 0.85, 0, 1)
    p_thermal = np.clip(out["thermal_risk"], 0, 1)
    p_water = out["water_contact"]
    p_panic = out["panic_button"]

    out["fusion_risk"] = np.clip(
        0.22 * p_depth
        + 0.14 * np.maximum(p_hr_high, p_hr_low)
        + 0.16 * p_spo2
        + 0.15 * p_motion
        + 0.10 * p_thermal
        + 0.08 * p_water
        + 0.05 * p_panic
        + 0.10 * out["distress_prob"],
        0,
        1,
    )

    variability = float(out["motion_s"].rolling(25, min_periods=1).std().iloc[-1])
    adaptive_threshold = min(0.88, 0.55 + 0.12 * variability + 0.10 * float(out["anomaly_flag"].mean()))
    out["adaptive_threshold"] = adaptive_threshold

    health = (
        100
        - 24 * out["fusion_risk"]
        - 18 * out["future_distress_prob"]
        - 12 * out["anomaly_flag"]
        - 6 * np.clip((95 - out["spo2_s"]) / 8, 0, 1)
        - 8 * np.clip((35.8 - out["body_temp_c"]) / 6, 0, 1)
    )
    out["health_score"] = np.clip(health, 0, 100)

    severity = []
    for _, row in out.iterrows():
        if row["fusion_risk"] >= 0.80 or row["future_distress_prob"] >= 0.88:
            severity.append("CRITICAL")
        elif row["fusion_risk"] >= 0.60 or row["future_distress_prob"] >= 0.70:
            severity.append("HIGH")
        elif row["fusion_risk"] >= 0.35 or row["anomaly_flag"] == 1:
            severity.append("MEDIUM")
        else:
            severity.append("LOW")
    out["severity"] = severity

    out["deploy_recommendation"] = np.where(
        (out["fusion_risk"] > out["adaptive_threshold"]) | (out["panic_button"] > 0),
        1,
        0,
    )

    out["alert_stage"] = np.select(
        [
            out["severity"] == "CRITICAL",
            out["severity"] == "HIGH",
            out["severity"] == "MEDIUM",
        ],
        [
            "Stage 3: Inflate + SOS",
            "Stage 2: Urgent Alert + Countdown",
            "Stage 1: Advisory Warning",
        ],
        default="Normal Monitoring",
    )

    return out


def build_dummy_map_points(lat_center, lon_center, severity):
    offset = {"LOW": 0.002, "MEDIUM": 0.004, "HIGH": 0.006, "CRITICAL": 0.008}.get(severity, 0.004)
    return pd.DataFrame(
        [
            {"type": "Incident", "lat": lat_center, "lon": lon_center},
            {"type": "Rescue Team Alpha", "lat": lat_center + offset, "lon": lon_center - offset / 2},
            {"type": "Rescue Boat", "lat": lat_center - offset / 1.5, "lon": lon_center + offset / 3},
            {"type": "Medical Point", "lat": lat_center + offset / 2, "lon": lon_center + offset},
        ]
    )


def latest_message(row):
    if row["severity"] == "CRITICAL":
        return "🚨 Critical distress pattern detected. Immediate inflation and rescue dispatch recommended.", "error"
    if row["severity"] == "HIGH":
        return "⚠️ High-risk pattern detected. Prepare intervention and keep close observation.", "warning"
    if row["severity"] == "MEDIUM":
        return "🟡 Elevated risk or anomaly detected. Continue monitoring and verify sensors.", "info"
    return "✅ System indicates low current risk.", "success"


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("System Configuration")
st.sidebar.markdown(role_badge(st.session_state.role), unsafe_allow_html=True)
st.sidebar.write(f"**Logged in as:** {st.session_state.display_name}")
st.sidebar.write(f"**Username:** {st.session_state.username}")
if st.sidebar.button("Logout", use_container_width=True):
    logout_user()

st.sidebar.markdown("---")
scenario = st.sidebar.selectbox(
    "Scenario",
    ["Normal Swim", "Vigorous Swim", "Panic Struggle", "Silent Distress", "Cold Shock", "Near Drowning", "Child Fall"],
)
n_samples = st.sidebar.slider("Simulation Length", 120, 1200, 300, 60)
turbulence = st.sidebar.slider("Turbulence Intensity", 0.0, 1.0, 0.35, 0.05)
dropout = st.sidebar.slider("Sensor Dropout Probability", 0.0, 0.30, 0.05, 0.01)
water_temp = st.sidebar.slider("Water Temperature (°C)", 10.0, 35.0, 24.0, 0.5)
user_mass = st.sidebar.slider("Estimated User Mass (kg)", 20, 100, 70, 1)
auto_refresh = st.sidebar.checkbox("Auto-refresh demo", value=False)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 1, 15, 4)
generate_new = st.sidebar.button("Generate New Demo Data")

st.sidebar.markdown("---")
st.sidebar.subheader("Safety Thresholds")
fusion_warn = st.sidebar.slider("Fusion risk warning", 0.10, 0.95, 0.60, 0.01)
future_warn = st.sidebar.slider("Future distress warning", 0.10, 0.95, 0.70, 0.01)
depth_warn = st.sidebar.slider("Depth warning (m)", 0.20, 3.00, 1.00, 0.05)
spo2_warn = st.sidebar.slider("SpO₂ warning (%)", 75, 99, 93)
temp_warn = st.sidebar.slider("Body temperature concern (°C)", 24.0, 37.0, 35.5, 0.1)

st.sidebar.markdown("---")
st.sidebar.info("This app is built for the aquatic airbag project.")


# ============================================================
# DATA LOAD / REFRESH
# ============================================================
if st.session_state.sim_df is None or generate_new:
    st.session_state.sim_df = generate_airbag_demo_data(
        scenario_name=scenario,
        n=n_samples,
        turbulence=turbulence,
        dropout=dropout,
        water_temp_c=water_temp,
    )

raw_df = st.session_state.sim_df.copy()

if auto_refresh:
    time.sleep(refresh_seconds)
    st.session_state.sim_df = generate_airbag_demo_data(
        scenario_name=scenario,
        n=n_samples,
        turbulence=turbulence,
        dropout=dropout,
        water_temp_c=water_temp,
    )
    st.rerun()

models = fit_models(raw_df)
proc_df = apply_models(raw_df, models)
latest = proc_df.iloc[-1]
prev = proc_df.iloc[-2] if len(proc_df) > 1 else None

lat_center = 23.8103 + np.random.uniform(-0.01, 0.01)
lon_center = 90.4125 + np.random.uniform(-0.01, 0.01)
map_df = build_dummy_map_points(lat_center, lon_center, latest["severity"])

if generate_new:
    db.log_event(
        st.session_state.username,
        st.session_state.role,
        "simulation_refresh",
        latest["severity"],
        f"Generated scenario {scenario}",
        prediction_prob=float(latest["future_distress_prob"]),
        fusion_risk=float(latest["fusion_risk"]),
        health_score=float(latest["health_score"]),
        latitude=float(lat_center),
        longitude=float(lon_center),
        scenario_name=scenario,
    )
    if latest["severity"] in ["HIGH", "CRITICAL"]:
        db.add_incident(
            scenario_name=scenario,
            severity=latest["severity"],
            depth=float(latest["depth_s"]),
            heart_rate=float(latest["hr_s"]),
            body_temp=float(latest["body_temp_c"]),
            spo2=float(latest["spo2_s"]),
            motion_index=float(latest["motion_s"]),
            fusion_risk=float(latest["fusion_risk"]),
            prediction_prob=float(latest["future_distress_prob"]),
            health_score=float(latest["health_score"]),
            location_name="Demo Aquatic Zone",
            latitude=float(lat_center),
            longitude=float(lon_center),
            assigned_team="Team Alpha",
            notes="Auto-generated demo incident.",
        )


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🌊 IoT-Driven Aquatic Airbag Safety System Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Advanced multi-sensor distress inference, adaptive airbag deployment support, rescue coordination, and research analytics platform.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="soft-box">
        <b>User:</b> {st.session_state.display_name} &nbsp;&nbsp;
        <b>Role:</b> {st.session_state.role} &nbsp;&nbsp;
        <b>Scenario:</b> {scenario} &nbsp;&nbsp;
        <b>System Status:</b> {'Attention Required' if latest['severity'] in ['HIGH', 'CRITICAL'] else 'Monitoring'}
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(8)
metrics = [
    ("Depth (m)", latest["depth_s"], safe_metric_delta(latest["depth_s"], None if prev is None else prev["depth_s"])),
    ("Heart Rate", latest["hr_s"], safe_metric_delta(latest["hr_s"], None if prev is None else prev["hr_s"])),
    ("Body Temp (°C)", latest["body_temp_c"], safe_metric_delta(latest["body_temp_c"], None if prev is None else prev["body_temp_c"])),
    ("SpO₂ (%)", latest["spo2_s"], safe_metric_delta(latest["spo2_s"], None if prev is None else prev["spo2_s"])),
    ("Motion Index", latest["motion_s"], safe_metric_delta(latest["motion_s"], None if prev is None else prev["motion_s"])),
    ("Fusion Risk", latest["fusion_risk"], safe_metric_delta(latest["fusion_risk"], None if prev is None else prev["fusion_risk"])),
    ("Future Distress", latest["future_distress_prob"], safe_metric_delta(latest["future_distress_prob"], None if prev is None else prev["future_distress_prob"])),
    ("Health Score", latest["health_score"], safe_metric_delta(latest["health_score"], None if prev is None else prev["health_score"])),
]
for c, (label, val, delta) in zip(cols, metrics):
    c.metric(label, f"{val:.2f}", None if delta is None else f"{delta:.3f}")

msg, level = latest_message(latest)
getattr(st, level)(msg)

if latest["future_distress_prob"] >= future_warn:
    st.warning("Early warning: elevated probability of distress in the next prediction horizon.")
if latest["deploy_recommendation"] == 1:
    st.error("Airbag deployment condition is satisfied by adaptive logic or manual panic trigger.")


# ============================================================
# TABS
# ============================================================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Role Workspace",
        "Live Monitoring",
        "Risk & Deployment",
        "ML Evaluation",
        "Rescue Map",
        "Incidents / Tickets / Notes",
        "Research Panel",
        "Dataset / Export",
    ]
)

with tab0:
    role = st.session_state.role

    if role == "Public User":
        st.markdown("### Public User Workspace")
        st.info("Designed for awareness, alerts, and simplified reporting.")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Severity", latest["severity"])
        a2.metric("Deploy Recommendation", int(latest["deploy_recommendation"]))
        a3.metric("Water Contact", int(latest["water_contact"]))
        a4.metric("Panic Button", int(latest["panic_button"]))

        with st.form("public_report_form"):
            issue_type = st.selectbox("Issue Type", ["Observed distress", "Water fall event", "Child fall suspicion", "General concern"])
            severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            description = st.text_area("Description")
            submit_public = st.form_submit_button("Submit Report")
            if submit_public:
                db.add_incident(
                    scenario_name=scenario,
                    severity=severity.upper(),
                    depth=float(latest["depth_s"]),
                    heart_rate=float(latest["hr_s"]),
                    body_temp=float(latest["body_temp_c"]),
                    spo2=float(latest["spo2_s"]),
                    motion_index=float(latest["motion_s"]),
                    fusion_risk=float(latest["fusion_risk"]),
                    prediction_prob=float(latest["future_distress_prob"]),
                    health_score=float(latest["health_score"]),
                    location_name="Public Report Zone",
                    latitude=float(lat_center),
                    longitude=float(lon_center),
                    assigned_team="Team Alpha",
                    notes=description,
                )
                db.log_event(
                    st.session_state.username,
                    role,
                    "public_report",
                    severity,
                    issue_type,
                    float(latest["future_distress_prob"]),
                    float(latest["fusion_risk"]),
                    float(latest["health_score"]),
                    float(lat_center),
                    float(lon_center),
                    scenario,
                )
                st.success("Report submitted.")

    elif role == "Supervisor":
        st.markdown("### Supervisor Workspace")
        s1, s2, s3, s4 = st.columns(4)
        incidents_df = db.query_df("SELECT * FROM incidents ORDER BY id DESC LIMIT 200")
        open_tickets_df = db.query_df("SELECT * FROM tickets WHERE status IN ('Open','Assigned','In Progress') ORDER BY id DESC LIMIT 200")
        s1.metric("Recent Incidents", len(incidents_df))
        s2.metric("Open Tickets", len(open_tickets_df))
        s3.metric("Latest Severity", latest["severity"])
        s4.metric("Estimated Buoyancy Need (L)", f"{flotation_liters(user_mass):.1f}")

        if not incidents_df.empty:
            st.dataframe(
                incidents_df[
                    ["ts", "scenario_name", "severity", "fusion_risk", "prediction_prob", "assigned_team", "location_name"]
                ].head(20),
                use_container_width=True,
            )

            with st.form("ticket_create_form"):
                incident_idx = st.number_input("Incident row index to convert (top table index)", min_value=0, max_value=max(0, len(incidents_df) - 1), value=0)
                issue_type = st.text_input("Issue Type", value="Aquatic distress response")
                description = st.text_area("Description", value="Supervisor-generated rescue/inspection ticket")
                create_ticket = st.form_submit_button("Create Ticket")
                if create_ticket and len(incidents_df) > 0:
                    chosen = incidents_df.iloc[int(incident_idx)]
                    db.execute(
                        """
                        INSERT INTO tickets (
                            ticket_id, created_time, created_by, incident_time, severity,
                            issue_type, description, status, assigned_to, action_note
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            create_ticket_id(),
                            now_str(),
                            st.session_state.display_name,
                            chosen["ts"],
                            chosen["severity"],
                            issue_type,
                            description,
                            "Open",
                            "Team Alpha",
                            "",
                        ),
                    )
                    db.log_event(
                        st.session_state.username,
                        role,
                        "ticket_create",
                        "success",
                        "Supervisor created ticket",
                        float(latest["future_distress_prob"]),
                        float(latest["fusion_risk"]),
                        float(latest["health_score"]),
                        float(lat_center),
                        float(lon_center),
                        scenario,
                    )
                    st.success("Ticket created.")

    elif role == "Rescue Team":
        st.markdown("### Rescue Team Workspace")
        st.warning("Operational panel for response coordination and status updates.")
        tickets_df = db.query_df("SELECT * FROM tickets ORDER BY id DESC LIMIT 100")
        if not tickets_df.empty:
            st.dataframe(tickets_df, use_container_width=True)
            with st.form("rescue_update_form"):
                ticket_id = st.selectbox("Ticket ID", tickets_df["ticket_id"].tolist())
                status = st.selectbox("Status Update", ["Assigned", "In Progress", "Resolved", "Closed"])
                note = st.text_area("Action Note")
                submit_update = st.form_submit_button("Update Ticket")
                if submit_update:
                    db.execute(
                        "UPDATE tickets SET status=?, action_note=?, assigned_to=? WHERE ticket_id=?",
                        (status, note, st.session_state.display_name, ticket_id),
                    )
                    db.log_event(
                        st.session_state.username,
                        role,
                        "ticket_update",
                        status,
                        note,
                        float(latest["future_distress_prob"]),
                        float(latest["fusion_risk"]),
                        float(latest["health_score"]),
                        float(lat_center),
                        float(lon_center),
                        scenario,
                    )
                    st.success("Ticket updated.")
        else:
            st.info("No tickets available yet.")

    elif role == "Researcher":
        st.markdown("### Researcher Workspace")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("ROC-AUC", "N/A" if np.isnan(models["eval_auc"]) else f"{models['eval_auc']:.3f}")
        r2.metric("Accuracy", f"{models['eval_acc']:.3f}" if not np.isnan(models["eval_acc"]) else "N/A")
        r3.metric("RF Accuracy", "N/A" if np.isnan(models["rf_acc"]) else f"{models['rf_acc']:.3f}")
        r4.metric("Adaptive Threshold", f"{latest['adaptive_threshold']:.3f}")

        st.write("Current feature set:")
        st.code(", ".join(feature_columns()))
        st.dataframe(proc_df.tail(20), use_container_width=True)


with tab1:
    left, right = st.columns([1.8, 1])

    with left:
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=("Depth", "Heart Rate", "SpO₂", "Motion", "Pressure", "Body Temperature"),
        )
        recent = proc_df.tail(180)
        fig.add_trace(go.Scatter(x=recent["t"], y=recent["depth_s"], name="Depth"), row=1, col=1)
        fig.add_trace(go.Scatter(x=recent["t"], y=recent["hr_s"], name="HR"), row=1, col=2)
        fig.add_trace(go.Scatter(x=recent["t"], y=recent["spo2_s"], name="SpO2"), row=2, col=1)
        fig.add_trace(go.Scatter(x=recent["t"], y=recent["motion_s"], name="Motion"), row=2, col=2)
        fig.add_trace(go.Scatter(x=recent["t"], y=recent["pressure_s"], name="Pressure"), row=3, col=1)
        fig.add_trace(go.Scatter(x=recent["t"], y=recent["body_temp_c"], name="Body Temp"), row=3, col=2)
        fig.update_layout(height=900, showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Latest System Summary")
        show_cols = [
            "t", "depth_s", "hr_s", "spo2_s", "motion_s", "body_temp_c",
            "fusion_risk", "future_distress_prob", "severity", "deploy_recommendation", "alert_stage"
        ]
        st.dataframe(recent[show_cols].tail(20).sort_values("t", ascending=False), use_container_width=True, height=700)


with tab2:
    c1, c2 = st.columns(2)
    with c1:
        risk_fig = go.Figure()
        risk_fig.add_trace(go.Scatter(x=proc_df["t"], y=proc_df["fusion_risk"], mode="lines", name="Fusion Risk"))
        risk_fig.add_trace(go.Scatter(x=proc_df["t"], y=proc_df["future_distress_prob"], mode="lines", name="Future Distress"))
        risk_fig.add_trace(go.Scatter(x=proc_df["t"], y=proc_df["adaptive_threshold"], mode="lines", name="Adaptive Threshold"))
        risk_fig.update_layout(title="Decision Risk Curves", template="plotly_white", yaxis=dict(range=[0, 1]))
        st.plotly_chart(risk_fig, use_container_width=True)

    with c2:
        sev_counts = proc_df["severity"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).fillna(0)
        sev_fig = px.pie(values=sev_counts.values, names=sev_counts.index, title="Severity Distribution")
        st.plotly_chart(sev_fig, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.metric("Current Deployment Recommendation", int(latest["deploy_recommendation"]))
    d2.metric("Required Flotation Volume (L)", f"{flotation_liters(user_mass):.1f}")
    d3.metric("Submerge Ratio (20-step)", f"{latest['submerge_ratio_20']:.2f}")

    st.markdown("### Explainable Trigger Drivers")
    drivers = []
    if latest["depth_s"] > depth_warn:
        drivers.append("Depth above warning level")
    if latest["spo2_s"] < spo2_warn:
        drivers.append("SpO₂ decline")
    if latest["motion_s"] > 0.95:
        drivers.append("High motion / struggle")
    if latest["hr_s"] > 130:
        drivers.append("Tachycardic response")
    if latest["hr_s"] < 55:
        drivers.append("Bradycardic pattern")
    if latest["body_temp_c"] < temp_warn:
        drivers.append("Thermal decline")
    if latest["panic_button"] > 0:
        drivers.append("Manual panic button activation")
    if latest["water_contact"] > 0:
        drivers.append("Confirmed water contact")
    if not drivers:
        drivers.append("No dominant adverse trigger at current step")
    for item in drivers:
        st.write(f"- {item}")


with tab3:
    st.markdown("### ML Evaluation and Robustness")
    st.write("This version is hardened against single-class crashes by falling back to a safe probability model when necessary.")

    evX = models["eval_X"]
    evy = models["eval_y"]
    evp = models["eval_prob"]
    evpred = (evp >= 0.5).astype(int)

    c1, c2 = st.columns(2)
    with c1:
        if len(np.unique(evy)) >= 2:
            fpr, tpr, _ = roc_curve(evy, evp)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance"))
            roc_fig.update_layout(title="ROC Curve", template="plotly_white")
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("ROC unavailable because the evaluation slice contains one class only.")

    with c2:
        cm = confusion_matrix(evy, evpred, labels=[0, 1])
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=True)

        if len(np.unique(evy)) >= 2:
            prec, rec, _ = precision_recall_curve(evy, evp)
            pr_fig = go.Figure()
            pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
            pr_fig.update_layout(title="Precision-Recall Curve", template="plotly_white")
            st.plotly_chart(pr_fig, use_container_width=True)

    st.metric("Validation Accuracy", f"{models['eval_acc']:.4f}" if not np.isnan(models["eval_acc"]) else "N/A")
    st.metric("Validation ROC-AUC", f"{models['eval_auc']:.4f}" if not np.isnan(models["eval_auc"]) else "N/A")


with tab4:
    st.markdown("### Rescue Map")
    st.caption("Synthetic map layer for demo presentation.")
    st.map(
        map_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
        latitude="latitude",
        longitude="longitude",
        size=20,
    )
    st.dataframe(map_df, use_container_width=True)

    st.write(
        """
        - Incident: detected aquatic emergency zone
        - Rescue Team Alpha: nearest response unit
        - Rescue Boat: mobile aquatic response asset
        - Medical Point: first-aid / stabilization point
        """
    )


with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Event Logs")
        logs_df = db.query_df(
            """
            SELECT ts, username, role, event_type, status, message,
                   prediction_prob, fusion_risk, health_score, scenario_name
            FROM event_logs
            ORDER BY id DESC LIMIT 100
            """
        )
        st.dataframe(logs_df, use_container_width=True, height=320)

        st.markdown("### Incidents")
        incidents_df = db.query_df(
            """
            SELECT ts, scenario_name, severity, depth, heart_rate, body_temp, spo2,
                   motion_index, fusion_risk, prediction_prob, health_score,
                   location_name, assigned_team
            FROM incidents
            ORDER BY id DESC LIMIT 100
            """
        )
        st.dataframe(incidents_df, use_container_width=True, height=320)

    with c2:
        st.markdown("### Tickets")
        tickets_df = db.query_df("SELECT * FROM tickets ORDER BY id DESC LIMIT 100")
        st.dataframe(tickets_df, use_container_width=True, height=320)

        st.markdown("### Notes")
        with st.form("note_form"):
            title = st.text_input("Note Title")
            note = st.text_area("Note")
            save_note = st.form_submit_button("Save Note")
            if save_note and title.strip():
                db.execute(
                    "INSERT INTO notes (ts, username, title, note) VALUES (?, ?, ?, ?)",
                    (now_str(), st.session_state.username, title.strip(), note.strip()),
                )
                db.log_event(
                    st.session_state.username,
                    st.session_state.role,
                    "note",
                    "saved",
                    title.strip(),
                    float(latest["future_distress_prob"]),
                    float(latest["fusion_risk"]),
                    float(latest["health_score"]),
                    float(lat_center),
                    float(lon_center),
                    scenario,
                )
                st.success("Note saved.")
        notes_df = db.query_df("SELECT ts, username, title, note FROM notes ORDER BY id DESC LIMIT 50")
        st.dataframe(notes_df, use_container_width=True, height=260)


with tab6:
    st.markdown("### Research Contribution Panel")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Mean Fusion Risk", f"{proc_df['fusion_risk'].mean():.3f}")
    r2.metric("Mean Future Distress", f"{proc_df['future_distress_prob'].mean():.3f}")
    r3.metric("Anomaly Rate", f"{100*proc_df['anomaly_flag'].mean():.2f}%")
    r4.metric("Mean Health Score", f"{proc_df['health_score'].mean():.2f}")

    c1, c2 = st.columns(2)
    with c1:
        rel_df = pd.DataFrame({"Turbulence": np.linspace(0.0, 1.0, 11)})
        rel_df["Estimated Reliability"] = np.clip(1.0 - 0.52 * rel_df["Turbulence"] - 0.35 * dropout, 0.15, 0.99)
        rel_fig = px.line(rel_df, x="Turbulence", y="Estimated Reliability", title="Estimated Reliability vs Turbulence")
        st.plotly_chart(rel_fig, use_container_width=True)

    with c2:
        hs_fig = px.histogram(proc_df, x="health_score", nbins=25, title="Health Score Distribution")
        st.plotly_chart(hs_fig, use_container_width=True)

    st.write(
        """
       
        """
    )


with tab7:
    st.markdown("### Dataset and Export")
    st.dataframe(proc_df.tail(30), use_container_width=True)

    st.download_button(
        "Download Current Simulation CSV",
        proc_df.to_csv(index=False).encode("utf-8"),
        "airbag_simulation.csv",
        "text/csv",
    )

    st.download_button(
        "Download Event Logs CSV",
        db.query_df("SELECT * FROM event_logs ORDER BY id DESC LIMIT 500").to_csv(index=False).encode("utf-8"),
        "airbag_event_logs.csv",
        "text/csv",
    )

    st.download_button(
        "Download Incidents CSV",
        db.query_df("SELECT * FROM incidents ORDER BY id DESC LIMIT 500").to_csv(index=False).encode("utf-8"),
        "airbag_incidents.csv",
        "text/csv",
    )
