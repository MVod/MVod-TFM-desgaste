# app/db.py
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from config import DB_PATH

def init_db():
    """Crea las tablas si no existen (esquema completo)."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experimentos (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre          TEXT,
                roscas_train    TEXT    NOT NULL,
                roscas_eval     TEXT,
                modelo_path     TEXT    NOT NULL,
                iou_cresta      REAL,
                iou_paso        REAL,
                loss_final      REAL,
                epochs          INTEGER,
                auc_roc         REAL,
                f1              REAL,
                auprc           REAL,
                es_mejor        INTEGER DEFAULT 0,
                tipo            TEXT    DEFAULT 'manual',
                creado_en       TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS resultados (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                experimento_id  INTEGER NOT NULL,
                rosca_id        TEXT    NOT NULL,
                es_buena        INTEGER NOT NULL,
                cv_cresta       REAL,
                score           REAL,
                diagnostico     TEXT,
                correcto        INTEGER,
                FOREIGN KEY (experimento_id) REFERENCES experimentos(id)
            );
        """)


def migrar_db():
    """Añade y renombra columnas en tablas existentes (idempotente)."""
    nuevas_cols = {
        "nombre":          "TEXT",
        "roscas_eval":     "TEXT",
        "auc_roc":         "REAL",
        "f1":              "REAL",
        "auprc":           "REAL",
        "es_mejor":        "INTEGER DEFAULT 0",
        "tipo":            "TEXT DEFAULT 'manual'",
        "tiempo_train_s":  "REAL",
        "tiempo_inf_s":    "REAL",
    }

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.isolation_level = None
    try:
        existentes = {r[1] for r in conn.execute("PRAGMA table_info(experimentos)").fetchall()}
        for col, tipo in nuevas_cols.items():
            if col not in existentes:
                conn.execute(f"ALTER TABLE experimentos ADD COLUMN {col} {tipo}")

        _old_iou = "iou_filete"
        if _old_iou in existentes and "iou_cresta" not in existentes:
            conn.execute(f"ALTER TABLE experimentos RENAME COLUMN {_old_iou} TO iou_cresta")
        cols_r = {r[1] for r in conn.execute("PRAGMA table_info(resultados)").fetchall()}
        _old_cv = "cv_filete"
        if _old_cv in cols_r and "cv_cresta" not in cols_r:
            conn.execute(f"ALTER TABLE resultados RENAME COLUMN {_old_cv} TO cv_cresta")

        loo_ids = [r[0] for r in conn.execute(
            "SELECT id FROM experimentos WHERE tipo = 'loo'"
        ).fetchall()]
        if loo_ids:
            ph = ",".join("?" * len(loo_ids))
            conn.execute(f"DELETE FROM resultados WHERE experimento_id IN ({ph})", loo_ids)
            conn.execute(f"DELETE FROM experimentos WHERE id IN ({ph})", loo_ids)
    finally:
        conn.close()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def guardar_experimento(
    roscas_train: list[str],
    modelo_path: Path,
    iou_cresta: float = None,
    iou_paso: float = None,
    loss_final: float = None,
    epochs: int = None,
    nombre: str = None,
    roscas_eval: list[str] = None,
    tipo: str = "manual",
    tiempo_train_s: float = None,
    tiempo_inf_s: float = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO experimentos
                (nombre, roscas_train, roscas_eval, modelo_path,
                 iou_cresta, iou_paso, loss_final, epochs,
                 tipo, tiempo_train_s, tiempo_inf_s, creado_en)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nombre,
                json.dumps(sorted(roscas_train)),
                json.dumps(sorted(roscas_eval)) if roscas_eval else None,
                str(modelo_path),
                iou_cresta,
                iou_paso,
                loss_final,
                epochs,
                tipo,
                tiempo_train_s,
                tiempo_inf_s,
                datetime.now().isoformat(),
            ),
        )
        return cur.lastrowid


def actualizar_tiempos(exp_id: int, tiempo_train_s: float = None, tiempo_inf_s: float = None) -> None:
    sets, vals = [], []
    if tiempo_train_s is not None:
        sets.append("tiempo_train_s=?"); vals.append(tiempo_train_s)
    if tiempo_inf_s is not None:
        sets.append("tiempo_inf_s=?"); vals.append(tiempo_inf_s)
    if not sets:
        return
    vals.append(exp_id)
    with _connect() as conn:
        conn.execute(f"UPDATE experimentos SET {', '.join(sets)} WHERE id=?", vals)


def actualizar_metricas_avanzadas(
    exp_id: int,
    auc_roc: float | None,
    f1: float | None,
    auprc: float | None,
) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE experimentos SET auc_roc=?, f1=?, auprc=? WHERE id=?",
            (auc_roc, f1, auprc, exp_id),
        )


def set_mejor_modelo(exp_id: int) -> None:
    """Marca exp_id como mejor modelo y desmarca el resto."""
    with _connect() as conn:
        conn.execute("UPDATE experimentos SET es_mejor = 0")
        conn.execute("UPDATE experimentos SET es_mejor = 1 WHERE id = ?", (exp_id,))


def get_mejor_modelo() -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM experimentos WHERE es_mejor = 1 LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def buscar_experimento(roscas_train: list[str]) -> dict | None:
    key = json.dumps(sorted(roscas_train))
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM experimentos
            WHERE roscas_train = ?
            ORDER BY creado_en DESC
            LIMIT 1
            """,
            (key,),
        ).fetchone()
    return dict(row) if row else None


def listar_experimentos() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM experimentos ORDER BY creado_en DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Resultados ────────────────────────────────────────────────────────────────


def guardar_resultado(
    experimento_id: int,
    rosca_id: str,
    es_buena: bool,
    cv_cresta: float,
    score: float,
    diagnostico: str,
) -> None:
    correcto = int(
        (diagnostico == "BUENA" and es_buena)
        or (diagnostico == "POSIBLE DESGASTE" and not es_buena)
    )
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO resultados
                (experimento_id, rosca_id, es_buena, cv_cresta, score, diagnostico, correcto)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (experimento_id, rosca_id, int(es_buena), cv_cresta, score, diagnostico, correcto),
        )


def get_resultados(experimento_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM resultados WHERE experimento_id = ?",
            (experimento_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_accuracy(experimento_id: int) -> float | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT AVG(correcto) as acc FROM resultados WHERE experimento_id = ?",
            (experimento_id,),
        ).fetchone()
    return row["acc"] if row else None


def borrar_experimento(exp_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM resultados WHERE experimento_id = ?", (exp_id,))
        conn.execute("DELETE FROM experimentos WHERE id = ?", (exp_id,))


def borrar_experimentos_tipo(tipo: str) -> int:
    """Borra todos los experimentos de un tipo. Devuelve el número eliminado."""
    with _connect() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM experimentos WHERE tipo = ?", (tipo,)
        ).fetchall()]
        if ids:
            ph = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM resultados WHERE experimento_id IN ({ph})", ids)
            conn.execute(f"DELETE FROM experimentos WHERE id IN ({ph})", ids)
        return len(ids)


def borrar_todos_experimentos() -> int:
    """Borra todos los experimentos y sus resultados. Devuelve el número eliminado."""
    with _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM experimentos").fetchone()[0]
        conn.execute("DELETE FROM resultados")
        conn.execute("DELETE FROM experimentos")
        return n
