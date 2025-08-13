import sqlite3
from datetime import datetime
from typing import List, Optional

DB_PATH = "chat_history.db"

# === Always enforce foreign keys ===
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# === Initialize DB ===
def init_db():
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            user TEXT NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT DEFAULT 'Untitled Chat',
            created_at TEXT NOT NULL,
            PRIMARY KEY (user, session_id)
        )
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT NOT NULL,
            session_id TEXT NOT NULL,
            title TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user, session_id)
                REFERENCES sessions(user, session_id)
                ON DELETE CASCADE
        )
        """)

# === Save chat entry ===
def save_chat_to_db(user: str, query: str, response: str, session_id: str):
    timestamp = datetime.utcnow().isoformat()

    with get_connection() as conn:
        # Fetch title from sessions table
        cur = conn.execute("""
            SELECT title FROM sessions
            WHERE user = ? AND session_id = ?
        """, (user, session_id))
        result = cur.fetchone()
        title = result[0] if result else "Untitled Chat"

        # Save to chat_history
        conn.execute("""
            INSERT INTO chat_history (user, session_id, title, query, response, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user, session_id, title, query, response, timestamp))

# === Load chat history ===
def load_chat_history(user: str, session_id: str = "default", limit: int = 10) -> List[dict]:
    with get_connection() as conn:
        cursor = conn.execute("""
        SELECT query, response FROM chat_history
        WHERE user = ? AND session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """, (user, session_id, limit))
        rows = cursor.fetchall()
        rows.reverse()

        messages = []
        for q, r in rows:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": r})
        return messages

# === List sessions ===
def get_user_sessions(user: str) -> List[dict]:
    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT s.session_id, s.title, MAX(c.timestamp) as last_used
            FROM sessions s
            LEFT JOIN chat_history c ON s.session_id = c.session_id AND s.user = c.user
            WHERE s.user = ?
            GROUP BY s.session_id
            ORDER BY last_used DESC NULLS LAST
        """, (user,))
        return [
            {
                "session_id": row[0],
                "title": row[1],
                "last_used": row[2]
            }
            for row in cursor.fetchall()
        ]
    

# === Create session ===
def create_new_session(user: str, session_id: str, title: str):
    with get_connection() as conn:
        now = datetime.utcnow().isoformat()
        conn.execute("""
            INSERT INTO sessions (user, session_id, title, created_at)
            VALUES (?, ?, ?, ?)
        """, (user, session_id, title, now))

# === Rename session ===
def rename_session(user: str, session_id: str, new_title: str) -> bool:
    with get_connection() as conn:
        # Update sessions table
        cur1 = conn.execute("""
            UPDATE sessions
            SET title = ?
            WHERE user = ? AND session_id = ?
        """, (new_title, user, session_id))

        # Update chat_history table
        conn.execute("""
            UPDATE chat_history
            SET title = ?
            WHERE user = ? AND session_id = ?
        """, (new_title, user, session_id))

        conn.commit()
        return cur1.rowcount > 0

# === Delete session (chat_history rows will auto-delete due to cascade) ===
def delete_session(user: str, session_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("""
            DELETE FROM sessions
            WHERE user = ? AND session_id = ?
        """, (user, session_id))
        conn.commit()
        return cur.rowcount > 0

# === Initialize DB on import ===
init_db()

# sqlite_web backend/chat_history.db --port 8082