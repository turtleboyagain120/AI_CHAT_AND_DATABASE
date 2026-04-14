import sqlite3
from datetime import datetime
import os

DB_PATH = 'chats.db'

def token_estimate(text):
    """Rough token estimate: words * 1.3 (avg English tokens)"""

    words = len(text.split())
    return int(words * 1.3)

class ChatDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.rate_limit_tracker = {}
        self.init_db()


    def init_db(self):
        cursor = self.conn.cursor()
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('human', 'assistant')),
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tokens INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')
        # Rate limits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limits (
                session_id INTEGER PRIMARY KEY,
                last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                retries INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')
        self.conn.commit()


    def create_session(self, name):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
        self.conn.commit()
        return cursor.lastrowid

    def get_or_create_session(self, name):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM sessions WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            session_id = row[0]
        else:
            session_id = self.create_session(name)
        self.reset_rate_limit(session_id)
        return session_id


    def add_message(self, session_id, role, content):
        tokens = token_estimate(content)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, tokens)
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, tokens))
        # Update session updated_at
        cursor.execute("""
            UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (session_id,))
        self.conn.commit()
        return cursor.lastrowid

    def get_session_history(self, session_id, max_tokens=1024):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT role, content FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        msgs = cursor.fetchall()
        history = []
        total_tokens = 0
        for role, content in msgs:
            msg_tokens = token_estimate(content)
            if total_tokens + msg_tokens > max_tokens:
                break  # Truncate old
            history.append({'role': role, 'content': content})
            total_tokens += msg_tokens
        return history

    def list_sessions(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, created_at, updated_at FROM sessions 
            ORDER BY updated_at DESC
        """)
        return cursor.fetchall()

    def delete_session(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_message_count(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        return cursor.fetchone()[0]

    def reset_rate_limit(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO rate_limits (session_id, last_reset, retries)
            VALUES (?, CURRENT_TIMESTAMP, 0)
        """, (session_id,))
        self.conn.commit()
        self.rate_limit_tracker[session_id] = {'retries': 0, 'last_reset': datetime.now()}

    def get_rate_limit_info(self, session_id):
        if session_id in self.rate_limit_tracker:
            return self.rate_limit_tracker[session_id]
        cursor = self.conn.cursor()
        cursor.execute("SELECT retries, last_reset FROM rate_limits WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            self.rate_limit_tracker[session_id] = {'retries': row[0], 'last_reset': row[1]}
            return self.rate_limit_tracker[session_id]
        return {'retries': 0, 'last_reset': None}

    def increment_retry(self, session_id):
        info = self.get_rate_limit_info(session_id)
        info['retries'] += 1
        cursor = self.conn.cursor()
        cursor.execute("UPDATE rate_limits SET retries = ? WHERE session_id = ?", (info['retries'], session_id))
        self.conn.commit()
        self.rate_limit_tracker[session_id] = info
        return info['retries']

    def close(self):
        self.conn.close()


# Example usage (for testing)
if __name__ == "__main__":
    db = ChatDatabase()
    sid = db.get_or_create_session("Test Chat")
    db.add_message(sid, "human", "Hello!")
    db.add_message(sid, "assistant", "Hi there!")
    print("Sessions:", db.list_sessions())
    print("History:", db.get_session_history(sid))
    print("Msg count:", db.get_message_count(sid))
    db.close()

