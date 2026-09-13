"""Individual state and event storage in PostgreSQL or a standalone SQLite DB."""
import base64
import ctypes
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def protect(text, decrypt=False):
    """Windows account-bound encryption. Secrets never enter repository files."""
    if os.name != 'nt':
        raise ValueError('이 연결 설정은 Windows 사용자 암호화를 사용합니다.')
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_char))]
    data = base64.b64decode(text) if decrypt else text.encode('utf-8')
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    target = Blob()
    function = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ValueError('현재 Windows 계정으로 DB 연결 설정을 읽을 수 없습니다.')
    try:
        value = ctypes.string_at(target.data, target.size)
        return value.decode('utf-8') if decrypt else base64.b64encode(value).decode('ascii')
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


def database_dsn(root):
    if os.environ.get('NEURONS_DATABASE_URL'):
        return os.environ['NEURONS_DATABASE_URL']
    config = Path(root)/'.runtime'/'database.json'
    if config.exists():
        return protect(json.loads(config.read_text(encoding='utf8'))['protected_dsn'], decrypt=True)
    return None


class IndividualStore:
    def __init__(self, directory, dsn=None):
        self.lock = threading.RLock()
        self.postgres = bool(dsn)
        self.backend = 'PostgreSQL' if self.postgres else 'SQLite'
        self.parameter = '%s' if self.postgres else '?'
        if self.postgres:
            import psycopg
            from psycopg.types.json import Jsonb
            self.json_parameter = Jsonb
            # An explicitly configured database must never silently fall back.
            self.db = psycopg.connect(dsn, connect_timeout=5, application_name='Neurons')
            with self.db.transaction():
                self.db.execute('CREATE SCHEMA IF NOT EXISTS neurons')
                self.db.execute('SET search_path TO neurons,pg_catalog')
        else:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(Path(directory)/'individuals.sqlite3', check_same_thread=False)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.json_parameter = lambda value: json.dumps(value, ensure_ascii=False)
        payload = 'JSONB' if self.postgres else 'TEXT'
        for statement in (
            f'CREATE TABLE IF NOT EXISTS individual_state (id TEXT PRIMARY KEY, updated DOUBLE PRECISION NOT NULL, payload {payload} NOT NULL)',
            f'CREATE TABLE IF NOT EXISTS individual_events (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, created DOUBLE PRECISION NOT NULL, payload {payload} NOT NULL)',
            'CREATE INDEX IF NOT EXISTS individual_events_agent_time ON individual_events(agent_id,created)',
        ):
            self.db.execute(statement)
        self.db.commit()

    def execute(self, sql, parameters=()):
        return self.db.execute(sql.replace('?', self.parameter), parameters)

    @staticmethod
    def decode(payload):
        return json.loads(payload) if isinstance(payload, str) else payload

    def put_many(self, states, events=()):
        with self.lock:
            try:
                for identifier, payload in states.items():
                    self.execute('''INSERT INTO individual_state(id,updated,payload) VALUES (?,?,?)
                        ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,payload=excluded.payload''',
                        (identifier, time.time(), self.json_parameter(payload)))
                for event in events:
                    self.execute('INSERT INTO individual_events VALUES (?,?,?,?)',
                                 (event['id'],event['agent_id'],event['created'],self.json_parameter(event['payload'])))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def all_states(self):
        with self.lock:
            rows = self.execute('SELECT id,payload FROM individual_state').fetchall()
            self.db.commit()
            return {r[0]: self.decode(r[1]) for r in rows}

    def event(self, identifier, payload):
        with self.lock:
            try:
                self.execute('INSERT INTO individual_events VALUES (?,?,?,?)',
                             (uuid.uuid4().hex, identifier, time.time(), self.json_parameter(payload)))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    def history(self, identifier, limit=30, only_lessons=False):
        with self.lock:
            condition = " AND payload->>'kind'<>'experience'" if self.postgres else " AND json_extract(payload,'$.kind')<>'experience'"
            rows = self.execute('SELECT created,payload FROM individual_events WHERE agent_id=?' + (condition if only_lessons else '') + ' ORDER BY created DESC LIMIT ?',
                                (identifier, int(limit))).fetchall()
            self.db.commit()
            return [{'created': r[0], **self.decode(r[1])} for r in rows]

    def import_sqlite(self, filename):
        """Copy a standalone world into an empty PostgreSQL store, preserving IDs.

        The source database and binary checkpoints remain untouched. Refuse to
        mix separate worlds or overwrite an existing destination.
        """
        if not self.postgres or not Path(filename).exists():
            return 0
        with self.lock, sqlite3.connect(f'{Path(filename).as_uri()}?mode=ro', uri=True) as source:
            count = source.execute('SELECT count(*) FROM individual_state').fetchone()[0]
            if not count:
                return 0
            if self.all_states():
                raise ValueError('PostgreSQL에 저장된 세계가 있어 자동으로 합치지 않았습니다.')
            try:
                for identifier, updated, payload in source.execute('SELECT id,updated,payload FROM individual_state'):
                    self.execute('INSERT INTO individual_state VALUES (?,?,?)', (identifier, updated, self.json_parameter(json.loads(payload))))
                for identifier, agent, created, payload in source.execute('SELECT id,agent_id,created,payload FROM individual_events'):
                    self.execute('INSERT INTO individual_events VALUES (?,?,?,?)', (identifier, agent, created, self.json_parameter(json.loads(payload))))
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            return count

    def close(self):
        self.db.close()
