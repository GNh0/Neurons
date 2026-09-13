"""A persistent plastic concept network, separate from measured fly connectivity."""
import itertools
import json
import time

import numpy as np


class LearningNetwork:
    def __init__(self, database):
        self.db = database
        self.revision = 0
        self.last_change = {}
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS concept_nodes (
                id INTEGER PRIMARY KEY, label TEXT UNIQUE NOT NULL,
                activity REAL NOT NULL DEFAULT 0, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS concept_edges (
                a INTEGER NOT NULL, b INTEGER NOT NULL, weight REAL NOT NULL,
                uses INTEGER NOT NULL DEFAULT 1, PRIMARY KEY(a,b));
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id INTEGER NOT NULL, node_id INTEGER NOT NULL,
                PRIMARY KEY(memory_id,node_id));
        ''')

    def learn(self, memory_id, concepts, allow_growth=True):
        before_nodes = self.db.execute('SELECT count(*) FROM concept_nodes').fetchone()[0]
        before_edges = self.db.execute('SELECT count(*) FROM concept_edges').fetchone()[0]
        added_nodes = added_edges = 0
        selected = []
        for label in list(dict.fromkeys(concepts))[:18]:
            row = self.db.execute('SELECT id FROM concept_nodes WHERE label=?', (label,)).fetchone()
            if row:
                identifier = row[0]
            elif allow_growth and before_nodes + added_nodes < 10000:
                identifier = self.db.execute('INSERT INTO concept_nodes(label,created) VALUES (?,?)',
                                             (label, time.time())).lastrowid
                added_nodes += 1
            else:
                continue
            selected.append(identifier)
            self.db.execute('INSERT OR IGNORE INTO memory_concepts VALUES (?,?)', (memory_id, identifier))
            self.db.execute('UPDATE concept_nodes SET activity=1 WHERE id=?', (identifier,))
        for a, b in itertools.combinations(sorted(selected), 2):
            changed = self.db.execute('UPDATE concept_edges SET weight=min(3,weight+0.1),uses=uses+1 WHERE a=? AND b=?', (a, b))
            if not changed.rowcount and allow_growth and before_edges + added_edges < 200000:
                self.db.execute('INSERT INTO concept_edges(a,b,weight) VALUES (?,?,0.2)', (a, b))
                added_edges += 1
        self.revision += 1
        self.last_change = {
            'new_nodes': self.db.execute('SELECT count(*) FROM concept_nodes').fetchone()[0] - before_nodes,
            'new_edges': self.db.execute('SELECT count(*) FROM concept_edges').fetchone()[0] - before_edges,
            'activated_concepts': len(selected), 'rule': 'bounded coactivation reinforcement',
            'biological_weights_changed': False}
        return dict(self.last_change)

    def recall(self, concepts):
        nodes = self.db.execute('SELECT id,label FROM concept_nodes ORDER BY id').fetchall()
        if not nodes:
            return {}
        index = {identifier: i for i, (identifier, _) in enumerate(nodes)}
        seed = np.array([1.0 if label in concepts else 0.0 for _, label in nodes], dtype=np.float32)
        activity = seed.copy()
        rows = self.db.execute('SELECT a,b,weight FROM concept_edges').fetchall()
        if rows and np.any(seed):
            a = np.array([index[r[0]] for r in rows], dtype=np.int32)
            b = np.array([index[r[1]] for r in rows], dtype=np.int32)
            weight = np.array([r[2] for r in rows], dtype=np.float32)
            degree = np.bincount(a, weights=weight, minlength=len(nodes)) + np.bincount(b, weights=weight, minlength=len(nodes))
            for _ in range(4):
                signal = np.bincount(b, weights=activity[a]*weight/np.maximum(degree[a], 1), minlength=len(nodes))
                signal += np.bincount(a, weights=activity[b]*weight/np.maximum(degree[b], 1), minlength=len(nodes))
                activity = np.minimum(1, seed + .65*signal).astype(np.float32)
        scores = {}
        for memory, node in self.db.execute('SELECT memory_id,node_id FROM memory_concepts'):
            scores.setdefault(memory, []).append(float(activity[index[node]]))
        self.db.executemany('UPDATE concept_nodes SET activity=? WHERE id=?',
                            [(float(activity[i]), identifier) for i, (identifier, _) in enumerate(nodes)])
        self.db.commit()
        self.revision += 1
        return {memory: sum(values)/len(values) for memory, values in scores.items()}

    def feedback(self, memory_id, positive):
        ids = sorted(r[0] for r in self.db.execute('SELECT node_id FROM memory_concepts WHERE memory_id=?', (memory_id,)))
        for a, b in itertools.combinations(ids, 2):
            self.db.execute('UPDATE concept_edges SET weight=max(0.02,min(3,weight+?)) WHERE a=? AND b=?',
                            (.08 if positive else -.12, a, b))
        self.revision += 1

    def summary(self):
        return {'nodes': self.db.execute('SELECT count(*) FROM concept_nodes').fetchone()[0],
                'edges': self.db.execute('SELECT count(*) FROM concept_edges').fetchone()[0],
                'revision': self.revision, 'last_change': self.last_change}

    def graph(self):
        # Only visualization is capped. Recall and learning use the full stored graph.
        rows = self.db.execute('SELECT id,label,activity FROM concept_nodes ORDER BY id DESC LIMIT 1500').fetchall()
        ids = {r[0] for r in rows}
        edges = [list(r) for r in self.db.execute('SELECT a,b,weight FROM concept_edges ORDER BY weight DESC')
                 if r[0] in ids and r[1] in ids][:6000]
        return {**self.summary(), 'items': [{'id': r[0], 'label': r[1], 'activity': r[2]} for r in rows],
                'connections': edges, 'node_display_limit': 1500, 'edge_display_limit': 6000,
                'kind': 'artificial_plastic_concept_network'}
