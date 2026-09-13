"""Compile the official tables into a complete classified-neuron CSR graph.

Selection: every annotation with a nonempty superclass, including neurons with no
internal partners. Keep every positive supplied weight where BOTH endpoints are
in that set. There is no edge weight threshold and no sampled simulation graph.
"""
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from scipy.sparse import coo_matrix, save_npz

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / '.data' / 'malecns'


def prepare():
    start = time.perf_counter()
    raw = DIRECTORY / 'raw'
    out = DIRECTORY / 'compiled'
    out.mkdir(parents=True, exist_ok=True)
    annotations = feather.read_table(raw / 'body-annotations-male-cns-v1.0-minconf-0.5.feather')
    selection = pc.and_(pc.is_valid(annotations['superclass']), pc.not_equal(annotations['superclass'], ''))
    neurons = annotations.filter(selection).sort_by([('bodyId', 'ascending')])
    ids = neurons['bodyId'].to_numpy()
    assert len(np.unique(ids)) == len(ids), 'Duplicate neuron identifiers'
    print(f'Classified neurons: {len(ids):,}', flush=True)
    nt = feather.read_table(raw / 'body-neurotransmitters-male-cns-v1.0.feather')
    nt = nt.filter(pc.is_in(nt['body'], value_set=neurons['bodyId']))
    nt_by_id = {int(r['body']): r for r in nt.to_pylist()}
    positions = np.zeros((len(ids), 3), dtype=np.float32)
    location_kind = np.zeros(len(ids), dtype=np.uint8)
    metadata = []
    for i, row in enumerate(neurons.to_pylist()):
        location = row['somaLocation'] or row['tosomaLocation']
        if location and len(location) == 3:
            positions[i] = np.asarray(location, dtype=np.float32) * .008  # 8 nm -> micrometres
            location_kind[i] = 1 if row['somaLocation'] else 2
        transmission = nt_by_id.get(int(row['bodyId']), {})
        transmitter = transmission.get('consensus_nt') or transmission.get('predicted_nt') or 'unknown'
        metadata.append({
            'index': i, 'id': int(row['bodyId']), 'type': row['type'] or '',
            'name': row['instance'] or row['type'] or str(row['bodyId']),
            'class': row['superclass'], 'side': row['somaSide'] or row['rootSide'] or '',
            'status': row['status'] or '', 'neurotransmitter': transmitter,
            'nt_source': 'consensus' if transmission.get('consensus_nt') else 'prediction',
            'position_kind': int(location_kind[i]),
        })
    # Column vectors may be large: process Arrow batches rather than building a
    # pandas dataframe for all 151 million fragment-to-fragment rows.
    source_path = raw / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    parts_pre, parts_post, parts_weights = [], [], []
    all_rows, all_contacts, kept_rows, contacts = 0, 0, 0, 0
    with pa.memory_map(str(source_path), 'r') as source:
        reader = pa.ipc.open_file(source)
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            pre = batch.column('body_pre').to_numpy()
            post = batch.column('body_post').to_numpy()
            weight = batch.column('weight').to_numpy()
            if np.any(weight <= 0):
                raise ValueError('Unexpected non-positive source weight')
            a = np.searchsorted(ids, pre)
            b = np.searchsorted(ids, post)
            valid = (a < len(ids)) & (b < len(ids))
            a_safe, b_safe = np.minimum(a, len(ids) - 1), np.minimum(b, len(ids) - 1)
            valid &= (ids[a_safe] == pre) & (ids[b_safe] == post)
            parts_pre.append(a[valid].astype(np.int32))
            parts_post.append(b[valid].astype(np.int32))
            parts_weights.append(weight[valid].astype(np.int32))
            all_rows += len(pre)
            all_contacts += int(weight.sum())
            kept_rows += int(valid.sum())
            contacts += int(weight[valid].sum())
            if batch_index % 200 == 0:
                print(f'Batch {batch_index}/{reader.num_record_batches}: {kept_rows:,} connections', flush=True)
    pres = np.concatenate(parts_pre)
    posts = np.concatenate(parts_post)
    weights = np.concatenate(parts_weights)
    del parts_pre, parts_post, parts_weights
    graph = coo_matrix((weights, (pres, posts)), shape=(len(ids), len(ids)), dtype=np.int32).tocsr()
    graph.sum_duplicates()
    graph.sort_indices()
    assert int(graph.sum()) == contacts
    assert graph.shape == (len(ids), len(ids))
    save_npz(out / 'connections.npz', graph, compressed=False)
    np.save(out / 'positions.npy', positions)
    np.save(out / 'location-kind.npy', location_kind)
    np.save(out / 'ids.npy', ids)
    (out / 'neurons.json').write_text(json.dumps(metadata, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    manifest = {
        'dataset': 'MaleCNS v1.0', 'source': 'https://male-cns.janelia.org/download/',
        'license': 'CC-BY-4.0', 'paper': 'https://doi.org/10.1016/j.cell.2026.08.015',
        'selection': 'all annotations with nonempty superclass; both endpoints selected; all provided positive weights',
        'neuron_count': len(ids), 'connection_count': int(graph.nnz), 'synaptic_contacts': contacts,
        'raw_annotation_rows': annotations.num_rows, 'raw_connection_rows': all_rows,
        'raw_synaptic_contacts': all_contacts, 'source_confidence_threshold': 0.5,
        'soma_positions': int((location_kind == 1).sum()),
        'soma_attachment_positions': int((location_kind == 2).sum()),
        'missing_positions': int((location_kind == 0).sum()),
        'matrix_bytes': int(graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes),
        'compile_seconds': round(time.perf_counter() - start, 2),
        'simulation_sampling': False,
        'position_unit': 'micrometres; official 8 nm voxel coordinates multiplied by 0.008',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
