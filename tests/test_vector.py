import numpy as np
import pytest
from backend.minidb.io import DiskCounter
from backend.minidb.vector import VectorStore, IVFIndex


@pytest.mark.parametrize('metric', ['COSINE', 'L2'])
def test_ivf_full_probes_matches_exact_and_reopens(tmp_path, metric):
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(140, 16)).astype(np.float32)
    counter = DiskCounter()
    store = VectorStore(tmp_path/'vectors.bin', dimension=16, metric=metric, counter=counter, create=True)
    rids = [(i//5 + 1, i%5) for i in range(len(vectors))]
    for rid, vector in zip(rids, vectors):
        store.append(rid, vector)
    index = IVFIndex(tmp_path/'ivf', counter=counter)
    index.build(store, seed=42)
    assert index.clusters == 11
    index.close()
    store.close()
    store = VectorStore(tmp_path/'vectors.bin', counter=counter)
    index = IVFIndex(tmp_path/'ivf', counter=counter)
    for query in vectors[[3, 45, 90]]:
        exact = store.knn(query, k=10)
        approx = index.knn(query, k=10, probes=index.clusters)
        assert exact == approx
        allowed = set(rids[::3])
        assert store.knn(query, 5, allowed_rids=allowed) == index.knn(query, 5, index.clusters, allowed_rids=allowed)
        excluded = set(rids[:80])
        assert store.knn(query, 10, exclude_rids=excluded) == index.knn(query, 10, index.clusters, exclude_rids=excluded)
    counter.reset()
    index.knn(vectors[0], 10, 1)
    reads = counter.disk_reads
    assert reads >= index.clusters
    counter.reset()
    index.knn(vectors[0], 10, 1)
    assert counter.disk_reads == reads
    assert counter.disk_writes == 0
    assert counter.by_role['index']['disk_reads'] == reads
    store.close(); index.close()


def test_large_vector_pages_range_and_ties(tmp_path):
    c = DiskCounter()
    store = VectorStore(tmp_path/'v.bin', dimension=512, metric='L2', counter=c, create=True)
    assert store.capacity == 1
    vectors = [np.zeros(512, np.float32), np.ones(512, np.float32), np.zeros(512, np.float32)]
    for rid, vector in zip([(5,0),(-3,1),(2,0)], vectors):
        store.append(rid, vector)
    c.reset()
    result = store.range_search(vectors[0], 0)
    assert result == [((2,0),0.),((5,0),0.)]
    assert c.disk_reads == 3
    assert store.knn(vectors[0], 1) == [((2,0),0.)]
    with pytest.raises(ValueError): store.append((1,1), [1,2])
    with pytest.raises(ValueError): store.range_search(vectors[0], -1)
    store.close()


def test_invalid_cosine_vector(tmp_path):
    store = VectorStore(tmp_path/'v.bin', dimension=3, create=True)
    with pytest.raises(ValueError): store.append((1,0), [0,0,0])
    with pytest.raises(ValueError): store.append((1,0), [float('nan'),1,2])
    store.close()
