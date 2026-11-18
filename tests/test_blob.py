from backend.minidb.blob import BlobStore
from backend.minidb.io import DiskCounter


def test_paged_blob_reopen_and_io(tmp_path):
    counter = DiskCounter()
    path = tmp_path/'objects.bin'
    blob = BlobStore(path, counter=counter)
    payload = bytes(range(256))*100
    head = blob.append(payload)
    empty = blob.append(b'')
    blob.close()
    blob = BlobStore(path, counter=counter)
    counter.reset()
    assert blob.read(head) == payload
    assert counter.disk_reads == (len(payload)+blob.capacity-1)//blob.capacity
    assert blob.read(empty) == b''
    assert counter.disk_writes == 0
    blob.close()
