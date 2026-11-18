"""Objetos de longitud variable distribuidos en cadenas de páginas binarias."""

import struct
from pathlib import Path

from .io import DiskFile, DiskCounter


HEADER = struct.Struct('<4sIIIQqq')


class BlobStore:
    def __init__(self, path, counter=None, page_size=4096):
        self.counter = counter if counter is not None else DiskCounter()
        self.page_size = page_size
        self.capacity = page_size - HEADER.size
        self.file = DiskFile(Path(path), page_size, self.counter, role='data')
        if not self.file.page_count:
            page = bytearray(page_size)
            HEADER.pack_into(page, 0, b'BLB2', 0, 0, HEADER.size, page_size, -1, -1)
            self.file.allocate(page)
        else:
            meta = self.file.read(0)
            magic, pid, count, free, stored_size, nxt, prev = HEADER.unpack_from(meta)
            if magic != b'BLB2' or pid != 0 or stored_size != page_size:
                raise ValueError('Archivo de objetos incompatible.')

    def append(self, data):
        """Esta función guarda bytes sin truncarlos y devuelve su página inicial."""
        data = bytes(data)
        head = self.file.page_count
        count = max(1, (len(data) + self.capacity - 1) // self.capacity)
        for i in range(count):
            chunk = data[i*self.capacity:(i+1)*self.capacity]
            pid = head + i
            page = bytearray(self.page_size)
            HEADER.pack_into(page, 0, b'BLB2', pid, 1, HEADER.size + len(chunk), len(data),
                             pid + 1 if i + 1 < count else -1, pid - 1 if i else -1)
            page[HEADER.size:HEADER.size + len(chunk)] = chunk
            self.file.allocate(page)
        return head

    def read(self, head):
        if not isinstance(head, int) or head < 1:
            raise ValueError('Referencia de objeto inválida.')
        chunks, pid, visited, expected = [], head, set(), None
        while pid >= 0:
            if pid in visited:
                raise ValueError('Cadena de objeto circular.')
            visited.add(pid)
            page = self.file.read(pid)
            magic, actual, count, free, length, following, previous = HEADER.unpack_from(page)
            used = free - HEADER.size
            if magic != b'BLB2' or actual != pid or count != 1 or not 0 <= used <= self.capacity:
                raise ValueError('Página de objeto inválida.')
            expected = length if expected is None else expected
            if length != expected:
                raise ValueError('Longitud de objeto inconsistente.')
            chunks.append(bytes(page[HEADER.size:HEADER.size + used]))
            pid = following
        value = b''.join(chunks)
        if len(value) != expected:
            raise ValueError('Objeto incompleto.')
        return value

    def close(self):
        self.file.close()
