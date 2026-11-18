"""Transferencias físicas de páginas; no existe un buffer de páginas oculto."""
from __future__ import annotations

import os
from pathlib import Path


class DiskCounter:
    """Cuenta cada transferencia física completa, incluidos los metadatos."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.disk_reads = 0
        self.disk_writes = 0
        self.by_role = {role: {"disk_reads": 0, "disk_writes": 0} for role in ("data", "index")}

    def _record(self, operation, role):
        name = "disk_reads" if operation == "read" else "disk_writes"
        setattr(self, name, getattr(self, name) + 1)
        self.by_role.setdefault(role, {"disk_reads": 0, "disk_writes": 0})[name] += 1

    def snapshot(self):
        return {"disk_reads": self.disk_reads, "disk_writes": self.disk_writes,
                "total": self.disk_reads + self.disk_writes,
                "by_role": {name: dict(counts) for name, counts in self.by_role.items()}}


class DiskFile:
    """Archivo binario de páginas fijas, abierto sin buffering de Python."""

    def __init__(self, path, page_size=4096, counter=None, role="data"):
        if not isinstance(page_size, int) or page_size < 128:
            raise ValueError("El tamaño de página debe ser un entero >= 128")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.page_size = page_size
        self.counter = counter if counter is not None else DiskCounter()
        self.role = role
        self._handle = open(self.path, "r+b" if self.path.exists() else "w+b", buffering=0)
        if os.fstat(self._handle.fileno()).st_size % page_size:
            self._handle.close()
            raise ValueError("Archivo truncado: su tamaño no es múltiplo de página")

    @property
    def page_count(self):
        return os.fstat(self._handle.fileno()).st_size // self.page_size

    def read(self, page_id):
        if not isinstance(page_id, int) or page_id < 0:
            raise ValueError("Identificador de página inválido")
        self._handle.seek(page_id * self.page_size)
        data = self._handle.read(self.page_size)
        if len(data) != self.page_size:
            raise EOFError(f"Página {page_id} inexistente o incompleta")
        self.counter._record("read", self.role)
        return bytearray(data)

    def write(self, page_id, data):
        if not isinstance(page_id, int) or page_id < 0:
            raise ValueError("Identificador de página inválido")
        if len(data) != self.page_size:
            raise ValueError("Cada escritura debe contener exactamente una página")
        if page_id > self.page_count:
            raise ValueError("No se permiten huecos entre páginas")
        self._handle.seek(page_id * self.page_size)
        if self._handle.write(data) != self.page_size:
            raise OSError("Escritura de página incompleta")
        self.counter._record("write", self.role)

    def allocate(self, data=None):
        page_id = self.page_count
        self.write(page_id, bytes(self.page_size) if data is None else data)
        return page_id

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
