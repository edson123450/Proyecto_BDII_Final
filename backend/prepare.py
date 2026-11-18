"""Generación reproducible del conjunto tabular de demostración."""

import argparse
from pathlib import Path

from .minidb.engine import Database


COLUMNS = [
    {'name': 'id', 'type': 'INT', 'primary': True},
    {'name': 'nombre', 'type': 'CHAR', 'size': 30},
    {'name': 'dept', 'type': 'CHAR', 'size': 20},
    {'name': 'salario', 'type': 'FLOAT'},
]


def prepare(directory, count=100000):
    db = Database(directory)
    try:
        for name, organization in [('empleados', 'HEAP'), ('empleados_seq', 'SEQUENTIAL')]:
            if name in db.catalog:
                continue
            db.create_table(name, COLUMNS, organization)
            rows = ({'id': i, 'nombre': f'Empleado {i:06d}', 'dept': ['Ventas', 'Sistemas', 'Operaciones', 'Finanzas'][i % 4], 'salario': float(1800 + (i * 37) % 12000)} for i in range(1, count + 1))
            file = db.tables[name]
            if organization == 'SEQUENTIAL':
                file.bulk_load(rows)
            else:
                for row in rows:
                    file.insert(row)
            if name == 'empleados':
                db.create_index(name, 'idx_emp_id', 'id', 'BTREE')
                db.create_index(name, 'idx_emp_dept', 'dept', 'HASH')
            print(f'{name}: {count:,} registros preparados.', flush=True)
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('data/runtime'))
    parser.add_argument('--count', type=int, default=100000)
    args = parser.parse_args()
    prepare(args.directory, args.count)
