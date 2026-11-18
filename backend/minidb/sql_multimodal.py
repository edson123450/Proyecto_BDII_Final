"""Extensión del AST relacional para índices, texto, espacio y vectores."""
from __future__ import annotations

import json
import math

from .sql import SQLParser

_MULTIMODAL_INDEXES = {'RTREE', 'INVERTED', 'IVF'}
_SPATIAL = {'WITHIN_RADIUS', 'WITHIN_WINDOW', 'KNN'}


class MultimodalSQLParser(SQLParser):
    def _finish(self, result):
        self.accept(';')
        if self.pos != len(self.tokens):
            raise ValueError(f'Texto adicional no reconocido: {self.tokens[self.pos:]}')
        return result

    @staticmethod
    def _finite(value, label):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{label} debe ser numérico')
        try:
            valid = math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f'{label} debe ser finito')
        return value

    @staticmethod
    def _positive_integer(value, label):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{label} debe ser un entero positivo')
        return value

    def parse(self):
        if self.accept('SET'):
            namespace = self.ident()
            self.need('.')
            option = self.ident()
            self.need('=')
            value = self._positive_integer(self.literal(), 'ivf.probes')
            if (namespace, option) != ('ivf', 'probes'):
                raise ValueError('La única opción SET soportada es ivf.probes')
            return self._finish(dict(action='set', table=None, parameter='ivf.probes', value=value))
        if self.accept('SELECT'):
            return self._finish(self._select())
        # La detección utiliza tokens, nunca palabras contenidas en literales SQL.
        if len(self.tokens) >= 2 and self.tokens[0][0] == self.tokens[1][0] == 'id' and [t[1].upper() for t in self.tokens[:2]] == ['CREATE', 'INDEX']:
            self.need('CREATE')
            self.need('INDEX')
            return self._finish(self._create_index())
        self.pos = 0
        return super().parse()

    def _create_index(self):
        name = self.ident()
        self.need('ON')
        table = self.ident()
        self.need('(')
        column = self.ident()
        self.need(')')
        self.need('USING')
        kind = self.ident().upper()
        if kind not in _MULTIMODAL_INDEXES | {'BTREE', 'HASH'}:
            raise ValueError('Índice no soportado: usar BTREE, HASH, RTREE, INVERTED o IVF')
        options = {}
        if self.accept('WITH'):
            self.need('(')
            while True:
                option = self.ident()
                if option in options:
                    raise ValueError('Opción de índice repetida: '+option)
                self.need('=')
                options[option] = self.literal()
                if not self.accept(','):
                    break
            self.need(')')
        allowed = {'INVERTED': {'scoring'}, 'IVF': {'lists', 'metric'}}.get(kind, set())
        if set(options) - allowed:
            raise ValueError('Opciones incompatibles con el tipo de índice')
        if 'lists' in options:
            options['lists'] = self._positive_integer(options['lists'], 'lists')
        if 'scoring' in options:
            if not isinstance(options['scoring'], str) or options['scoring'].upper() != 'TFIDF_COSINE':
                raise ValueError('scoring debe ser TFIDF_COSINE')
            options['scoring'] = options['scoring'].upper()
        if 'metric' in options:
            if not isinstance(options['metric'], str) or options['metric'].upper() not in ('COSINE', 'L2'):
                raise ValueError('metric debe ser COSINE o L2')
            options['metric'] = options['metric'].upper()
        result = dict(action='create_index', table=table, name=name, column=column, kind=kind)
        if kind in _MULTIMODAL_INDEXES:
            result['options'] = options
        return result

    def _projection(self):
        column = self.ident()
        if column == 'cosine_score' and self.accept('('):
            self.need(')')
            alias = self.ident() if self.accept('AS') else 'score'
            return dict(kind='cosine_score', alias=alias)
        if self.peek() in ('<=>', '<->') and self.tokens[self.pos][0] == 'sym':
            operator = self.tokens[self.pos][1]
            self.pos += 1
            literal = self.literal()
            if not isinstance(literal, str):
                raise ValueError('El vector debe ser un arreglo JSON entre comillas simples')
            try:
                vector = json.loads(literal)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError('Vector JSON inválido') from exc
            if not isinstance(vector, list) or not vector or len(vector) > 4096:
                raise ValueError('El vector debe tener entre 1 y 4096 componentes')
            try:
                vector = [float(self._finite(value, 'Cada componente del vector')) for value in vector]
            except OverflowError as exc:
                raise ValueError('Componente vectorial fuera de rango') from exc
            alias = self.ident() if self.accept('AS') else 'distance'
            return dict(kind='distance', column=column, metric='COSINE' if operator == '<=>' else 'L2', vector=vector, alias=alias)
        return column

    def _select(self):
        if self.accept('*'):
            columns = ['*']
        else:
            columns = [self._projection()]
            while self.accept(','):
                columns.append(self._projection())
        self.need('FROM')
        table = self.ident()
        where = self.expression() if self.accept('WHERE') else None
        order = None
        if self.accept('ORDER'):
            self.need('BY')
            column = self.ident()
            descending = self.accept('DESC')
            if not descending:
                self.accept('ASC')
            order = column, descending
        limit = self.literal() if self.accept('LIMIT') else None
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError('LIMIT debe ser un entero no negativo')
        return dict(action='select', table=table, columns=columns, where=where, order=order, limit=limit)

    def _point(self):
        self.need('(')
        x = self._finite(self.literal(), 'La coordenada X')
        self.need(',')
        y = self._finite(self.literal(), 'La coordenada Y')
        self.need(')')
        return float(x), float(y)

    def predicate(self):
        if self.accept('('):
            result = self.expression()
            self.need(')')
            return result
        column = self.ident()
        if self.accept('MATCH'):
            text = self.literal()
            if not isinstance(text, str):
                raise ValueError('MATCH requiere texto entre comillas simples')
            return 'MATCH', column, text
        if self.peek() in _SPATIAL and self.tokens[self.pos][0] == 'id':
            operator = self.tokens[self.pos][1].upper()
            self.pos += 1
            self.need('(')
            first = self._point()
            self.need(',')
            if operator == 'WITHIN_WINDOW':
                second = self._point()
                if first[0] > second[0] or first[1] > second[1]:
                    raise ValueError('La ventana requiere mínimos no mayores que máximos')
            elif operator == 'KNN':
                second = self._positive_integer(self.literal(), 'k')
            else:
                second = self._finite(self.literal(), 'El radio')
                if second < 0:
                    raise ValueError('El radio no puede ser negativo')
                second = float(second)
            self.need(')')
            return operator, column, (first, second)
        operator = self.peek()
        if operator not in ('=', '<', '<=', '>', '>=', '!=', '<>') or self.tokens[self.pos][0] != 'sym':
            raise ValueError('Se esperaba comparación, MATCH o predicado espacial')
        self.pos += 1
        literal = self.literal()
        if isinstance(literal, (int, float)):
            self._finite(literal, 'El valor de comparación')
        return operator, column, literal


def parse_multimodal_sql(sql):
    """Retorna el AST P1, ampliando proyecciones, predicados y opciones P2."""
    return MultimodalSQLParser(sql).parse()


def is_multimodal(ast):
    """Permite delegar el AST relacional al ejecutor P1 sin buscar texto crudo."""
    if ast['action'] == 'set' or ast.get('kind') in _MULTIMODAL_INDEXES:
        return True
    if any(isinstance(column, dict) for column in ast.get('columns', [])) and ast['action'] == 'select':
        return True
    def visit(condition):
        if condition is None:
            return False
        if condition[0] in ('AND', 'OR'):
            return visit(condition[1]) or visit(condition[2])
        return condition[0] in _SPATIAL | {'MATCH'}
    return visit(ast.get('where'))


parse_sql = parse_multimodal_sql
