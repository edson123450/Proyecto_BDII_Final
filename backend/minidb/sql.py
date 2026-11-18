"""Analizador del subconjunto SQL utilizado por el gestor."""

import re


TOKEN = re.compile(r"\s*(?:(--[^\n]*)|('(?:''|[^'])*')|([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z_0-9]*)|(<=>|<->|<=|>=|<>|!=|[(),;*=<>.]))")


class SQLParser:
    def __init__(self, sql):
        self.tokens = []
        pos = 0
        while pos < len(sql):
            if not sql[pos:].strip():
                break
            match = TOKEN.match(sql, pos)
            if not match:
                raise ValueError(f"Símbolo SQL no reconocido cerca de: {sql[pos:pos+35]}")
            pos = match.end()
            if match.group(1):
                continue
            if match.group(2):
                self.tokens.append(('str', match.group(2)[1:-1].replace("''", "'")))
            elif match.group(3):
                value = match.group(3)
                self.tokens.append(('num', float(value) if any(c in value.lower() for c in '.e') else int(value)))
            elif match.group(4):
                self.tokens.append(('id', match.group(4)))
            else:
                self.tokens.append(('sym', match.group(5)))
        self.pos = 0

    def peek(self):
        return str(self.tokens[self.pos][1]).upper() if self.pos < len(self.tokens) else ''

    def accept(self, word):
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] in ('id', 'sym') and self.peek() == word:
            self.pos += 1
            return True
        return False

    def need(self, word):
        if not self.accept(word):
            raise ValueError(f"Se esperaba {word}; se encontró {self.peek() or 'el final de la consulta'}.")

    def ident(self):
        if self.pos >= len(self.tokens) or self.tokens[self.pos][0] != 'id':
            raise ValueError('Se esperaba un nombre de tabla, columna o índice.')
        value = self.tokens[self.pos][1].lower()
        self.pos += 1
        return value

    def literal(self):
        if self.pos >= len(self.tokens) or self.tokens[self.pos][0] not in ('num', 'str'):
            raise ValueError('Se esperaba un número o texto entre comillas simples.')
        value = self.tokens[self.pos][1]
        self.pos += 1
        return value

    def parse(self):
        if self.accept('CREATE'):
            if self.accept('TABLE'):
                name = self.ident()
                self.need('(')
                columns = []
                while True:
                    col = {'name': self.ident(), 'type': self.ident().upper()}
                    if col['type'] == 'CHAR':
                        self.need('(')
                        col['size'] = self.literal()
                        self.need(')')
                    if self.accept('PRIMARY'):
                        self.need('KEY')
                        col['primary'] = True
                    columns.append(col)
                    if not self.accept(','):
                        break
                self.need(')')
                organization = 'HEAP'
                if self.accept('USING'):
                    organization = self.ident().upper()
                out = dict(action='create_table', table=name, columns=columns, organization=organization)
            else:
                self.need('INDEX')
                name = self.ident()
                self.need('ON')
                table = self.ident()
                self.need('(')
                column = self.ident()
                self.need(')')
                self.need('USING')
                kind = self.ident().upper()
                out = dict(action='create_index', table=table, name=name, column=column, kind=kind)
        elif self.accept('INSERT'):
            self.need('INTO')
            table = self.ident()
            self.need('VALUES')
            self.need('(')
            values = [self.literal()]
            while self.accept(','):
                values.append(self.literal())
            self.need(')')
            out = dict(action='insert', table=table, values=values)
        elif self.accept('SELECT'):
            if self.accept('*'):
                columns = ['*']
            else:
                columns = [self.ident()]
                while self.accept(','):
                    columns.append(self.ident())
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
                order = (column, descending)
            limit = self.literal() if self.accept('LIMIT') else None
            if limit is not None and (not isinstance(limit, int) or limit < 0):
                raise ValueError('LIMIT debe ser un entero no negativo.')
            out = dict(action='select', table=table, columns=columns, where=where, order=order, limit=limit)
        elif self.accept('DELETE'):
            self.need('FROM')
            table = self.ident()
            where = self.expression() if self.accept('WHERE') else None
            out = dict(action='delete', table=table, where=where)
        else:
            raise ValueError('Sentencia no soportada. Usar CREATE TABLE/INDEX, INSERT, SELECT o DELETE.')
        self.accept(';')
        if self.pos != len(self.tokens):
            raise ValueError(f'Texto adicional no reconocido: {self.tokens[self.pos:]}')
        return out

    def expression(self):
        left = self.conjunction()
        while self.accept('OR'):
            left = ('OR', left, self.conjunction())
        return left

    def conjunction(self):
        left = self.predicate()
        while self.accept('AND'):
            left = ('AND', left, self.predicate())
        return left

    def predicate(self):
        if self.accept('('):
            expression = self.expression()
            self.need(')')
            return expression
        column = self.ident()
        op = self.peek()
        if op not in ('=', '<', '<=', '>', '>=', '!=', '<>') or self.tokens[self.pos][0] != 'sym':
            raise ValueError('Se esperaba un operador de comparación.')
        self.pos += 1
        return (op, column, self.literal())


def parse_sql(sql):
    """Esta función convierte una sentencia SQL en un árbol de operaciones."""
    return SQLParser(sql).parse()


def matches(row, condition):
    if condition is None:
        return True
    op, left, right = condition
    if op == 'AND':
        return matches(row, left) and matches(row, right)
    if op == 'OR':
        return matches(row, left) or matches(row, right)
    if left not in row:
        raise ValueError(f'La columna {left} no existe.')
    value = row[left]
    if op == '=': return value == right
    if op in ('!=', '<>'): return value != right
    if op == '<': return value < right
    if op == '<=': return value <= right
    if op == '>': return value > right
    if op == '>=': return value >= right
    raise ValueError(f'Operador no soportado: {op}')


def conjuncts(condition):
    if condition is None:
        return []
    if condition[0] == 'OR':
        return []
    if condition[0] == 'AND':
        return conjuncts(condition[1]) + conjuncts(condition[2])
    return [condition]
