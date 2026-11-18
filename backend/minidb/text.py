"""Índice invertido binario paginado y CosineScore; sin motores externos."""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
import re
import struct
import unicodedata

from .io import DiskCounter, DiskFile
from .blob import BlobStore

_HEADER = struct.Struct('<4sIHHiiI')
_VOCAB = struct.Struct('<256sIIQ')
_POSTING = struct.Struct('<QI')
_NORM = struct.Struct('<Qd')
_DOCMAP = struct.Struct('<QQ')
_MAGIC = b'TXT1'


class TextNormalizer:
    """La modalidad auto aplica ambos stemmers siempre, también a la consulta."""
    def __init__(self, language='auto'):
        from nltk.stem.snowball import SnowballStemmer
        if language not in ('auto', 'spanish', 'english', 'none'):
            raise ValueError('Idioma debe ser auto, spanish, english o none')
        self.language = language
        root = Path(__file__).resolve().parents[2] / 'data'
        languages = ('es', 'en') if language in ('auto', 'none') else ('es',) if language == 'spanish' else ('en',)
        self.stopwords = set()
        for lang in languages:
            self.stopwords.update(self.fold(word) for word in (root / f'stopwords_{lang}.txt').read_text().split())
        self.stemmers = [SnowballStemmer(lang) for lang in (('spanish', 'english') if language == 'auto' else (language,))] if language != 'none' else []

    @staticmethod
    def fold(text):
        return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if not unicodedata.combining(c))

    def __call__(self, text):
        if not isinstance(text, str):
            raise ValueError('El documento debe ser texto')
        tokens = []
        for word in re.findall(r'[^\W_]+', self.fold(text), re.UNICODE):
            if word in self.stopwords:
                continue
            for stemmer in self.stemmers:
                word = stemmer.stem(word)
            if word and len(word.encode('utf8')) <= 256:
                tokens.append(word)
        return tokens


class _Records:
    def __init__(self, path, record, counter, page_size, create=False, metadata=None, role='index'):
        self.record, self.page_size = record, page_size
        self.capacity = (page_size - _HEADER.size) // record.size
        if self.capacity < 1:
            raise ValueError('La página no admite el registro del diccionario')
        if create:
            Path(path).write_bytes(b'')
        self.file = DiskFile(path, page_size, counter, role)
        self.metadata = metadata or {}
        self.count = 0
        self.buffer = None
        self.buffer_count = 0
        if create:
            self.file.allocate()
        else:
            data = self.file.read(0)
            self._check(data, 0)
            size = struct.unpack_from('<I', data, _HEADER.size)[0]
            self.metadata = json.loads(bytes(data[_HEADER.size + 4:_HEADER.size + 4 + size]))
            if self.metadata['page_size'] != page_size or self.metadata['record_size'] != record.size:
                raise ValueError('Formato textual incompatible')
            self.count = self.metadata['count']

    def _check(self, page, pid):
        magic, actual, count, free, *_ = _HEADER.unpack_from(page)
        if magic != _MAGIC or actual != pid or not _HEADER.size <= free <= self.page_size:
            raise ValueError('Página textual inválida')
        return count

    def append(self, values):
        if self.buffer is None:
            self.buffer = bytearray(self.page_size)
            self.buffer_count = 0
        pid = self.count // self.capacity + 1
        offset = _HEADER.size + self.buffer_count * self.record.size
        self.record.pack_into(self.buffer, offset, *values)
        result = pid * self.page_size + offset
        self.buffer_count += 1
        self.count += 1
        if self.buffer_count == self.capacity:
            self._flush(pid)
        return result

    def _flush(self, pid):
        _HEADER.pack_into(self.buffer, 0, _MAGIC, pid, self.buffer_count, _HEADER.size + self.buffer_count * self.record.size,
                          pid + 1 if self.buffer_count == self.capacity else -1, pid - 1 if pid > 1 else -1, 1)
        self.file.write(pid, self.buffer)
        self.buffer = None

    def finish(self):
        if self.buffer is not None:
            self._flush((self.count - 1) // self.capacity + 1)
        elif self.count:
            pid = self.count // self.capacity
            final = self.file.read(pid)
            struct.pack_into('<i', final, 12, -1)
            self.file.write(pid, final)
        self.metadata.update(count=self.count, page_size=self.page_size, record_size=self.record.size, version=1)
        raw = json.dumps(self.metadata, ensure_ascii=True, separators=(',', ':')).encode()
        if _HEADER.size + 4 + len(raw) > self.page_size:
            raise ValueError('Metadatos textuales demasiado grandes')
        data = bytearray(self.page_size)
        _HEADER.pack_into(data, 0, _MAGIC, 0, 1, _HEADER.size + 4 + len(raw), -1, -1, 0)
        struct.pack_into('<I', data, _HEADER.size, len(raw))
        data[_HEADER.size + 4:_HEADER.size + 4 + len(raw)] = raw
        self.file.write(0, data)

    def record_at(self, position, cache=None):
        pid, slot = divmod(position, self.capacity)
        pid += 1
        page = cache.get(pid) if cache is not None else None
        if page is None:
            page = self.file.read(pid)
            self._check(page, pid)
            if cache is not None:
                if len(cache) >= 32:
                    cache.pop(next(iter(cache)))
                cache[pid] = page
        return self.record.unpack_from(page, _HEADER.size + slot * self.record.size)

    def seek_records(self, offset, count):
        pid, start = divmod(offset, self.page_size)
        slot = (start - _HEADER.size) // self.record.size
        while count:
            page = self.file.read(pid)
            active = self._check(page, pid)
            take = min(count, active - slot)
            if take <= 0:
                raise ValueError('Offset o longitud de postings inválido')
            for i in range(slot, slot + take):
                yield self.record.unpack_from(page, _HEADER.size + i * self.record.size)
            count -= take
            pid += 1
            slot = 0

    def scan(self):
        if self.count:
            yield from self.seek_records(self.page_size + _HEADER.size, self.count)

    def close(self):
        self.file.close()


def merge_postings(left, right, operator):
    """Intersección/unión/diferencia ordenada con dos punteros, O(n+m)."""
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        if a == b:
            if operator in ('AND', 'OR'):
                result.append(a)
            i += 1
            j += 1
        elif a < b:
            if operator in ('OR', 'AND-NOT'):
                result.append(a)
            i += 1
        else:
            if operator == 'OR':
                result.append(b)
            j += 1
    if operator in ('OR', 'AND-NOT'):
        result.extend(left[i:])
    if operator == 'OR':
        result.extend(right[j:])
    return result


class InvertedIndex:
    def __init__(self, path, counter=None, page_size=4096, create=False, language='auto'):
        if not 512 <= page_size <= 65535:
            raise ValueError('Página textual debe tener entre 512 y 65535 bytes')
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.counter = counter if counter is not None else DiskCounter()
        self.page_size, self.language = page_size, language
        self.normalizer = None
        self.vocab = self.postings = self.norms = None
        self.documents = self.docmap = None
        self._building_text = False
        self.document_count = 0
        if not create and (self.path / 'vocab.dic').exists():
            self._open()

    def _open(self):
        self.vocab = _Records(self.path / 'vocab.dic', _VOCAB, self.counter, self.page_size)
        self.postings = _Records(self.path / 'postings.inv', _POSTING, self.counter, self.page_size)
        self.norms = _Records(self.path / 'norms.bin', _NORM, self.counter, self.page_size)
        self.document_count = self.vocab.metadata['documents']
        self.language = self.vocab.metadata['language']
        self.normalizer = None
        if self.vocab.metadata.get('source_text', False):
            self.documents = BlobStore(self.path / 'documents.bin', counter=self.counter, page_size=self.page_size)
            self.docmap = _Records(self.path / 'docmap.dic', _DOCMAP, self.counter, self.page_size, role='data')

    def tokenize(self, text):
        if self.normalizer is None:
            self.normalizer = TextNormalizer(self.language)
        return self.normalizer(text)

    @staticmethod
    def _counts(tokens):
        counts = Counter({str(k): v for k, v in tokens.items()}) if hasattr(tokens, 'items') else Counter(str(t) for t in tokens)
        for term, tf in counts.items():
            if not term or '\x00' in term or len(term.encode('utf8')) > 256:
                raise ValueError('Cada término debe ocupar entre 1 y 256 bytes, sin nulos')
            if isinstance(tf, bool) or not isinstance(tf, int) or not 0 < tf <= 0xffffffff:
                raise ValueError('Las frecuencias deben ser enteros positivos de 32 bits')
        return counts

    def build(self, documents):
        self.close()
        source = self.path / 'documents.bin'
        source.unlink(missing_ok=True)
        blob = BlobStore(source, counter=self.counter, page_size=self.page_size)
        mapping = []
        self._building_text = True
        def prepared():
            for docid, text in documents:
                mapping.append((docid, blob.append(text.encode('utf8'))))
                yield docid, self.tokenize(text)
        try:
            self.build_tokens(prepared())
            docmap = _Records(self.path / 'docmap.dic', _DOCMAP, self.counter, self.page_size, True, role='data')
            for record in sorted(mapping):
                docmap.append(record)
            docmap.finish()
            self.documents, self.docmap = blob, docmap
        except Exception:
            blob.close()
            raise
        finally:
            self._building_text = False
        return self.stats()

    def gettext(self, docid):
        if self.docmap is None:
            raise ValueError('Este índice de tokens no contiene texto original')
        low, high, cache = 0, self.docmap.count, {}
        while low < high:
            middle = (low + high) // 2
            found, pointer = self.docmap.record_at(middle, cache)
            if found < docid:
                low = middle + 1
            else:
                high = middle
        if low < self.docmap.count:
            found, pointer = self.docmap.record_at(low, cache)
            if found == docid:
                return self.documents.read(pointer).decode('utf8')
        return None

    get_document = gettext

    def iter_docs(self):
        if self.docmap is None:
            raise ValueError('Este índice de tokens no contiene texto original')
        for docid, pointer in self.docmap.scan():
            yield docid, self.documents.read(pointer).decode('utf8')

    def build_tokens(self, documents):
        self.close()
        inverted, docids = defaultdict(list), set()
        for docid, tokens in documents:
            if isinstance(docid, bool) or not isinstance(docid, int) or not 0 <= docid < (1 << 63):
                raise ValueError('doc_id debe ser un entero no negativo signed64')
            if docid in docids:
                raise ValueError('doc_id duplicado')
            docids.add(docid)
            for term, tf in self._counts(tokens).items():
                inverted[term].append((docid, tf))
        self.document_count = len(docids)
        metadata = dict(documents=len(docids), language=self.language, source_text=self._building_text)
        self.vocab = _Records(self.path / 'vocab.dic', _VOCAB, self.counter, self.page_size, True, metadata)
        self.postings = _Records(self.path / 'postings.inv', _POSTING, self.counter, self.page_size, True)
        self.norms = _Records(self.path / 'norms.bin', _NORM, self.counter, self.page_size, True)
        squared_norms = dict.fromkeys(docids, 0.0)
        for term in sorted(inverted):
            posting = sorted(inverted[term])
            df = len(posting)
            idf = math.log10(len(docids) / df)
            offset = None
            for docid, tf in posting:
                current = self.postings.append((docid, tf))
                if offset is None:
                    offset = current
                squared_norms[docid] += ((1 + math.log10(tf)) * idf) ** 2
            self.vocab.append((term.encode('utf8'), df, df * _POSTING.size, offset))
        for docid in sorted(docids):
            self.norms.append((docid, math.sqrt(squared_norms[docid])))
        for file in (self.vocab, self.postings, self.norms):
            file.finish()
        return self.stats()

    def _lookup(self, term):
        if self.vocab is None:
            raise ValueError('El índice todavía no fue construido')
        low, high = 0, self.vocab.count
        cache = {}
        while low < high:
            middle = (low + high) // 2
            row = self.vocab.record_at(middle, cache)
            found = row[0].rstrip(b'\0').decode('utf8')
            if found < term:
                low = middle + 1
            else:
                high = middle
        if low < self.vocab.count:
            row = self.vocab.record_at(low, cache)
            if row[0].rstrip(b'\0').decode('utf8') == term:
                return row[1:]
        return None

    def get_postings(self, term):
        entry = self._lookup(str(term))
        return list(self.postings.seek_records(entry[2], entry[0])) if entry else []

    def _document_norms(self, ids):
        wanted = sorted(ids)
        if len(wanted) > 32:
            position = 0
            for docid, norm in self.norms.scan():
                while position < len(wanted) and wanted[position] < docid:
                    position += 1
                if position == len(wanted):
                    return
                if wanted[position] == docid:
                    yield docid, norm
                    position += 1
        else:
            cache = {}
            for docid in wanted:
                low, high = 0, self.norms.count
                while low < high:
                    middle = (low + high) // 2
                    found, norm = self.norms.record_at(middle, cache)
                    if found < docid:
                        low = middle + 1
                    else:
                        high = middle
                if low < self.norms.count:
                    found, norm = self.norms.record_at(low, cache)
                    if found == docid:
                        yield found, norm

    def query_tokens(self, tokens, k=10):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError('k debe ser un entero positivo')
        counts = self._counts(tokens)
        scores, query_squared = defaultdict(float), 0.0
        for term, tf_query in counts.items():
            entry = self._lookup(term)
            if not entry:
                continue
            df, length, offset = entry
            idf = math.log10(self.document_count / df)
            weight = (1 + math.log10(tf_query)) * idf
            if weight == 0:
                continue
            query_squared += weight * weight
            for docid, tf in self.postings.seek_records(offset, df):
                scores[docid] += weight * (1 + math.log10(tf)) * idf
        if not query_squared:
            return []
        query_norm = math.sqrt(query_squared)
        top = []
        for docid, norm in self._document_norms(scores):
            if not norm:
                continue
            score = scores[docid] / (norm * query_norm)
            pair = (score, -docid)
            if len(top) < k:
                heapq.heappush(top, pair)
            elif pair > top[0]:
                heapq.heapreplace(top, pair)
        return [(-negative_id, score) for score, negative_id in sorted(top, reverse=True)]

    def boolean(self, expression):
        raw = re.findall(r'AND-NOT|[^\s()]+|[()]', expression, flags=re.I)
        tokens, previous = [], None
        for item in raw:
            item = item.upper() if item.upper() in ('AND', 'OR', 'NOT', 'AND-NOT') else item
            if item == 'NOT' and tokens and tokens[-1] == 'AND':
                tokens[-1] = 'AND-NOT'
                previous = 'AND-NOT'
                continue
            operand = item not in ('AND', 'OR', 'AND-NOT', 'NOT', '(', ')')
            if (operand or item == '(') and previous in ('value', ')'):
                tokens.append('AND')
            tokens.append(item)
            previous = 'value' if operand else item
        if not tokens:
            return []
        position = 0

        def atom():
            nonlocal position
            if position >= len(tokens):
                raise ValueError('Consulta booleana incompleta')
            token = tokens[position]
            position += 1
            if token == '(':
                result = expression_or()
                if position >= len(tokens) or tokens[position] != ')':
                    raise ValueError('Paréntesis booleano sin cerrar')
                position += 1
                return result
            if token in ('AND', 'OR', 'AND-NOT', 'NOT', ')'):
                raise ValueError('Operador booleano sin término')
            terms = self.tokenize(token)
            result = None
            for term in terms:
                ids = [docid for docid, _ in self.get_postings(term)]
                result = ids if result is None else merge_postings(result, ids, 'AND')
            return result or []

        def expression_and():
            nonlocal position
            result = atom()
            while position < len(tokens) and tokens[position] in ('AND', 'AND-NOT'):
                operation = tokens[position]
                position += 1
                result = merge_postings(result, atom(), operation)
            return result

        def expression_or():
            nonlocal position
            result = expression_and()
            while position < len(tokens) and tokens[position] == 'OR':
                position += 1
                result = merge_postings(result, expression_and(), 'OR')
            return result

        result = expression_or()
        if position != len(tokens):
            raise ValueError('Consulta booleana inválida')
        return result

    def query(self, text, k=10, mode='rank'):
        if mode == 'boolean':
            return [(docid, 1.0) for docid in self.boolean(text)]
        if mode != 'rank':
            raise ValueError('Modo debe ser rank o boolean')
        return self.query_tokens(self.tokenize(text), k)

    def stats(self):
        return dict(documents=self.document_count, terms=self.vocab.count if self.vocab else 0,
                    postings=self.postings.count if self.postings else 0, page_size=self.page_size,
                    bytes=sum(file.file.page_count * self.page_size for file in (self.vocab, self.postings, self.norms) if file))

    def close(self):
        for name in ('vocab', 'postings', 'norms', 'documents', 'docmap'):
            file = getattr(self, name, None)
            if file is not None:
                file.close()
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
