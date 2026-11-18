import unittest

from backend.minidb.sql import parse_sql as parse_relational
from backend.minidb.sql_multimodal import is_multimodal, parse_multimodal_sql as parse


class MultimodalSQLTests(unittest.TestCase):
    def test_create_indexes_and_set(self):
        self.assertEqual(parse('CREATE INDEX idx_geo ON locales (ubicacion) USING RTREE;'),
                         dict(action='create_index',table='locales',name='idx_geo',column='ubicacion',kind='RTREE',options={}))
        ast = parse("CREATE INDEX idx_txt ON documentos(cuerpo) USING INVERTED WITH(scoring='tfidf_cosine')")
        self.assertEqual(ast['options'], {'scoring':'TFIDF_COSINE'})
        ast = parse("CREATE INDEX idx_vec ON multimedia(embedding) USING IVF WITH(lists=100,metric='COSINE')")
        self.assertEqual(ast['options'], {'lists':100,'metric':'COSINE'})
        self.assertEqual(parse('SET ivf.probes = 5;'), dict(action='set',table=None,parameter='ivf.probes',value=5))

    def test_text_ranking_and_boolean_payload(self):
        ast = parse("SELECT id, titulo, COSINE_SCORE() AS relevancia FROM documentos WHERE cuerpo MATCH 'aprendizaje AND profundo AND NOT basico' ORDER BY relevancia DESC LIMIT 10;")
        self.assertEqual(ast['columns'], ['id','titulo',dict(kind='cosine_score',alias='relevancia')])
        self.assertEqual(ast['where'],('MATCH','cuerpo','aprendizaje AND profundo AND NOT basico'))
        self.assertEqual(ast['order'],('relevancia',True))
        self.assertEqual(ast['limit'],10)
        self.assertTrue(is_multimodal(ast))

    def test_vector_projection_hybrid_and_metric(self):
        for operator,metric in [('<=>','COSINE'),('<->','L2')]:
            ast = parse(f"SELECT id, embedding {operator} '[0.12,-0.34,5e-2]' AS distancia FROM multimedia WHERE categoria='jazz' AND precio <= 250 ORDER BY distancia ASC LIMIT 5")
            self.assertEqual(ast['columns'][1],dict(kind='distance',column='embedding',metric=metric,vector=[.12,-.34,.05],alias='distancia'))
            self.assertEqual(ast['where'],('AND',('=','categoria','jazz'),('<=','precio',250)))
            self.assertEqual(ast['order'],('distancia',False))

    def test_spatial_predicates_and_precedence(self):
        ast=parse('SELECT id FROM locales WHERE ubicacion WITHIN_RADIUS((-77.0282,-12.1354),2.5) AND precio<20 OR tipo=1')
        self.assertEqual(ast['where'],('OR',('AND',('WITHIN_RADIUS','ubicacion',((-77.0282,-12.1354),2.5)),('<','precio',20)),('=','tipo',1)))
        self.assertEqual(parse('SELECT * FROM locales WHERE ubicacion KNN((0,1),5)')['where'],('KNN','ubicacion',((0.,1.),5)))
        self.assertEqual(parse('SELECT * FROM locales WHERE ubicacion WITHIN_WINDOW((-1,-2),(3,4))')['where'],('WITHIN_WINDOW','ubicacion',((-1.,-2.),(3.,4.))))
        ast=parse("SELECT * FROM locales WHERE (precio=3 OR precio=5) AND cuerpo MATCH 'rock OR jazz'")
        self.assertEqual(ast['where'][0],'AND')
        self.assertEqual(ast['where'][1][0],'OR')

    def test_relational_regressions_and_keywords_in_literals(self):
        statements=[
            'CREATE TABLE empleados (id INT PRIMARY KEY, nombre CHAR(30), salario FLOAT) USING HEAP;',
            'CREATE INDEX idx ON empleados(id) USING BTREE;',
            'CREATE INDEX idx ON empleados(id) USING HASH;',
            "INSERT INTO empleados VALUES (1,'MATCH RTREE IVF COSINE_SCORE',100.0);",
            "SELECT id,nombre FROM empleados WHERE nombre='MATCH AND OR <=>' AND (id>=1 OR salario<500) ORDER BY id DESC LIMIT 10;",
            "DELETE FROM empleados WHERE nombre='SET ivf.probes=5;';",
            "SELECT * FROM empleados WHERE nombre='O''Brien; SELECT x FROM y'; -- comment MATCH\n",
        ]
        for sql in statements:
            with self.subTest(sql=sql):
                ast=parse(sql)
                self.assertEqual(ast,parse_relational(sql))
                self.assertFalse(is_multimodal(ast))

    def test_invalid_vectors(self):
        for literal in ['[]','[true,2]','["1",2]','[[1],2]','[NaN,2]','[Infinity]','[1e999]','{}','[1,...]']:
            with self.subTest(literal=literal),self.assertRaises(ValueError):
                parse(f"SELECT embedding <=> '{literal}' AS dist FROM audios")
        with self.assertRaises(ValueError):
            parse('SELECT embedding <=> 12 FROM audios')

    def test_invalid_configs_and_spatial_values(self):
        statements=[
            'SET ivf.probes=0','SET ivf.probes=1.5','SET other.probes=2','SET ivf.lists=5',
            'CREATE INDEX x ON t(e) USING IVF WITH(lists=0)',
            "CREATE INDEX x ON t(e) USING IVF WITH(metric='HAMMING')",
            "CREATE INDEX x ON t(e) USING IVF WITH(lists=3,lists=4)",
            "CREATE INDEX x ON t(e) USING INVERTED WITH(scoring='ts_rank')",
            'CREATE INDEX x ON t(e) USING RTREE WITH(lists=3)',
            'SELECT * FROM t WHERE p WITHIN_RADIUS((0,0),-1)',
            'SELECT * FROM t WHERE p WITHIN_RADIUS((1e999,0),1)',
            "SELECT * FROM t WHERE p WITHIN_RADIUS(('1',0),1)",
            'SELECT * FROM t WHERE p KNN((0,0),0)',
            'SELECT * FROM t WHERE p KNN((0,0),2.5)',
            'SELECT * FROM t WHERE p WITHIN_WINDOW((5,0),(1,3))',
            'SELECT * FROM t WHERE body MATCH 7',
            'SELECT * FROM t LIMIT -1','SELECT * FROM t LIMIT 1.5',
            'SELECT * FROM t; SELECT * FROM other;',
            'SET ivf.probes=5; DELETE FROM t;',
            "SELECT embedding '<=>' '[1,2]' FROM audios",
            "SELECT * FROM t WHERE p 'WITHIN_RADIUS'((0,0),1)",
        ]
        for sql in statements:
            with self.subTest(sql=sql),self.assertRaises(ValueError):
                parse(sql)

    def test_escaped_quotes_comments_and_default_aliases(self):
        ast=parse("SELECT COSINE_SCORE() FROM docs WHERE body MATCH 'l''art AND music; OR rock' -- WITH(foo=3)\nLIMIT 0;")
        self.assertEqual(ast['columns'],[dict(kind='cosine_score',alias='score')])
        self.assertEqual(ast['where'][2],"l'art AND music; OR rock")
        self.assertEqual(ast['limit'],0)
        self.assertEqual(parse("SELECT vector <=> '[1,2]' FROM docs")['columns'][0]['alias'],'distance')


if __name__=='__main__':
    unittest.main()
