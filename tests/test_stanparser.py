from hboms.deparse import deparse_stmt
from hboms.stanparser import parser

class TestStanParser:
    def test_decl_assign_stmt(self):
        code_str = "int x = 3;"
        AST = parser.parse(code_str)
        stmt_strs = [deparse_stmt(stmt) for stmt in AST]
        assert stmt_strs == [code_str]
            
    def test_if_stmt(self):
        code_str = "if ( x > 0 ) break;"
        AST = parser.parse(code_str)
        stmt_strs = [deparse_stmt(stmt) for stmt in AST]
        assert stmt_strs == [code_str]
        
    def test_for_loop(self):
        code_str = "for ( i in 1:10 ) y += x^2;"
        AST = parser.parse(code_str)
        stmt_strs = [deparse_stmt(stmt) for stmt in AST]
        assert stmt_strs == [code_str]
        
    def test_sampling_stmt(self):
        code_str = "X ~ beta_binomial(alpha, beta);"
        AST = parser.parse(code_str)
        stmt_strs = [deparse_stmt(stmt) for stmt in AST]
        assert stmt_strs == [code_str]




