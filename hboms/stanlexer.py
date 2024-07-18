import ply.lex as lex

# reserved keywords
reserved = {
    "if": "IF",
    "else": "ELSE",
    "for": "FOR",
    "return": "RETURN",
    "in": "IN",
    "break": "BREAK",
    "real": "TYPE_REAL",
    "int": "TYPE_INT",
    "vector": "TYPE_VECTOR",
    "row_vector": "TYPE_ROW_VECTOR",
    "matrix": "TYPE_MATRIX",
    "complex_vector" : "TYPE_COMPLEX_VECTOR",
    "complex_row_vector" : "TYPE_COMPLEX_ROW_VECTOR",
    "complex_matrix": "TYPE_COMPLEX_MATRIX",
    "array": "TYPE_ARRAY",
    "tuple": "TYPE_TUPLE",
    "lower": "LOWER",
    "upper": "UPPER",
}

# List of token names.
tokens = [
    "COMMENT",
    "LONGCOMMENT",
    "REAL",
    "IMAG",
    "INT",
    "STRING",
    "COMMA",
    "PLUS",
    "MINUS",
    "TIMES",
    "DOTTIMES",
    "DIVIDE",
    "DOTDIVIDE",
    "IDIVIDE",
    "MODULUS",
    "TRANSPOSE",
    "ASSIGN",
    "ADDASSIGN",
    "PIPE",
    "TILDE",
    "POWER",
    "DOTPOWER",
    "OR",
    "AND",
    "EQUALS",
    "NOTEQUALS",
    "LESSEQUALS",
    "GREATEREQUALS",
    "QUESTIONMARK",
    "LPAREN",
    "RPAREN",
    "LBRACE",
    "RBRACE",
    "LESS",
    "GREATER",
    "LBRACKET",
    "RBRACKET",
    "COLON",
    "SEMICOLON",
    "ID",
] + sorted(list(reserved.values()))


# Regular expression rules for simple tokens
t_COMMA = r","
t_PLUS = r"\+"
t_MINUS = r"-"
t_TIMES = r"\*"
t_DOTTIMES = r"\.\*"
t_DIVIDE = r"/"
t_DOTDIVIDE = r"\./"
t_IDIVIDE = r"\%/\%"
t_MODULUS = r"\%"
t_TRANSPOSE = r"'"
t_ASSIGN = r"="
t_ADDASSIGN = r"\+="
t_TILDE = r"~"
t_PIPE = r"\|"
t_POWER = r"\^"
t_DOTPOWER = r"\.\^"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_AND = r"\&\&"
t_OR = r"\|\|"
t_LESS = r"<"
t_GREATER = r">"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_EQUALS = r"=="
t_NOTEQUALS = r"!="
t_LESSEQUALS = r"<="
t_GREATEREQUALS = r">="
t_QUESTIONMARK = r"\?"
t_COLON = r":"
t_SEMICOLON = r";"
t_COMMENT = r"//.*"
t_LONGCOMMENT = r"/\*((\*(?!/)|[^*])|\n)*\*/"

# A regular expression rule with some action code
def t_INT(t):
    r"\d+(?![\.eEi])"
    t.value = int(t.value)
    return t

def t_IMAG(t):
    r"(\d+\.)?\d+([eE][+-]?\d+)?i"
    t.value = float(t.value[:-1]) # remove the i...
    return t

def t_REAL(t):
    r"(\d+\.)?\d+([eE][+-]?\d+)?"
    t.value = float(t.value)
    return t

def t_STRING(t):
    r'"[^"]*"'
    t.value = t.value[1:-1]  # remove string quotation marks
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_ID(t):
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    t.type = reserved.get(t.value, "ID")  # Check for reserved words
    return t


# A string containing ignored characters (spaces and tabs)
t_ignore = " \t"

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()


def find_identifier_strings(code: str) -> list[str]:
    """
    Find all idenitfier strings in a code string.
    First tokenize the code string, and then find all
    identifier tokens. Return a list of their lexemes
    """
    lexer.input(code)
    tokens = [tok for tok in lexer]
    id_strings = [tok.value for tok in tokens if tok.type == "ID"]
    return id_strings


def find_used_names(code: str, names: list[str]) -> list[str]:
    """
    find all identifer names in `names` present in the string `code`
    The order of the returned names is the same as in the argument
    'names'
    """
    id_strings = find_identifier_strings(code)
    used_names = [name for name in names if name in id_strings]
    return used_names
