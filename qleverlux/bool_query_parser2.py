"""
Boolean Query Parser using PLY (Python Lex-Yacc)

This module provides a parser for boolean queries with the following features:
- Boolean operators: AND, OR, NOT (case-sensitive, capitalized)
- Parentheses for grouping: ( )
- Term lists: consecutive terms treated as separate items in a list
- Quoted strings for multi-word terms: "hello world"
- Single word terms: hello, world
- Field-qualified terms: title:fish, author:"John Doe"
- Proper operator precedence: NOT > AND > OR

Examples:
    Basic usage:
        parser = BooleanQueryParser()
        ast = parser.parse('title:fish AND author:gibson')
        result = evaluate_ast(ast, context)

    Supported query formats:
        - Simple terms: a, hello
        - Field terms: title:fish, author:"John Doe"
        - Comparitors for terms: name>X OR name<B
        - Term lists: a b c, title:hello "world test" foo
        - Quoted terms: "hello world", "machine learning"
        - Boolean operations: title:a AND author:b, x OR y, NOT title:z
        - Grouping: (title:a OR author:b) AND content:c
        - Complex: (title:a b "c d" author:e OR content:f) AND NOT tags:g
        - Relationship Chains: creator->name:Rob AND subject->broader->name:Mathematics

Author: Generated for luxql project
License: Same as parent project
"""

# This doesn't work:
# creator->(name:Rob AND name:Sanderson)

import ply.lex as lex
import ply.yacc as yacc

# Token definitions
tokens = ("AND", "OR", "NOT", "LPAREN", "RPAREN", "QUOTED_STRING", "WORD", "COLON", "ARROW")


# Token rules
def t_AND(t):
    r"AND"
    return t


def t_OR(t):
    r"OR"
    return t


def t_NOT(t):
    r"NOT"
    return t


def t_LPAREN(t):
    r"\("
    return t


def t_RPAREN(t):
    r"\)"
    return t


def t_COLON(t):
    r":"
    return t


def t_QUOTED_STRING(t):
    r'"[^"]*"'
    # Remove the quotes from the value
    t.value = t.value[1:-1]
    return t


# This should be updated when there are more characters possible in qlever
def t_WORD(t):
    r"[\w._0-9]+"
    return t


def t_ARROW(t):
    r"->"
    return t


# Ignored characters (spaces and tabs)
t_ignore = " \t"


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()


# AST Node classes
class ASTNode:
    pass


class BinaryOp(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f"BinaryOp({self.left}, {self.op}, {self.right})"

    def to_json(self):
        return {self.op: [self.left.to_json(), self.right.to_json()]}


class UnaryOp(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp({self.op}, {self.operand})"

    def to_json(self):
        return {self.op: [self.operand.to_json()]}


class Term(ASTNode):
    def __init__(self, value, fields=[]):
        self.value = value
        self.fields = fields

    def __repr__(self):
        if self.fields:
            f = "->".join(self.fields)
            return f"Term({f}:{self.value})"
        return f"Term({self.value})"

    def __str__(self):
        if self.fields:
            f = "->".join(self.fields) + ":"
        else:
            f = ""
        if " " in self.value:
            return f'{f}"{self.value}"'
        else:
            return f"{f}{self.value}"

    def to_json(self):
        if self.fields:
            result = {}
            top = result
            for f in self.fields[:-1]:
                result[f] = {}
                result = result[f]
            result[self.fields[-1]] = str(self.value)
            return top
        else:
            return {"text": str(self.value)}


class TermList(ASTNode):
    def __init__(self, terms):
        self.terms = terms

    def __repr__(self):
        return f"TermList({self.terms})"

    def __iter__(self):
        return iter(self.terms)

    def __getitem__(self, index):
        return self.terms[index]

    def __len__(self):
        return len(self.terms)

    def to_json(self):
        return {"AND": [term.to_json() for term in self.terms]}


# Grammar rules with precedence
precedence = (
    ("left", "OR"),
    ("left", "AND"),
    ("right", "NOT"),
)


def p_query(p):
    """query : expression"""
    p[0] = p[1]


def p_expression_binop(p):
    """expression : expression AND expression
    | expression OR expression"""
    p[0] = BinaryOp(p[1], p[2], p[3])


def p_expression_not(p):
    """expression : NOT expression"""
    p[0] = UnaryOp(p[1], p[2])


def p_expression_group(p):
    """expression : LPAREN expression RPAREN"""
    p[0] = p[2]


def p_expression_term_list(p):
    """expression : term_list"""
    p[0] = p[1]


def p_term_list_single(p):
    """term_list : term"""
    if isinstance(p[1], TermList):
        p[0] = p[1]
    else:
        p[0] = TermList([p[1]])


def p_term_list_multiple(p):
    """term_list : term_list term"""
    if isinstance(p[1], TermList):
        p[1].terms.append(p[2])
        p[0] = p[1]
    else:
        p[0] = TermList([p[1], p[2]])


def p_term_word(p):
    """term : WORD"""
    p[0] = Term(p[1])


def p_term_quoted(p):
    """term : QUOTED_STRING"""
    p[0] = Term(p[1])


def p_term_field_word(p):
    """term : WORD COLON WORD"""
    p[0] = Term(p[3], fields=[p[1]])


def p_term_field_quoted(p):
    """term : WORD COLON QUOTED_STRING"""
    p[0] = Term(p[3], fields=[p[1]])


def p_term_field_chain_word(p):
    """term : field_chain COLON WORD"""
    p[0] = Term(p[3], fields=p[1])


def p_term_field_chain_quoted(p):
    """term : field_chain COLON QUOTED_STRING"""
    p[0] = Term(p[3], fields=p[1])


def p_field_chain_single(p):
    """field_chain : WORD"""
    p[0] = [p[1]]


def p_field_chain_multiple(p):
    """field_chain : field_chain ARROW WORD"""
    p[1].append(p[3])
    p[0] = p[1]


def p_error(p):
    if p:
        print(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}")
    else:
        print("Syntax error at EOF")


# Build the parser
parser = yacc.yacc()


class BooleanQueryParser:
    """
    A parser for boolean queries using PLY (Python Lex-Yacc).

    This parser handles boolean expressions with AND, OR, NOT operators,
    parentheses for grouping, field-qualified terms, and term lists.

    Features:
        - Case-sensitive boolean operators (AND, OR, NOT)
        - Parentheses for expression grouping
        - Term lists: consecutive terms as separate list items
        - Quoted strings for exact multi-word terms
        - Single word terms
        - Field-qualified terms: field:value, field:"quoted value"
        - Proper operator precedence (NOT > AND > OR)

    Example:
        parser = BooleanQueryParser()
        ast = parser.parse('title:python author:gibson "machine learning" OR tags:scala')
    """

    def __init__(self):
        """Initialize the parser with lexer and parser instances."""
        self.lexer = lexer
        self.parser = parser

    def parse(self, query_string):
        """
        Parse a boolean query string and return an Abstract Syntax Tree (AST).

        Args:
            query_string (str): The boolean query to parse

        Returns:
            ASTNode or None: The root node of the AST, or None if parsing fails

        Example:
            ast = parser.parse('title:a author:b AND (content:c OR tags:d)')
            # Returns: BinaryOp with field-qualified terms
        """
        try:
            result = self.parser.parse(query_string, lexer=self.lexer)
            return result
        except Exception as e:
            print(f"Parsing error: {e}")
            return None

    def tokenize(self, query_string):
        """
        Tokenize a query string for debugging purposes.

        Args:
            query_string (str): The query string to tokenize

        Returns:
            list: List of (token_type, token_value) tuples

        Example:
            tokens = parser.tokenize('title:a author:"John Doe" AND content:b')
            # Returns: [('WORD', 'title'), ('COLON', ':'), ('WORD', 'a'), ...]
        """
        self.lexer.input(query_string)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append((tok.type, tok.value))
        return tokens


def print_ast(node, indent=0):
    """
    Pretty print the AST structure for debugging and visualization.

    Args:
        node (ASTNode): The AST node to print
        indent (int, optional): Current indentation level. Defaults to 0.

    Example:
        ast = parser.parse('title:a AND (author:b OR content:c)')
        print_ast(ast)
        # Output shows field-qualified terms in the structure
    """
    spaces = "  " * indent

    if isinstance(node, Term):
        if node.fields:
            field_chain = "->".join(node.fields)
            print(f"{spaces}Term: '{field_chain} : {node.value}'")
        else:
            print(f"{spaces}Term: '{node.value}'")

    elif isinstance(node, TermList):
        if len(node.terms) == 1:
            term = node.terms[0]
            if term.fields:
                field_chain = "->".join(term.fields)
                print(f"{spaces}Term: '{field_chain} : {term.value}'")
            else:
                print(f"{spaces}Term: '{term.value}'")
        else:
            term_strs = []
            for term in node.terms:
                if term.fields:
                    field_chain = "->".join(term.fields)
                    term_strs.append(f"{field_chain} : {term.value}")
                else:
                    term_strs.append(term.value)
            print(f"{spaces}TermList: {term_strs}")
            for term in node.terms:
                if term.fields:
                    field_chain = "->".join(term.fields)
                    print(f"{spaces}  Term: '{field_chain} : {term.value}'")
                else:
                    print(f"{spaces}  Term: '{term.value}'")

    elif isinstance(node, UnaryOp):
        print(f"{spaces}UnaryOp: {node.op}")
        print_ast(node.operand, indent + 1)

    elif isinstance(node, BinaryOp):
        print(f"{spaces}BinaryOp: {node.op}")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)


# Quick usage examples and basic testing
if __name__ == "__main__":
    parser = BooleanQueryParser()

    # Test queries with fields
    test_queries = [
        "title:fish author:gibson",
        'title:"science fiction" AND author:gibson',
        'title:python java OR author:"John Doe"',
        'content:a title:b "unfielded term" OR tags:c',
        "NOT title:fish AND (author:gibson OR tags:cyberpunk)",
        "title:a title:b AND content:c",  # Multiple terms for same field
        '(title:fish OR author:gibson) AND content:"neural network"',
        "classification->name:painting AND shows->depicts->encountered->classification:fossil",
        "author->name:Rob OR about->broader->broader:paintings",
        'creator->person->name:"John Doe" AND subject->classification->broader:art',
        'NOT creator->name:Smith AND (about->type:book OR format->medium:"digital")',
        'a->b:simple AND complex->chain->with->many->levels:"complex value"',
    ]

    for query in test_queries:
        print(f"\nParsing query: {query}")
        print("Tokens:", parser.tokenize(query))

        ast = parser.parse(query)
        if ast:
            print("AST:")
            print_ast(ast)
        print("-" * 50)

    # Test Term.__str__ method with field chains
    print("\nTesting Term.__str__ method with field chains:")
    test_terms = [
        Term("value", fields=["field"]),
        Term("Rob", fields=["author", "name"]),
        Term("paintings", fields=["about", "broader", "broader"]),
        Term("complex value", fields=["complex", "chain", "with", "many", "levels"]),
        Term("simple", fields=[]),
    ]

    for term in test_terms:
        print(f"Term: {term}")
