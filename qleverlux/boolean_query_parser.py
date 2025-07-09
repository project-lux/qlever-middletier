"""
Boolean Query Parser using PLY (Python Lex-Yacc)

This module provides a parser for boolean queries with the following features:
- Boolean operators: AND, OR, NOT (case-sensitive, capitalized)
- Parentheses for grouping: ( )
- Term lists: consecutive terms treated as separate items in a list
- Quoted strings for multi-word terms: "hello world"
- Single word terms: hello, world
- Proper operator precedence: NOT > AND > OR

Examples:
    Basic usage:
        parser = BooleanQueryParser()
        ast = parser.parse('(a OR "b c") AND (d e f OR g)')
        result = evaluate_ast(ast, context)

    Supported query formats:
        - Simple terms: a, hello
        - Term lists: a b c, hello "world test" foo
        - Quoted terms: "hello world", "machine learning"
        - Boolean operations: a AND b, x OR y, NOT z
        - Grouping: (a OR b) AND c
        - Complex: (a b "c d" e OR f) AND NOT g

Author: Generated for luxql project
License: Same as parent project
"""

import ply.lex as lex
import ply.yacc as yacc

# Token definitions
tokens = (
    "AND",
    "OR",
    "NOT",
    "LPAREN",
    "RPAREN",
    "QUOTED_STRING",
    "WORD",
)


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


def t_QUOTED_STRING(t):
    r'"[^"]*"'
    # Remove the quotes from the value
    t.value = t.value[1:-1]
    return t


# This should be updated when there are more characters possible in qlever
def t_WORD(t):
    r"[\w._0-9]+"
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
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Term({self.value})"

    def __str__(self):
        if " " in self.value:
            return f'"{self.value}"'
        else:
            return self.value

    def to_json(self):
        return {"text": str(self)}


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
    parentheses for grouping, and term lists with individual terms.

    Features:
        - Case-sensitive boolean operators (AND, OR, NOT)
        - Parentheses for expression grouping
        - Term lists: consecutive terms as separate list items
        - Quoted strings for exact multi-word terms
        - Single word terms
        - Proper operator precedence (NOT > AND > OR)

    Example:
        parser = BooleanQueryParser()
        ast = parser.parse('python java "machine learning" OR scala')
        if ast:
            result = evaluate_ast(ast, {"python": True, "java": False, "machine learning": True, "scala": False})
            print(result)  # True
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
            ast = parser.parse('a b AND (c OR d)')
            # Returns: BinaryOp(TermList([Term('a'), Term('b')]), 'AND', BinaryOp(TermList([Term('c')]), 'OR', TermList([Term('d')])))
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
            tokens = parser.tokenize('a b "c d" AND e')
            # Returns: [('WORD', 'a'), ('WORD', 'b'), ('QUOTED_STRING', 'c d'), ('AND', 'AND'), ('WORD', 'e')]
        """
        self.lexer.input(query_string)
        tokens = []
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            tokens.append((tok.type, tok.value))
        return tokens


def evaluate_ast(node, context=None):
    """
    Evaluate the AST with a given context.

    Args:
        node (ASTNode): The AST node to evaluate
        context (dict, optional): Dictionary mapping term names to boolean values.
                                 Defaults to empty dict if not provided.

    Returns:
        bool: The boolean result of evaluating the expression

    Example:
        context = {"python": True, "java": False, "machine learning": True}
        ast = parser.parse('python java "machine learning"')
        result = evaluate_ast(ast, context)  # Returns True if any term in the list matches
    """
    if context is None:
        context = {}

    if isinstance(node, Term):
        # Return the boolean value from context, default to False if not found
        return context.get(node.value, False)

    elif isinstance(node, TermList):
        # For term lists, return True if ANY term in the list matches (OR logic)
        return any(context.get(term.value, False) for term in node.terms)

    elif isinstance(node, UnaryOp):
        if node.op == "NOT":
            return not evaluate_ast(node.operand, context)

    elif isinstance(node, BinaryOp):
        left_val = evaluate_ast(node.left, context)
        right_val = evaluate_ast(node.right, context)

        if node.op == "AND":
            return left_val and right_val
        elif node.op == "OR":
            return left_val or right_val

    return False


def print_ast(node, indent=0):
    """
    Pretty print the AST structure for debugging and visualization.

    Args:
        node (ASTNode): The AST node to print
        indent (int, optional): Current indentation level. Defaults to 0.

    Example:
        ast = parser.parse('a AND (b OR c)')
        print_ast(ast)
        # Output:
        # BinaryOp: AND
        #   TermList: ['a', 'b']
        #   BinaryOp: OR
        #     TermList: ['c']
        #     TermList: ['d']
    """
    spaces = "  " * indent

    if isinstance(node, Term):
        print(f"{spaces}Term: '{node.value}'")

    elif isinstance(node, TermList):
        if len(node.terms) == 1:
            print(f"{spaces}Term: '{node.terms[0].value}'")
        else:
            print(f"{spaces}TermList: {[term.value for term in node.terms]}")
            for term in node.terms:
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

    # Test queries
    test_queries = [
        'a b "c d" OR e',
        "a AND b c OR d",
        "NOT a b AND c",
        '"hello world" test OR foo',
        "(a b AND c) OR (d e AND f)",
        "NOT (a b OR c d) AND e",
    ]

    for query in test_queries:
        print(f"\nParsing query: {query}")
        print("Tokens:", parser.tokenize(query))

        ast = parser.parse(query)
        if ast:
            print("AST:")
            print_ast(ast)

            # Example evaluation with some context
            context = {
                "a": True,
                "b": False,
                "c d": True,
                "d": True,
                "e": False,
                "c": True,
                "hello world": True,
                "test": False,
                "foo": True,
                "f": False,
            }
            result = evaluate_ast(ast, context)
            print(f"Evaluation result: {result}")
        print("-" * 50)
