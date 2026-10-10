"""Strict source-span semantic AST for an explicitly limited LaTeX grammar.

Unsupported TeX is rejected, never silently repaired. Positions refer to the
original Python string; conversion to UTF-8 offsets is explicit. The parser does
not evaluate or rewrite formulae. Whitespace remains in the original string.
"""
from dataclasses import dataclass
import re

class ParseError(ValueError):
    pass

@dataclass(frozen=True)
class Node:
    kind: str
    start: int
    end: int
    children: tuple = ()

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

@dataclass(frozen=True)
class Lexeme:
    text: str
    start: int
    end: int

SYMBOLS = set(('alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta '
             'iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau '
             'upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi '
             'Sigma Upsilon Phi Psi Omega partial nabla infty ell hbar emptyset '
             'varnothing forall exists').split())
FUNCTIONS = set('sin cos tan cot sec csc sinh cosh tanh exp log ln det gcd max min'.split())
DECORATORS = set('mathrm mathbb mathcal mathbf mathrm mathit mathsf boldsymbol hat bar vec tilde overline dot ddot'.split())
BINARY = {'=':1, '<':1, '>':1, r'\le':1, r'\leq':1, r'\ge':1, r'\geq':1,
          r'\ne':1, r'\neq':1, r'\approx':1, r'\sim':1, r'\equiv':1,
          r'\in':1, r'\subset':1, r'\subseteq':1, r'\to':1, r'\rightarrow':1,
          '+':2, '-':2, r'\pm':2, r'\cup':2, r'\cap':2,
          '*':3, '/':3, r'\cdot':3, r'\times':3, r'\otimes':3}
TOKEN = re.compile(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[^\s]')

class Parser:
    def __init__(self, text):
        self.text = text
        self.tokens = [Lexeme(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]
        self.pos = 0
    def peek(self):
        return self.tokens[self.pos].text if self.pos < len(self.tokens) else None
    def take(self, expected=None):
        if self.pos >= len(self.tokens):
            raise ParseError('unexpected end')
        t = self.tokens[self.pos]
        if expected is not None and t.text != expected:
            raise ParseError(f'expected {expected}, got {t.text}')
        self.pos += 1
        return t
    def argument(self):
        if self.peek() != '{':
            raise ParseError('command argument requires braces')
        return self.atom()
    def atom(self):
        t = self.take()
        s = t.text
        if s in ('+', '-'):
            a = self.atom()
            return Node('unary'+s, t.start, a.end, (a,))
        if s in ('{', '(', '['):
            close = {'{':'}','(':')','[':']'}[s]
            a = self.expr(0)
            end = self.take(close)
            return Node('group', t.start, end.end, (a,))
        if s in (r'\frac', r'\dfrac', r'\tfrac'):
            a, b = self.argument(), self.argument()
            return Node('fraction', t.start, b.end, (a,b))
        if s == r'\sqrt':
            # Optional nth roots are deliberately not accepted in v1.
            a = self.argument()
            return Node('sqrt', t.start, a.end, (a,))
        if s.startswith('\\'):
            name = s[1:]
            if name in FUNCTIONS:
                a = self.atom()
                return Node('function:'+name,t.start,a.end,(a,))
            if name in DECORATORS:
                a = self.argument()
                return Node('decoration:'+name,t.start,a.end,(a,))
            if name in SYMBOLS:
                return Node('symbol',t.start,t.end)
            raise ParseError('unsupported command: '+s)
        if re.fullmatch(r'[A-Za-z]|[0-9]+(?:\.[0-9]+)?|[α-ωΑ-Ω]',s):
            return Node('number' if s[0].isdigit() else 'symbol',t.start,t.end)
        raise ParseError('unsupported atom: '+s)
    def expr(self, minimum):
        a = self.atom()
        while self.peek() is not None and self.peek() not in ('}',')',']'):
            s = self.peek()
            if s in ('^','_'):
                if 5 < minimum:
                    break
                self.take()
                b = self.atom()
                a = Node('power' if s=='^' else 'subscript',a.start,b.end,(a,b))
            elif s in BINARY:
                precedence = BINARY[s]
                if precedence < minimum:
                    break
                self.take()
                b = self.expr(precedence+1)
                a = Node('binary:'+s,a.start,b.end,(a,b))
            else:
                if 3 < minimum:
                    break
                b = self.expr(4)
                a = Node('implicit_product',a.start,b.end,(a,b))
        return a

def parse(text):
    if not text.strip():
        raise ParseError('empty formula')
    p = Parser(text)
    root = p.expr(0)
    if p.pos != len(p.tokens):
        raise ParseError('unconsumed input')
    return root

def byte_spans(text, root=None):
    root = root or parse(text)
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1]+len(char.encode('utf-8')))
    return tuple(sorted(set((offsets[n.start],offsets[n.end]) for n in root.walk())))

def compatible(start, end, spans):
    """A token may contain/lie inside a subtree, but never cut across it."""
    return all(end <= a or start >= b or (start <= a and end >= b)
               or (start >= a and end <= b) for a,b in spans)

def shape(text):
    def key(n):
        return (n.kind, tuple(key(c) for c in n.children))
    return repr(key(parse(text)))
