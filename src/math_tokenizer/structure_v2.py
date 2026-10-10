"""Versioned structural front-end: semantic AST plus explicit LaTeX syntax tree.

Semantic grammar from v1 is preserved. Unsupported semantic forms use a lossless
concrete syntax tree with groups and command leaves, NOT a fabricated semantic
AST. Unsupported/unsafe commands or malformed grouping fall back explicitly.
"""
from dataclasses import dataclass
from functools import lru_cache
import re
from .ast import parse as parse_v1,byte_spans as spans_v1,Node,ParseError,TOKEN
from .validation_v2 import ALLOWED_V2 as ALLOWED

@dataclass(frozen=True)
class Structure:
    kind: str
    spans: tuple
    lexemes: tuple
    depth: int
    reason: str = ''

def offsets(text):
    result=[0]
    for c in text:result.append(result[-1]+len(c.encode()))
    return result

def lexemes(text):
    off=offsets(text)
    return tuple((off[m.start()],off[m.end()],m.group()) for m in TOKEN.finditer(text))

@lru_cache(maxsize=30000)
def structure(text):
    lex=lexemes(text)
    try:
        root=parse_v1(text)
        def depth(n):return 1+max((depth(c) for c in n.children),default=0)
        return Structure('semantic_ast',spans_v1(text,root),lex,depth(root))
    except (ParseError,RecursionError) as exc:
        semantic_reason=str(exc)
    # Build all source-span leaves and balanced groups. Separate node kinds and
    # coverage counts keep syntactic recovery distinct from semantic parsing.
    spans=[];stack=[];maximum=0
    for start,end,token in lex:
        if token.startswith('\\') and token[1:].isalpha() and token[1:] not in ALLOWED:
            return Structure('byte_fallback',(),lex,0,'unknown_macro:'+token)
        if token in ('{','(','['):
            stack.append((token,start));maximum=max(maximum,len(stack))
        elif token in ('}',')',']'):
            expected={'}':'{',')':'(',']':'['}[token]
            if not stack or stack[-1][0]!=expected:
                # Parentheses can be half-open intervals [a,b), which are legal
                # LaTeX syntax, so only braces enforce exact matching here.
                if token in (')',']') and stack and stack[-1][0] in ('(','['):
                    opened,begin=stack.pop();spans.append((begin,end))
                else:return Structure('byte_fallback',(),lex,0,'unbalanced_delimiter')
            else:
                opened,begin=stack.pop();spans.append((begin,end))
        spans.append((start,end))
    if stack:return Structure('byte_fallback',(),lex,0,'unclosed_delimiter')
    if not lex:return Structure('byte_fallback',(),lex,0,'empty')
    spans.append((0,len(text.encode())))
    return Structure('latex_syntax_tree',tuple(sorted(set(spans))),lex,maximum+1,semantic_reason)


def complexity(text):
    """Two fixed classes; thresholds are independent of train/dev/test scores."""
    s=structure(text);lex=s.lexemes
    commands=[t for a,b,t in lex if t.startswith('\\') and t[1:].isalpha()]
    operators=sum(t in ('+','-','*','/','=','<','>','^','_') or t in (r'\sum',r'\prod',r'\int',r'\partial',r'\nabla',r'\frac',r'\sqrt') for a,b,t in lex)
    # syntax nesting, not recursive binary-AST depth (long sums skew the latter)
    depth=0;maxdepth=0
    for a,b,t in lex:
        if t in ('{','(','['):depth+=1;maxdepth=max(depth,maxdepth)
        elif t in ('}',')',']'):depth=max(0,depth-1)
    length=len(lex);symbol_types=len({t for a,b,t in lex if t not in ('{','}','(',')','[',']')})
    command_types=len(set(commands))
    score=.40*min(length/60,2)+.20*min(command_types/6,2)+.15*min(maxdepth/3,2)+.15*min(symbol_types/18,2)+.10*min(operators/12,2)
    return {'class':'complex' if score>=.65 else 'simple','score':round(score,6),
            'latex_lexemes':length,'utf8_bytes':len(text.encode()),'symbol_types':symbol_types,
            'command_types':command_types,'nesting_depth':maxdepth,'operator_count':operators,
            'rule':'complexity-v2-fixed: .40*L/60+.20*C/6+.15*D/3+.15*S/18+.10*O/12 (each ratio capped2); complex>=.65'}
