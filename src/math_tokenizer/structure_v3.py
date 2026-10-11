"""Single-pass lexing/offset construction; exact v2 structure semantics.

No global memoization: parser runtime is included for every encode, making cold
measurements meaningful and avoiding a per-formula result cache speed claim.
"""
from .ast import Parser,Lexeme,ParseError,TOKEN
from .structure_v2 import Structure
from .validation_v2 import ALLOWED_V2


def structure_fast(text):
    matches=list(TOKEN.finditer(text))
    if text.isascii():
        byteoffset=None
        tokens=[Lexeme(m.group(),m.start(),m.end()) for m in matches]
        lex=tuple((m.start(),m.end(),m.group()) for m in matches)
    else:
        byteoffset=[0];pos=0
        for c in text:pos+=len(c.encode());byteoffset.append(pos)
        tokens=[Lexeme(m.group(),m.start(),m.end()) for m in matches]
        lex=tuple((byteoffset[m.start()],byteoffset[m.end()],m.group()) for m in matches)
    try:
        if not text.strip():raise ParseError('empty formula')
        parser=Parser.__new__(Parser);parser.text=text;parser.tokens=tokens;parser.pos=0
        root=parser.expr(0)
        if parser.pos!=len(tokens):raise ParseError('unconsumed input')
        nodes=[(root,1)];spans=set();depth=1
        while nodes:
            node,d=nodes.pop();depth=max(depth,d)
            spans.add((byteoffset[node.start],byteoffset[node.end]) if byteoffset else (node.start,node.end))
            for child in node.children:nodes.append((child,d+1))
        return Structure('semantic_ast',tuple(sorted(spans)),lex,depth)
    except (ParseError,RecursionError) as exc:reason=str(exc)
    spans=[];stack=[];maximum=0
    for start,end,token in lex:
        if token.startswith('\\') and token[1:].isalpha() and token[1:] not in ALLOWED_V2:
            return Structure('byte_fallback',(),lex,0,'unknown_macro:'+token)
        if token in ('{','(','['):stack.append((token,start));maximum=max(maximum,len(stack))
        elif token in ('}',')',']'):
            expected={'}':'{',')':'(',']':'['}[token]
            if not stack or stack[-1][0]!=expected:
                if token in (')',']') and stack and stack[-1][0] in ('(','['):
                    opened,begin=stack.pop();spans.append((begin,end))
                else:return Structure('byte_fallback',(),lex,0,'unbalanced_delimiter')
            else:opened,begin=stack.pop();spans.append((begin,end))
        spans.append((start,end))
    if stack:return Structure('byte_fallback',(),lex,0,'unclosed_delimiter')
    if not lex:return Structure('byte_fallback',(),lex,0,'empty')
    spans.append((0,len(text.encode())))
    return Structure('latex_syntax_tree',tuple(sorted(set(spans))),lex,maximum+1,reason)
