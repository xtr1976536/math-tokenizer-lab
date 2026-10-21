"""Reversible coefficient-free GRE/LaTeX interface, without generation masks.

The parser accepts an explicit mathematical subset, never executes TeX, never
repairs malformed input, and maps only expressions supported by the author's
numerical terms. It has no access to target equations or observed rewards.
"""
from __future__ import annotations
import re
from .author import DICTIONARY

class LatexError(ValueError): pass
class Unsupported(LatexError): pass


def key(node):
    kind=node[0]
    if kind in ('add','mul'):
        children=[]
        for n in node[1:]:
            children.extend(key(n)[1:] if n[0]==kind else [key(n)])
        return (kind,*sorted(children,key=repr))
    if kind=='d':return ('d',key(node[1]),''.join(sorted(node[2])))
    if kind in ('v','n'):return node
    return (kind,*(key(n) if isinstance(n,tuple) else n for n in node[1:]))


def join(kind,nodes):
    if len(nodes)==1:return nodes[0]
    return (kind,*nodes)


class GREParser:
    def __init__(self,text):
        self.t=re.findall(r'BiLaplace|Laplace|sqrt|sinh|sin|exp|[uxyzt]|[0-9]+|[-+*/^()]',text)
        if ''.join(self.t)!=text:raise Unsupported('Unsupported GRE atom: '+text)
        self.p=0
    def peek(self):return self.t[self.p] if self.p<len(self.t) else None
    def take(self,value=None):
        if self.peek() is None:raise LatexError('Unexpected end')
        t=self.peek();self.p+=1
        if value is not None and value!=t:raise LatexError('Expected '+value)
        return t
    def expression(self):
        parts=[self.product()]
        while self.peek()=='+':self.take();parts.append(self.product())
        return join('add',parts)
    def product(self):
        a=self.power()
        while self.peek() is not None and self.peek() not in (')','+'):
            op=self.take() if self.peek() in ('*','/') else '*'
            b=self.power();a=('div',a,b) if op=='/' else ('mul',a,b)
        return a
    def power(self):
        a=self.atom()
        if self.peek()=='^':
            self.take();sign=-1 if self.peek()=='-' and self.take() else 1
            value=self.take()
            if not value.isdigit():raise Unsupported('Noninteger power')
            a=('pow',a,sign*int(value))
        return a
    def atom(self):
        t=self.take()
        if t=='(':
            a=self.expression();self.take(')');suffix=''
            while self.peek() in ('x','y','z','t'):suffix+=self.take()
            return ('d',a,suffix) if suffix else a
        if t in ('sin','sinh','sqrt','exp','Laplace','BiLaplace'):
            if self.peek()=='(':
                self.take();negative=self.peek()=='-'
                if negative:self.take()
                a=self.expression();self.take(')')
                if negative:a=('neg',a)
            else:a=self.atom()
            return ('lap',a,2 if t=='BiLaplace' else 1) if t in ('Laplace','BiLaplace') else ('fn',t,a)
        if t=='u':
            suffix=''
            while self.peek() in ('x','y','z','t'):suffix+=self.take()
            return ('d',('v','u'),suffix) if suffix else ('v','u')
        if t in ('x','y','z','t'):return ('v',t)
        if t.isdigit():return ('n',int(t))
        raise Unsupported('Unknown GRE syntax: '+t)
    def parse(self):
        a=self.expression()
        if self.peek() is not None:raise LatexError('Trailing GRE input')
        return a


def render(node):
    kind=node[0]
    if kind=='v':return node[1]
    if kind=='n':return str(node[1])
    if kind=='add':return '('+'+'.join(render(x) for x in node[1:])+')'
    if kind=='mul':return '('+r'\cdot '.join(render(x) for x in node[1:])+')'
    if kind=='div':return r'\frac{'+render(node[1])+'}{'+render(node[2])+'}'
    if kind=='pow':return '{'+render(node[1])+'}^{'+str(node[2])+'}'
    if kind=='neg':return '-{'+render(node[1])+'}'
    if kind=='fn':return '\\'+node[1]+'{'+render(node[2])+'}'
    if kind=='lap':return r'\Delta'+('^{2}' if node[2]==2 else '')+'{'+render(node[1])+'}'
    if kind=='d':
        if node[1]==('v','u'):return 'u_{'+node[2]+'}'
        return r'\partial_{'+node[2]+'}{'+render(node[1])+'}'
    raise Unsupported(kind)


class LatexParser:
    TOKEN=re.compile(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+|[^\s]')
    def __init__(self,text):
        if len(text)>16384:raise LatexError('Input too long')
        self.t=[m.group() for m in self.TOKEN.finditer(text) if m.group() not in (r'\left',r'\right',r'\,',r'\!',r'\;')];self.p=0
    def peek(self):return self.t[self.p] if self.p<len(self.t) else None
    def take(self,expected=None):
        t=self.peek()
        if t is None:raise LatexError('Unexpected end')
        if expected is not None and expected!=t:raise LatexError('Expected '+expected)
        self.p+=1;return t
    def group(self):
        opening=self.take();closing={"{":"}","(":")","[":"]"}.get(opening)
        if closing is None:raise LatexError('A group is required')
        a=self.expression();self.take(closing);return a
    def exponent(self):
        grouped=self.peek()=='{'
        if grouped:self.take()
        sign=-1 if self.peek()=='-' and self.take() else 1
        t=self.take()
        if not t.isdigit():raise Unsupported('Only integer powers supported')
        if grouped:self.take('}')
        return sign*int(t)
    def derivative_variables(self):
        grouped=self.peek()=='{'
        if grouped:self.take()
        result=''
        while self.peek() in ('x','y','z','t'):
            v=self.take();repeat=1
            if self.peek()=='^':self.take();repeat=self.exponent()
            if not 1<=repeat<=8:raise Unsupported('Derivative order')
            result+=v*repeat
            if not grouped:break
        if grouped:self.take('}')
        if not result:raise LatexError('Derivative variables required')
        return result
    def atom(self):
        t=self.peek()
        if t in ('{','(','['):a=self.group()
        elif t in ('u','x','y','z','t'):a=('v',self.take())
        elif t and t.isdigit():a=('n',int(self.take()))
        elif t==r'\frac':
            self.take();a=('div',self.group(),self.group())
        elif t in (r'\sin',r'\sinh',r'\exp',r'\sqrt'):
            self.take();a=('fn',t[1:],self.group() if self.peek() in ('{','(','[') else self.atom())
        elif t==r'\partial':
            self.take();self.take('_');variables=self.derivative_variables();a=('d',self.group(),variables)
        elif t==r'\Delta':
            self.take();order=1
            if self.peek()=='^':self.take();order=self.exponent()
            if order not in (1,2):raise Unsupported('Laplacian order')
            a=('lap',self.group(),order)
        elif t=='-':self.take();a=('neg',self.atom())
        else:raise Unsupported('Unsupported LaTeX token: '+str(t))
        if self.peek()=='_':
            if a!=('v','u'):raise Unsupported('Unsupported subscript')
            self.take();a=('d',a,self.derivative_variables())
        if self.peek()=='^':self.take();a=('pow',a,self.exponent())
        return a
    def product(self):
        a=self.atom()
        while self.peek() is not None and self.peek() not in ('+','-','=',')','}',']'):
            explicit=self.peek() in ('*',r'\cdot',r'\times','/')
            op=self.take() if explicit else '*'
            b=self.atom();a=('div',a,b) if op=='/' else ('mul',a,b)
        return a
    def expression(self):
        a=self.product()
        while self.peek() in ('+','-'):
            op=self.take();b=self.product();a=('add',a,b if op=='+' else ('neg',b))
        return a
    def parse(self):
        try:
            a=self.expression()
            if self.peek()=='=':
                self.take();rhs=self.expression()
                if rhs!=('n',0):raise Unsupported('Only coefficient-free structures with zero RHS are supported')
            if self.peek() is not None:raise LatexError('Trailing input')
            return a
        except RecursionError as e:raise LatexError('Nesting limit') from e


ATOM_TREES={i:GREParser(word).parse() for i,word in enumerate(DICTIONARY['id2word']) if i>=6}
ATOM_KEYS={key(tree):i for i,tree in ATOM_TREES.items()}
if len(ATOM_KEYS)!=len(ATOM_TREES):raise RuntimeError('Ambiguous GRE semantic mapping')


def flatten(node,kind):
    if node[0]!=kind:return [node]
    return [z for x in node[1:] for z in flatten(x,kind)]


def structure_to_latex(sequence):
    """Preserve the author's sequence order; do not include GRE control IDs."""
    parts=[]
    for value in sequence:
        i=int(value)
        if i in (0,1,5):continue
        if i in (2,3,4):parts.append({2:'+',3:r'\cdot ',4:'/'}[i])
        else:parts.append('('+render(ATOM_TREES[i])+')')
    return ''.join(parts)+'=0'


def latex_to_structure(text):
    expression=LatexParser(text).parse();result=[5]
    if key(expression) in ATOM_KEYS:return [5,ATOM_KEYS[key(expression)],1]
    def factors(node,power=1):
        lookup=ATOM_KEYS.get(key(node))
        if lookup is not None:return [(lookup,power)]
        if node[0]=='mul':
            return [f for n in flatten(node,'mul') for f in factors(n,power)]
        if node[0]=='div':return factors(node[1],power)+factors(node[2],-power)
        raise Unsupported('Expression not implemented in author numerical evaluator')
    for term in flatten(expression,'add'):
        if len(result)>1:result.append(2)
        pieces=factors(term)
        if not pieces or pieces[0][1]!=1:raise Unsupported('Unsupported leading reciprocal')
        for i,(atom,power) in enumerate(pieces):
            if i:result.append(3 if power==1 else 4)
            result.append(atom)
    result.append(1)
    return result
