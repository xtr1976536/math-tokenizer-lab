"""Conservative paper-LaTeX to the unchanged 57-ID EqGPT dictionary.

No eval, TeX execution, vocabulary extension or unsupported-term deletion.
Unknown scalar coefficients are assumed constant and disclosed. Systems,
ambiguous dimensions and unimplemented operators are retained as rejections.
"""
from collections import Counter
from functools import lru_cache
import json
import re
import sympy as sp
from math_tokenizer.corpus import ROOT

DICTIONARY=json.loads((ROOT/'eqgpt_original/code/dict_datas_0725.json').read_text())
COORDS=sp.symbols('x y z t')
X,Y,Z,T=COORDS
U=sp.Function('u')(*COORDS)

class ConversionError(ValueError):
    def __init__(self,code,detail):
        self.code=code
        super().__init__(detail)


def reject(code,detail):
    raise ConversionError(code,detail)


class PaperParser:
    token=re.compile(r'\\[A-Za-z]+|\\[^A-Za-z]|(?:\d+\.\d+|\d+)|[^\s]')
    spacing={r'\left',r'\right',r'\,',r'\!',r'\;',r'\:',r'\ ',r'\big',r'\Big',r'\bigl',r'\bigr',r'\Bigl',r'\Bigr'}
    greek={r'\alpha',r'\beta',r'\gamma',r'\delta',r'\epsilon',r'\varepsilon',r'\lambda',r'\mu',r'\nu',r'\kappa',r'\rho',r'\sigma',r'\tau',r'\theta',r'\eta',r'\omega',r'\pi',r'\chi',r'\zeta',r'\varphi',r'\phi',r'\psi'}
    def __init__(self,text,fields=None):
        self.tokens=[m.group() for m in self.token.finditer(text) if m.group() not in self.spacing]
        self.p=0
        self.fields=set(fields or ['u'])
        self.renamings={name:'u' for name in self.fields if name!='u'}
        self.rules=[]
        self.field_arguments=[]
    def peek(self):
        return self.tokens[self.p] if self.p<len(self.tokens) else None
    def take(self,expected=None):
        v=self.peek()
        if v is None or expected is not None and v!=expected:
            reject('syntax','Expected '+str(expected)+'; got '+str(v))
        self.p+=1
        return v
    def raw_group(self):
        opening=self.take()
        if opening not in ('{','(','['):
            return [opening]
        close={'{':'}','(':')','[':']'}[opening];out=[];depth=1
        while depth:
            v=self.take()
            if v==opening:depth+=1
            if v==close:depth-=1
            if depth:out.append(v)
        return out
    def subparse(self,tokens):
        parser=PaperParser(' '.join(tokens),self.fields)
        node=parser.expression()
        if parser.peek() is not None:reject('syntax','Trailing grouped expression: '+str(parser.peek()))
        self.rules.extend(parser.rules)
        self.field_arguments.extend(parser.field_arguments)
        return node
    def exponent(self):
        tokens=self.raw_group()
        if not re.fullmatch(r'-?\d+',''.join(tokens)):
            reject('unsupported_power','Only integer powers are supported: '+''.join(tokens))
        n=int(''.join(tokens))
        if abs(n)>32:reject('unsupported_power','Power exceeds conversion bound')
        return n
    def derivative_suffix(self):
        raw=''.join(self.raw_group())
        raw=raw.replace('{','').replace('}','')
        parts=re.findall(r'([xyzt])(?:\^(\d+))?',raw)
        if ''.join(a+('^'+b if b else '') for a,b in parts)!=raw:
            reject('unsupported_derivative','Derivative coordinates must be x/y/z/t: '+raw)
        axes=''.join(a*int(b or 1) for a,b in parts)
        if not axes or len(axes)>8:reject('unsupported_derivative','Derivative order outside 1..8')
        return axes
    def differentiated(self,node,axes):
        unknown=node.free_symbols-set(COORDS)
        if unknown:
            reject('coefficient_inside_derivative_unresolved','Cannot assume differentiated symbols constant: '+','.join(sorted(map(str,unknown))))
        if not node.has(U):
            reject('unresolved_differentiated_field','Cannot treat an unknown differentiated field as a constant: '+str(node))
        self.rules.append('explicit partial derivative -> symbolic derivative')
        return sp.Derivative(node,*[dict(zip('xyzt',COORDS))[a] for a in axes],evaluate=False)
    def tex_argument(self):
        token=self.peek()
        if token and token.isdigit() and len(token)>1:
            self.tokens[self.p:self.p+1]=list(token)
        return self.raw_group()
    def fraction(self):
        numerator=self.tex_argument();denominator=self.tex_argument()
        if numerator and numerator[0]==r'\partial':
            n=1;rest=numerator[1:]
            if rest and rest[0]=='^':
                p=PaperParser(' '.join(rest[1:]),self.fields);n=p.exponent();rest=p.tokens[p.p:]
            d=PaperParser(' '.join(denominator),self.fields);axes=''
            while d.peek() is not None:
                d.take(r'\partial');axis=d.take()
                if axis not in 'xyzt' or len(axis)!=1:reject('unsupported_derivative','Non-Cartesian differential denominator')
                repeat=1
                if d.peek()=='^':d.take();repeat=d.exponent()
                if repeat<1 or repeat>8:reject('unsupported_derivative','Invalid derivative order')
                axes+=axis*repeat
            if len(axes)!=n:reject('ambiguous_derivative','Numerator/denominator derivative orders differ')
            node=self.subparse(rest) if rest else self.atom()
            return self.differentiated(node,axes)
        return self.subparse(numerator)/self.subparse(denominator)
    def atom(self):
        token=self.peek()
        if token in ('+','-'):
            self.take();return self.atom()*(1 if token=='+' else -1)
        if token in ('{','(','['):
            node=self.subparse(self.raw_group())
        elif token and re.fullmatch(r'\d+(?:\.\d+)?',token):
            node=sp.Rational(self.take())
        elif token in (r'\frac',r'\dfrac',r'\tfrac'):
            self.take();node=self.fraction()
        elif token==r'\partial':
            self.take();order=None;axes=None
            if self.peek()=='^':self.take();order=self.exponent()
            if self.peek()=='_':self.take();axes=self.derivative_suffix()
            if axes is None:reject('ambiguous_derivative','Partial without coordinate')
            if self.peek()=='^':self.take();order=self.exponent()
            if order is not None:
                if len(axes)!=1:reject('ambiguous_derivative','Both mixed suffix and power')
                axes*=order
            if not 1<=len(axes)<=8:reject('unsupported_derivative','Derivative order outside 1..8')
            node=self.differentiated(self.atom(),axes)
        elif token in (r'\sin',r'\sinh',r'\exp',r'\sqrt',r'\cos',r'\log',r'\cosh',r'\tanh'):
            self.take()
            fn={r'\sin':sp.sin,r'\sinh':sp.sinh,r'\exp':sp.exp,r'\sqrt':sp.sqrt,r'\cos':sp.cos,r'\log':sp.log,r'\cosh':sp.cosh,r'\tanh':sp.tanh}[token]
            node=fn(self.atom())
        elif token in (r'\Delta',r'\nabla',r'\Box',r'\div',r'\operatorname'):
            reject('operator_or_dimension_unresolved','Operator requires explicit dimension/semantics: '+str(token))
        elif token and (re.fullmatch('[A-Za-z]',token) or token in self.greek):
            name=self.take()
            node=U if name in self.fields else dict(zip('xyzt',COORDS)).get(name,sp.Symbol(name.lstrip('\\')))
            if self.peek()=='_':
                self.take()
                if name in self.fields:
                    node=self.differentiated(node,self.derivative_suffix())
                else:
                    suffix=''.join(self.raw_group())
                    if not re.fullmatch('[0-9]+',suffix):reject('unsupported_index','Indexed coefficient: '+suffix)
                    node=sp.Symbol(name.lstrip('\\')+'_'+suffix)
            if self.peek()=='(':
                saved=self.p;arg_tokens=self.raw_group();self.p=saved
                args=''.join(arg_tokens)
                if name in self.fields and re.fullmatch('[xyzt](?:,[xyzt])*',args):
                    self.raw_group()
                    self.field_arguments.append(set(args.replace(',','')))
                    self.rules.append('explicit dependent-variable coordinate arguments suppressed')
                elif name in self.fields and (any(t in ('x','y','z','t',',') for t in arg_tokens) or re.fullmatch('[0-9]+',args)):
                    reject('unsupported_field_arguments','Shifted/restricted field arguments: '+args)
                elif name not in self.fields and (name in ('f','F','g','G','h','H','P','Q','R','A','B','D','W') or any(t in self.fields or t in ('x','y','z','t',',') for t in arg_tokens)):
                    reject('unknown_function','Unknown functional dependence: '+name)
                # Otherwise this is ordinary juxtaposed multiplication.
        else:
            reject('unsupported_token','Unsupported token '+str(token))
        if self.peek()=='_' and token in ('{','(','['):
            self.take();node=self.differentiated(node,self.derivative_suffix())
        if self.peek()=='^':self.take();node=node**self.exponent()
        return node
    def product(self):
        node=self.atom()
        while self.peek() is not None and self.peek() not in ('+','-','=',')','}',']',','):
            op=self.take() if self.peek() in ('*','/',r'\cdot',r'\times') else '*'
            right=self.atom();node=node/right if op=='/' else node*right
        return node
    def expression(self):
        node=self.product()
        while self.peek() in ('+','-'):
            op=self.take();other=self.product();node=node+other if op=='+' else node-other
        return node
    def equation(self):
        lhs=self.expression();self.take('=');rhs=self.expression()
        if self.peek() is not None:reject('trailing_content','Remaining token '+str(self.peek()))
        return lhs-rhs


def prepare(text):
    rules=[];domain=None
    # Preserve a standalone equation, never silently select one row of a system.
    text=re.sub(r'\\(?:begin|end)\{(?:equation\*?|aligned|split|gathered)\}','',text)
    if re.search(r'\\(?:begin|end)\{|\\\\',text):
        reject('system_or_multiline','System/multiline requires explicit component review')
    # A trailing domain annotation is metadata, not an equation term.
    m=re.search(r'(?:\\quad|\\qquad|,\s*)\s*(?:\\text\{\s*(?:in|on|for)\b|\\mathrm\{in\}|\\textrm\{in\})',text)
    if m:
        domain=text[m.start():];text=text[:m.start()];rules.append('trailing domain annotation retained separately')
    text=text.replace('&','').strip().rstrip('.,')
    text=re.sub(r'\\(?:quad|qquad)', ' ',text)
    # A single known scalar field may be renamed; systems/multiple fields rejected.
    field_token=r'(\\[A-Za-z]+|[A-Za-z])'
    suffix=r'(?:\{[xyzt^0-9]+\}|[xyzt])'
    power=r'(?:\s*\^\s*(?:\{\d+\}|\d+))?'
    fields=set(re.findall(field_token+r'\s*_\s*'+suffix,text))
    fields.update(re.findall(r'\\partial'+power+r'\s*_\s*'+suffix+power+r'\s*'+field_token,text))
    fields.update(re.findall(r'\\partial'+power+r'\s+'+field_token,text))
    fields={name for name in fields if name not in ('x','y','z','t') and name != r'\partial'}
    # TeX juxtaposition uv is two symbols, not a scalar coefficient named uv.
    tokens=[m.group() for m in PaperParser.token.finditer(text)]
    fields.update(token for token in tokens if token in ('u','v','w'))
    if len(fields)>1:reject('multiple_fields','More than one dependent field: '+','.join(sorted(fields)))
    if not fields:reject('field_unresolved','No unambiguous supported scalar field with derivative evidence')
    return text,fields,rules,domain


def tree_to_sympy(node):
    kind=node[0]
    if kind=='v':return U if node[1]=='u' else dict(zip('xyzt',COORDS))[node[1]]
    if kind=='n':return sp.Integer(node[1])
    if kind=='add':return sp.Add(*(tree_to_sympy(x) for x in node[1:]))
    if kind=='mul':return sp.Mul(*(tree_to_sympy(x) for x in node[1:]))
    if kind=='div':return tree_to_sympy(node[1])/tree_to_sympy(node[2])
    if kind=='pow':return tree_to_sympy(node[1])**node[2]
    if kind=='neg':return -tree_to_sympy(node[1])
    if kind=='d':return sp.diff(tree_to_sympy(node[1]),*[dict(zip('xyzt',COORDS))[a] for a in node[2]])
    if kind=='fn':return {'sin':sp.sin,'sinh':sp.sinh,'sqrt':sp.sqrt,'exp':sp.exp}[node[1]](tree_to_sympy(node[2]))
    if kind=='lap':
        a=tree_to_sympy(node[1])
        for _ in range(node[2]):a=sp.diff(a,X,2)+sp.diff(a,Y,2)
        return a
    raise ValueError(kind)


@lru_cache(None)
def atoms():
    # Importing the established parser also checks every official atom mapping.
    from .latex_codec import ATOM_TREES
    return {i:tree_to_sympy(tree) for i,tree in ATOM_TREES.items()}


def factor_ids(expr):
    lookup={v:k for k,v in atoms().items()}
    if expr in lookup:return [lookup[expr]]
    numerator=[];denominator=[]
    for factor in sp.Mul.make_args(expr):
        if factor in lookup:numerator.append(lookup[factor]);continue
        base,power=factor.as_base_exp()
        if base in lookup and power.is_Integer and abs(power)<=32:
            (numerator if power>0 else denominator).extend([lookup[base]]*abs(int(power)))
        else:reject('outside_57_vocabulary','Unrepresented differential factor: '+str(factor))
    if not numerator:reject('outside_57_vocabulary','Leading reciprocal/constant term is not expressible')
    result=[]
    for i,value in enumerate(sorted(numerator)):
        if i:result.append(3)
        result.append(value)
    for value in sorted(denominator):result.extend([4,value])
    return result


def gre_expression(ids):
    expression=sp.Integer(0);term=None;op=3
    for i in ids[1:-1]+[2]:
        if i==2:
            if term is None:reject('invalid_gre','Empty term')
            expression+=term;term=None;op=3
        elif i in (3,4):op=i
        else:
            atom=atoms()[i]
            term=atom if term is None else term*atom if op==3 else term/atom
    return sp.expand(expression)


def support_key(expr):
    """Exact algebraic/product-rule expansion, then discard only scalar coefficients."""
    expanded=sp.expand(expr.doit())
    terms=[]
    for term in sp.Add.make_args(expanded):
        coefficient,body=term.as_independent(*COORDS,U,*expanded.atoms(sp.Derivative),as_Add=False)
        # as_independent handles symbolic constants; a body free of u is still a term.
        if coefficient==0:continue
        terms.append(sp.srepr(body))
    return tuple(sorted(set(terms)))


def convert(text):
    cleaned,fields,rules,domain=prepare(text)
    parser=PaperParser(cleaned,fields)
    expression=sp.expand(parser.equation().doit())
    used_axes={str(v) for derivative in expression.atoms(sp.Derivative) for v in derivative.variables}
    if parser.field_arguments and (any(not used_axes<=a for a in parser.field_arguments) or any(a!=parser.field_arguments[0] for a in parser.field_arguments)):
        reject('inconsistent_field_arguments','Declared field coordinates do not support all derivatives consistently')
    if expression==0:reject('identity','Equation simplifies to an identity')
    if not expression.has(U):reject('field_unresolved','No dependent field remains')
    if expression.count_ops()>1500:reject('expression_limit','Expanded expression too large')
    constant_symbols=expression.free_symbols-set(COORDS)
    if any(str(s) in ('i','imath') for s in constant_symbols):
        reject('complex_coefficient','Imaginary coefficient is outside the real scalar EqGPT fit protocol')
    sequence=[5];coefficients=[];bodies=[]
    derivatives=expression.atoms(sp.Derivative)
    grouped={}
    for term in sp.Add.make_args(expression):
        coefficient,body=term.as_independent(*COORDS,U,*derivatives,as_Add=False)
        grouped[body]=grouped.get(body,sp.Integer(0))+coefficient
    for body,coefficient in sorted(grouped.items(),key=lambda item:sp.srepr(item[0])):
        coefficient=sp.simplify(coefficient)
        if coefficient==0:continue
        if body==1:reject('constant_forcing','Constant source term has no unit atom in fixed GRE vocabulary')
        ids=factor_ids(body)
        if len(sequence)>1:sequence.append(2)
        sequence.extend(ids);coefficients.append(str(coefficient));bodies.append(body)
    sequence.append(1)
    if not all(0<=i<57 for i in sequence):reject('invalid_gre','ID outside dictionary')
    expected=sp.expand(sum(bodies));actual=gre_expression(sequence)
    if sp.expand(expected-actual)!=0:reject('roundtrip_mismatch','GRE reconstruction differs from coefficient-free source')
    from .latex_codec import structure_to_latex,latex_to_structure
    rendered=structure_to_latex(sequence)
    if latex_to_structure(rendered)!=sequence:reject('roundtrip_mismatch','Existing LaTeX/GRE codec failed roundtrip')
    return dict(gre_ids=sequence,gre_words=[DICTIONARY['id2word'][i] for i in sequence],
                normalized_latex=rendered,field_mapping={name:'u' for name in fields},
                scalar_coefficient_symbols=sorted(map(str,constant_symbols)),coefficients=coefficients,
                coefficient_assumption='Unspecified scalar symbols treated as spatially/temporally constant; zero/degenerate parameter cases not claimed',
                transformations=rules+parser.rules+['move RHS to LHS','exact product-rule/algebra expansion','strip only multiplicative scalar coefficients per term'],
                domain_annotation=domain,expanded_source=str(expression),support=list(support_key(expression)),
                reconstructed_support=list(support_key(actual)),roundtrip_verified=True)
