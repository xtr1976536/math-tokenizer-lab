"""Auditable notation normalization and component extraction, no term deletion.

A component of a system is a separate record, never a conversion of the system.
Dimension-free Laplacians remain unresolved unless nearby source states a
single explicit dimension. All removed conditions remain attached as metadata.
"""
import re
from .corpus_conversion import convert,ConversionError


def components(text):
    """Keep every component, including unconvertible and boundary equations."""
    text=re.sub(r'\\label\{[^{}]*\}|\\(?:nonumber|notag)\b','',text)
    # Environment syntax is layout, never unknown mathematical content.
    text=re.sub(r'\\begin\{array\}\{[^{}]*\}','',text)
    text=re.sub(r'\\(?:begin|end)\{(?:equation\*?|aligned\*?|align\*?|split|gathered|gather\*?|cases|array)\}','',text)
    text=re.sub(r'\\left\s*\\\{|\\right\s*\.|\\right\s*\\\}', '',text)
    pieces=re.split(r'\\\\(?:\[[^\]]*\])?|\\cr\b',text)
    return [(i,p.strip()) for i,p in enumerate(pieces) if p.strip()]


def spatial_dimension(row):
    text=' '.join([row.get('latex',''),row.get('title',''),row.get('context_before','')[-1000:],row.get('context_after','')[:700]])
    dimensions=[];evidence=[]
    for m in re.finditer(r'\\(?:mathbb|mathbf|Bbb)\s*\{?R\}?\s*\^\s*\{?([123])\}?(?![0-9])',text):
        dimensions.append(int(m[1]));evidence.append(m.group())
    title=row.get('title','').lower()
    for n,word in [(1,'one'),(2,'two'),(3,'three')]:
        if re.search(r'\b'+word+r'[- ]dimensional\b',title):dimensions.append(n);evidence.append('title: '+word+'-dimensional')
    # An explicit scalar Cartesian spatial domain is R, not R^n.
    for m in re.finditer(r'(?<![A-Za-z])x\s*\\in\s*\\mathbb\s*\{?R\}?(?!\s*\^)',row.get('latex','')):
        dimensions.append(1);evidence.append(m.group())
    return (dimensions[0],evidence) if dimensions and len(set(dimensions))==1 else (None,evidence)


def braced_group(text,start):
    if start>=len(text) or text[start] not in '({[':return None
    opening=text[start];closing={'(':')','{':'}','[':']'}[opening];depth=1;i=start+1
    while i<len(text) and depth:
        if text[i]==opening:depth+=1
        if text[i]==closing:depth-=1
        i+=1
    return (text[start+1:i-1],i) if depth==0 else None


def normalize(text,row):
    rules=[];conditions=[]
    text=text.replace('&','')
    text=re.sub(r'\\(?:displaystyle|textstyle)\b','',text)
    text=text.strip().rstrip('.,')
    # Only trailing syntactically recognizable domain/condition annotations.
    marker=re.search(r'(?:,\s*(?:\\quad|\\qquad)?\s*|\\quad\s*|\\qquad\s*)(?=(?:\(?[tx](?:\s*,\s*[tx])?\)?\s*(?:>|<|\\in)|\\text\{(?:in|on|for|with)\b))',text)
    if marker:
        conditions.append(text[marker.start():]);text=text[:marker.start()];rules.append('trailing domain/condition kept as metadata')
    text=text.strip().rstrip('.,')
    # A single scalar field carrying an approximation/sequence label is a
    # family member; this is allowed only when exactly one label is present.
    decorations=re.findall(r'(?<![A-Za-z\\])u_\s*(\{(?:\\(?:varepsilon|epsilon|delta)|[njk])\}|\\(?:varepsilon|epsilon|delta)|[njk])(?![A-Za-z])',text)
    if decorations and len(set(decorations))==1:
        pattern=r'(?<![A-Za-z\\])u_\s*'+re.escape(decorations[0])+r'(?![A-Za-z])'
        replaced,n=re.subn(pattern,'u',text)
        # Do not merge a separately occurring undecorated u into a family field.
        remainder=re.sub(pattern,'',text)
        if not re.search(r'(?<![A-Za-z\\])u(?![A-Za-z])',remainder):
            text=replaced;rules.append('single labeled scalar family member u_'+decorations[0]+' -> u')
    # Explicitly indexed Cartesian coordinates, e.g. x_1,x_2, preserve axes.
    for index,axis in [(1,'x'),(2,'y'),(3,'z')]:
        pattern=r'x_\s*(?:\{'+str(index)+r'\}|'+str(index)+r'(?![0-9]))'
        if re.search(pattern,text):
            text=re.sub(pattern,axis,text);rules.append('Cartesian coordinate x_'+str(index)+' -> '+axis)
    dimension,evidence=spatial_dimension(row)
    # Expand Delta on a single atomic field/group only, never assume an operand
    # beyond the normal differential-operator precedence.
    if dimension and r'\Delta' in text:
        pattern=re.compile(r'\\Delta(?![A-Za-z])\s*(?:\^\s*(?:\{([12])\}|([12])))?\s*')
        pos=0
        while (match:=pattern.search(text,pos)) is not None:
            start=match.end()
            group=braced_group(text,start)
            if group:operand,end=group
            else:
                atom=re.match(r'(?:\\(?:phi|varphi|psi|rho|theta|mu)|[A-Za-z])(?:_\{[xyzt]+\})?',text[start:])
                if not atom:break
                operand=atom.group();end=start+atom.end()
            power=int(match[1] or match[2] or 1)
            expression='{'+operand+'}'
            for _ in range(power):
                expression='('+ '+'.join(r'\partial_{'+a+a+'}{'+expression+'}' for a in 'xyz'[:dimension])+')'
            text=text[:match.start()]+expression+text[end:];pos=match.start()+len(expression)
            rules.append('Delta expanded in explicit dimension '+str(dimension))
    return text,rules,conditions,evidence


def review_row(row):
    output=[]
    pieces=components(row['latex'])
    for index,text in pieces:
        normalized,rules,conditions,evidence=normalize(text,row)
        item=dict(component_index=index,component_latex=text,conversion_input=normalized,
                  normalization_rules=rules,conditions=conditions,dimension_evidence=evidence,
                  component_of_system=len(pieces)>1)
        try:
            result=convert(normalized)
            context=' '.join([row.get('context_before','')[-1000:],row.get('context_after','')[:700]])
            if result['scalar_coefficient_symbols'] and re.search(r'(?:functions?[^.]{0,160}(?:analytic|smooth|depend)|(?:analytic|smooth)[^.]{0,160}functions?)',context,re.I):
                raise ConversionError('variable_coefficient_context','Source describes coefficient functions; scalar-constant conversion is not certified')
            # An initial/boundary value or algebraic identity is not a PDE.
            if 'Derivative(' not in result['expanded_source']:
                raise ConversionError('not_differential_equation','No differential term remains')
            item.update(result,status='converted')
        except ConversionError as exc:
            item.update(status='unresolved',reason=exc.code,detail=str(exc))
        except (ValueError,TypeError,KeyError,RecursionError,NotImplementedError) as exc:
            item.update(status='unresolved',reason='parser_or_symbolic_failure',detail=type(exc).__name__+': '+str(exc))
        output.append(item)
    return output
