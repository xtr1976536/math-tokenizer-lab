"""Conservative source-fidelity and formula quality checks, no TeX execution."""
from dataclasses import dataclass
import gzip
import hashlib
import io
import re
import tarfile
import unicodedata

from .ast import TOKEN

MATH_ENVS = {'equation', 'equation*', 'align', 'align*', 'gather', 'gather*',
             'multline', 'multline*', 'eqnarray', 'eqnarray*', 'displaymath', 'math'}
LAYOUT_MAP = {'align': 'aligned', 'gather': 'gathered', 'multline': 'gathered', 'eqnarray': 'aligned'}
ADVANCED = re.compile(r'\\(?:frac|dfrac|tfrac|int|iint|iiint|oint|sum|prod|lim|partial|nabla|Delta|det|sqrt|forall|exists|binom|mathbb|mathcal|mathfrak|operatorname)(?![A-Za-z])|\\begin\{(?:cases|aligned|gathered|[pbBvV]?matrix)\}')
FORBIDDEN = re.compile(r'\\(?:input|include|write|openout|read|usepackage|documentclass|catcode|def|gdef|newcommand|renewcommand|providecommand|special|href|url|label|tag|ref|eqref|cite)(?![A-Za-z])')
MACRO = re.compile(r'\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?\s*|\\def\s*')
TARGET_CATEGORIES = {'math.AG','math.DG','math.GT','math.AT','math.MG','math.SG','math.KT','math.CT','math.GR','math.RA','math.RT','math.AC','math.OA','math.QA','math.NT','math.CO','math.LO'}


def library_topics(path):
    name=path.rsplit('/',1)[-1].removesuffix('.py')
    number={'gcd','fibonacci','totient','prime_numbers','dirichlet','riemann_zeta','hurwitz_zeta','multiple_zeta_values','eisenstein','modular_lambda','modular_j','modular_transformations','dedekind_eta'}
    combinatorics={'factorials','bell_numbers','partitions','stirling_numbers','bernoulli_numbers'}
    primary='number_theory' if name in number else 'combinatorics' if name in combinatorics else 'analysis'
    tags={primary}
    if name in {'riemann_zeta','hurwitz_zeta','multiple_zeta_values','eisenstein','modular_lambda','modular_j','modular_transformations','dedekind_eta'}:tags.add('analysis')
    if name in {'fibonacci','bernoulli_numbers'}:tags.update(('number_theory','combinatorics'))
    if name in {'modular_lambda','modular_j','modular_transformations','weierstrass_elliptic'}:tags.add('geometry_topology')
    return primary,sorted(tags)


def escaped(text, position):
    count = 0; position -= 1
    while position >= 0 and text[position] == '\\':count += 1; position -= 1
    return bool(count % 2)


def group(text, position, opening='{', closing='}'):
    while position < len(text) and text[position].isspace():position += 1
    if position >= len(text) or text[position] != opening:return None
    depth = 1; start = position + 1; position += 1
    while position < len(text):
        if not escaped(text, position):
            if text[position] == opening:depth += 1
            elif text[position] == closing:
                depth -= 1
                if depth == 0:return text[start:position], position + 1
        position += 1
    return None


def source_members(data):
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode='r:*')
    except tarfile.ReadError:
        raw = gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data
        entries = [('main.tex', raw)]
    else:
        with archive:
            entries = [(m.name, archive.extractfile(m).read()) for m in sorted(archive.getmembers(), key=lambda m:m.name)
                       if m.isfile() and m.name.endswith(('.tex','.ltx','.sty','.cls')) and m.size <= 4_000_000]
    for name, raw in entries:
        try:text = raw.decode('utf-8'); encoding = 'utf-8'
        except UnicodeDecodeError:
            declaration = re.search(rb'\\usepackage\[([^\]]+)\]\{inputenc\}', raw)
            label = declaration[1].decode('ascii', errors='ignore').lower() if declaration else ''
            encoding = {'latin1':'latin-1','latin9':'iso-8859-15','ansinew':'cp1252','cp1252':'cp1252'}.get(label)
            if encoding is None:yield name, None, 'unverified_non_utf8_encoding', raw;continue
            text = raw.decode(encoding)
        yield name, text, encoding, raw


def strip_comments(text):
    chars = list(text)
    for match in re.finditer(r'%[^\n]*', text):
        if not escaped(text, match.start()):chars[match.start():match.end()] = ' ' * (match.end()-match.start())
    return ''.join(chars)


def macro_definitions(text):
    text = strip_comments(text); definitions = []; ranges = []
    for match in MACRO.finditer(text):
        position = match.end(); parsed = group(text, position)
        if parsed is not None and re.fullmatch(r'\\[A-Za-z]+', parsed[0]):name, position = parsed
        else:
            command = re.match(r'\s*(\\[A-Za-z]+)', text[position:])
            if not command:continue
            name = command[1]; position += command.end()
        optional = group(text, position, '[', ']'); nargs = 0
        if optional is not None:
            if not optional[0].isdigit():continue
            nargs = int(optional[0]); position = optional[1]
            if group(text, position, '[', ']') is not None:continue  # defaults need scoped TeX semantics
        body = group(text, position)
        if body is None or nargs > 3:continue
        replacement, end = body; ranges.append((match.start(), end))
        if 'DeclareMathOperator' in match[0]:replacement = r'\operatorname{' + replacement + '}'
        if len(replacement) <= 512 and not FORBIDDEN.search(replacement):
            parameters = {int(i) for i in re.findall(r'#([1-9])', replacement)}
            if parameters <= set(range(1,nargs+1)):definitions.append((name, nargs, replacement))
    return definitions, ranges


def build_macros(members):
    by_name = {}
    for name, text, encoding, raw in members:
        if text is None:continue
        for macro, nargs, body in macro_definitions(text)[0]:by_name.setdefault(macro,set()).add((nargs,body))
    ambiguous = {name for name, values in by_name.items() if len(values) != 1}
    return {name:next(iter(values)) for name,values in by_name.items() if name not in ambiguous}, ambiguous


def expand_macros(formula, macros, ambiguous):
    used = []
    for iteration in range(12):
        parts = []; position = 0; changed = False
        for match in re.finditer(r'\\[A-Za-z]+', formula):
            if match.start() < position:continue
            name = match[0]
            if name in ambiguous:return None, used, 'ambiguous_macro:' + name
            if name not in macros:continue
            nargs, body = macros[name]; end = match.end(); arguments = []
            for _ in range(nargs):
                parsed = group(formula,end)
                if parsed is None:return None, used, 'unbraced_macro_argument:' + name
                argument,end = parsed; arguments.append(argument)
            replacement = body
            # TeX strips the argument-delimiting braces. Adding a new group
            # here would change #1^2 applied to {x+y} from x+y^2 to (x+y)^2.
            for i,argument in enumerate(arguments,1):replacement = replacement.replace('#'+str(i),argument)
            parts.extend((formula[position:match.start()], replacement));position=end;changed=True;used.append(name)
        if not changed:return formula, sorted(set(used)), None
        parts.append(formula[position:]);new = ''.join(parts)
        if new == formula:return None, used, 'recursive_macro'
        formula = new
        if len(formula.encode()) > 16384:return None, used, 'macro_expansion_limit'
    return None, used, 'macro_recursion_limit'


def candidates(text):
    """Keep character coordinates in the decoded source; ignore preamble/code."""
    masked = list(strip_comments(text))
    document = re.search(r'\\begin\{document\}', text)
    if document:masked[:document.end()] = ' ' * document.end()
    for start,end in macro_definitions(text)[1]:masked[start:end] = ' ' * (end-start)
    for match in re.finditer(r'\\begin\{(verbatim\*?|lstlisting|minted|thebibliography)\}[\s\S]*?\\end\{\1\}', text):
        masked[match.start():match.end()] = ' ' * (match.end()-match.start())
    work = ''.join(masked); covered = []
    environment = re.compile(r'\\begin\{([^{}]+)\}')
    for match in environment.finditer(work):
        name = match[1]
        if name not in MATH_ENVS or any(a<=match.start()<b for a,b in covered):continue
        ending = re.search(r'\\end\{'+re.escape(name)+r'\}', work[match.end():])
        if ending is None:continue
        start,end = match.end(),match.end()+ending.start();outer_end=match.end()+ending.end();covered.append((match.start(),outer_end))
        body = work[start:end]; wrapper = LAYOUT_MAP.get(name.rstrip('*'))
        yield {'raw_latex':text[start:end], 'working_latex':body, 'source_span':[start,end],
               'line':1+text.count('\n',0,start), 'delimiter':name, 'layout_wrapper':wrapper}
    for opening,closing in [(r'\[',r'\]'),(r'\(',r'\)')]:
        pattern = re.compile(re.escape(opening)+r'([\s\S]*?)'+re.escape(closing))
        for match in pattern.finditer(work):
            if any(a<=match.start()<b for a,b in covered):continue
            start,end=match.start(1),match.end(1);covered.append((match.start(),match.end()))
            yield {'raw_latex':text[start:end],'working_latex':work[start:end], 'source_span':[start,end],
                   'line':1+text.count('\n',0,start),'delimiter':opening,'layout_wrapper':None}
    position = 0
    while position < len(work):
        if work[position]!='$' or escaped(work,position) or any(a<=position<b for a,b in covered):position+=1;continue
        width=2 if work.startswith('$$',position) else 1;start=position+width;end=start
        while end<len(work):
            if work.startswith('$'*width,end) and not escaped(work,end):break
            end+=1
        if end>=len(work):break
        yield {'raw_latex':text[start:end],'working_latex':work[start:end], 'source_span':[start,end],
               'line':1+text.count('\n',0,start),'delimiter':'$'*width,'layout_wrapper':None}
        position=end+width


def normalize(candidate, macros, ambiguous):
    formula = candidate['working_latex']; operations = []
    # Tags and labels are layout metadata; balanced groups prevent truncation.
    edits=[]
    for match in re.finditer(r'\\(?:label|tag)\*?\s*',formula):
        parsed=group(formula,match.end())
        if parsed is not None:edits.append((match.start(),parsed[1]))
    for a,b in reversed(edits):formula=formula[:a]+formula[b:]
    if edits:operations.append('remove_equation_metadata')
    formula=re.sub(r'\\(?:nonumber|notag)(?![A-Za-z])','',formula).strip()
    # Preserve escaped spacing commands (old rstrip could damage a final \,).
    while formula and formula[-1] in '.,;' and not escaped(formula,len(formula)-1):formula=formula[:-1].rstrip()
    expanded, used, reason = expand_macros(formula,macros,ambiguous)
    if reason:return None,operations,reason
    formula=expanded
    if used:operations.append({'source_macro_expansion':used})
    if candidate['layout_wrapper']:
        wrapper=candidate['layout_wrapper'];formula=r'\begin{'+wrapper+'}'+formula+r'\end{'+wrapper+'}';operations.append({'layout_environment':candidate['delimiter']+'->'+wrapper})
    return formula.strip(),operations,None


def source_macro_reference(candidate,macros,ambiguous,used):
    r"""Unexpanded input plus bounded \def program for independent rendering."""
    original=dict(candidate)
    original['working_latex']=candidate['working_latex']
    # Strip only metadata/layout using the same normalization before expansion.
    original_text,operations,reason=normalize(original,{},set())
    definitions=''.join(r'\def'+name+''.join('#'+str(i) for i in range(1,macros[name][0]+1))+'{'+macros[name][1]+'}' for name in sorted(set(used)))
    return {'unexpanded_latex':original_text,'macro_definitions':definitions}


def formula_gate(text):
    if not 8<=len(text.encode())<=8192:return 'length_or_trivial_fragment'
    if any(c in text for c in ('\ufffd','\x00','`','#','%','$')):return 'control_parameter_or_damaged_encoding'
    if any(ord(c)<32 and c not in '\n\r\t' for c in text):return 'control_character'
    if FORBIDDEN.search(text):return 'document_metadata_or_executable_command'
    if re.search(r'\\\d',text):return 'numeric_control_sequence'
    plain=re.sub(r'\\(?:begin|end)\{[^{}]*\}','',text)
    # Mathematical operator/text blocks may carry meaningful names; prose
    # without any mathematical payload is still rejected below.
    plain=re.sub(r'\\(?:operatorname|text|textrm|mathrm|mathsf)\{[^{}]*\}','',plain)
    plain=re.sub(r'\\[A-Za-z]+|\\[^A-Za-z]','',plain)
    if re.search(r'[A-Za-z]{4,}',plain):return 'unformatted_multiword_or_prose'
    lex=[m.group() for m in TOKEN.finditer(text)]
    if not ADVANCED.search(text) and len(lex)<12:return 'insufficient_advanced_math_evidence'
    if not re.search(r'[=+*/^_<>]|\\(?:in|to|forall|exists|partial|nabla|Delta|sum|prod|int|lim)(?![A-Za-z])',text):return 'isolated_symbol_or_label'
    if re.fullmatch(r'\\(?:text|textrm|mathrm|operatorname)\{[\s\S]*\}',text):return 'text_only_math_environment'
    if not re.search(r'\\begin\{(?:aligned|gathered|cases)',text):
        if re.match(r'^\s*[=<>]',text) or re.search(r'[=+*/<>]\s*$',text):return 'incomplete_relation_or_operator'
    for match in re.finditer(r'\\(?:frac|dfrac|tfrac|sqrt|overset|underset)(?![A-Za-z])',text):
        first=group(text,match.end())
        if first is not None and not first[0].strip():return 'empty_required_math_argument'
        if first is not None and 'frac' in match[0]:
            second=group(text,first[1])
            if second is not None and not second[0].strip():return 'empty_required_math_argument'
    return None


def classify(text, paper_domain, categories):
    tags=set()
    # A condition inside a subscript, e.g. |_{t=0}, is not an equation.
    depth=0;relation=False
    for match in re.finditer(r'\\[A-Za-z]+|\\[^A-Za-z]|[^\s]',text):
        token=match[0]
        if token=='{':depth+=1
        elif token=='}':depth=max(0,depth-1)
        elif depth==0 and token in ('=',r'\le',r'\leq',r'\ge',r'\geq',r'\ne',r'\neq',r'\approx'):relation=True
    partial=bool(re.search(r'\\(?:d?frac|tfrac)\s*\{[^{}]*\\partial|\\partial(?:\^\{?\d+\}?)?\s*_[{A-Za-z]',text))
    laplace=bool(re.search(r'\\(?:Delta|nabla)\s*(?:\^\{?2\}?)?\s*[uvfw]|\\(?:div|operatorname)\{?div',text))
    derivative_of_function=bool(re.search(r'\\(?:d?frac|tfrac)\s*\{\s*\\partial\s*(?:\^\{?\d+\}?\s*)?(?:[A-Za-z]|\\[A-Za-z]+)',text)
        or re.search(r'\\partial(?:\^\{?\d+\}?)?\s*_(?:\{[^{}]+\}|[A-Za-z])\s*(?:[A-Za-z]|\\[A-Za-z]+)',text))
    if relation and 'math.AP' in categories and ((partial and derivative_of_function) or laplace):tags.add('pde_equation')
    elif relation and partial:tags.add('partial_differential_relation')
    elif partial or laplace:tags.add('differential_expression')
    if relation and re.search(r'\\(?:frac|dfrac)\s*\{\s*d(?:\^\{?\d+\}?)?\s*[A-Za-z]|\\(?:dot|ddot)(?![A-Za-z])',text):tags.add('ode_equation' if paper_domain=='ode' else 'ordinary_differential_relation')
    if re.search(r'\\begin\{(?:[pbBvV]?matrix|smallmatrix)\}|\\(?:det|operatorname)\{?(?:det|rank|tr)',text):tags.add('matrix_formula')
    if r'\begin{cases}' in text:tags.add('piecewise_or_system')
    if re.search(r'\\begin\{(?:aligned|gathered)',text):tags.add('multirow_equation')
    if re.search(r'\\(?:int|iint|iiint|oint)(?![A-Za-z])',text):tags.add('integral_formula')
    if re.search(r'\\(?:sum|prod)(?![A-Za-z])',text):tags.add('sum_or_product')
    if re.search(r'\\(?:lim|limsup|liminf)(?![A-Za-z])',text):tags.add('limit_or_asymptotic')
    if re.search(r'\\(?:forall|exists|implies|iff|land|lor|neg)(?![A-Za-z])',text):tags.add('logical_formula')
    if not tags:tags.add('identity_or_bound' if relation else 'mathematical_expression')
    # Paper subject and structural formula type are distinct axes. PDE primary
    # needs derivative-equation evidence, never just the paper's subject label.
    domain='pde' if 'pde_equation' in tags else 'ode' if 'ode_equation' in tags else paper_domain
    if domain in ('pde','ode') and not tags & {'pde_equation','ode_equation'}:domain='analysis'
    return domain,sorted(tags)


def template_key(text):
    output=[];symbols={}
    for token in re.findall(r'\\[A-Za-z]+|\\[^A-Za-z]|[0-9]+(?:\.[0-9]+)?|[^\s]',text):
        if re.fullmatch(r'[A-Za-z]',token):symbols.setdefault(token,len(symbols));output.append('S'+str(symbols[token]))
        elif token[0:1].isdigit():output.append('N')
        else:output.append(token)
    return ''.join(output)
