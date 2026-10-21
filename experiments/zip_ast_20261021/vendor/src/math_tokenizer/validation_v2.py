"""Formula-only whitelist including complete matrix environments."""
import re
from .corpus import validate as validate_v1,ALLOWED
MATRIX_ENVS={'matrix','pmatrix','bmatrix','Bmatrix','vmatrix','Vmatrix','smallmatrix'}
KNOWN_OPERATORS={'det','tr','rank','spec','diag','span','Re','Im','Var','Cov','arg','sgn','lcm','erf','erfc','Ai','Bi','Li','Ei','Cl','res','Res','sech','csch','sinc','and','or','not','SL','PSL','SU','GL','Aut','Hom','End','ord','vol','dist','supp'}
ALLOWED_V2=ALLOWED|{'begin','end','mathbin','mathrel','mathord','mathopen','mathclose','mathinner','differential','lVert','rVert','imath','jmath','triangle','triangleq','nmid','mid','notin','ddots','gets','Re','Im','hom','ker','deg','dim','Pr','liminf','limsup','overbrace','underbrace'}

def validate(text):
    if not isinstance(text,str):return 'not_string'
    environments=re.findall(r'\\(begin|end)\{([^{}]+)\}',text)
    stack=[]
    for action,name in environments:
        if name not in MATRIX_ENVS:return 'unsupported_math_environment:'+name
        if action=='begin':stack.append(name)
        elif not stack or stack.pop()!=name:return 'mismatched_environment'
    if stack:return 'unclosed_environment'
    cleaned=re.sub(r'\\(?:begin|end)\{[^{}]+\}','',text)
    if environments:cleaned=cleaned.replace(r'\\',' ').replace('&',' ')
    elif r'\\' in cleaned or '&' in cleaned:return 'multiline_alignment_requires_review'
    # Multi-letter named mathematical operators have explicit whitelist; prose
    # remains rejected. Do not accept arbitrary operatorname words.
    def operator(m):
        return r'\alpha' if m[1] in KNOWN_OPERATORS else m[0]
    cleaned=re.sub(r'\\operatorname\{([A-Za-z]+)\}',operator,cleaned)
    unknown=set(re.findall(r'\\([A-Za-z]+)',cleaned))-ALLOWED_V2
    if unknown:return 'unknown_macro:'+','.join(sorted(unknown))
    # Feed v1's prose/brace/length filter a standard placeholder for extra
    # mathematical commands. Stored/evaluated strings are never substituted.
    extra=ALLOWED_V2-ALLOWED
    cleaned=re.sub(r'\\[A-Za-z]+',lambda m:r'\alpha' if m[0][1:] in extra else m[0],cleaned)
    return validate_v1(cleaned)
