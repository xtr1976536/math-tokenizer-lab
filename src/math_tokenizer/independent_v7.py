"""Additional intrinsic probes; independent pylatexenc source nodes, not gold.

No optimization/training calls this parser. Definitions report measurement
coverage and signed effects; no claim of semantic correctness or expert review.
"""
from collections import Counter
import math,re
from .frontend_v6 import reference_lex
from .metrics_v2 import token_bytes


def source_scopes(text):
    from pylatexenc.latexwalker import LatexWalker,LatexGroupNode,LatexMacroNode,LatexEnvironmentNode
    offsets=[0]
    for c in text:offsets.append(offsets[-1]+len(c.encode()))
    try:nodes,_,_=LatexWalker(text,tolerant_parsing=False).get_latex_nodes()
    except Exception:return None
    scopes=set();typed=[]
    def visit(node):
        if isinstance(node,LatexGroupNode):
            a,b=node.pos+len(node.delimiters[0]),node.pos+node.len-len(node.delimiters[1])
            if b>a:scopes.add((offsets[a],offsets[b]));typed.append(('group',offsets[a],offsets[b]))
        if isinstance(node,LatexMacroNode):
            if node.nodeargd:
                for arg in node.nodeargd.argnlist:
                    if arg is not None:visit(arg)
        for child in getattr(node,'nodelist',[]) or []:visit(child)
    for node in nodes:visit(node)
    return sorted(scopes)


def boundaries(model,text):
    bounds=[0]
    for id in model.encode(text):bounds.append(bounds[-1]+len(token_bytes(model,id)))
    if bounds[-1]!=len(text.encode()):raise AssertionError('independent payload protocol not lossless')
    return set(bounds)


def additional(model,records,training_usage=None,scope_cache=None,probe_limit=256):
    tp=total=parsed=nonempty=0;edit=[];scope_rows=[];logbits={str(alpha):0. for alpha in (.01,.1,.5,1.)}
    token_total=byte_total=0;vocab=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size
    denominators={str(alpha):sum(training_usage.values())+alpha*vocab for alpha in (.01,.1,.5,1.)} if training_usage is not None else {}
    for i,r in enumerate(records):
        text=r['latex'];raw=text.encode();bounds=boundaries(model,text)
        spans=scope_cache[r['id']] if scope_cache is not None else source_scopes(text)
        if spans is not None:
            parsed+=1;expected={p for a,b in spans for p in (a,b)}-{0,len(raw)}
            total+=len(expected);tp+=len(expected&bounds);nonempty+=bool(expected)
            scope_rows.append({'id':r['id'],'scope_boundaries':len(expected),'scope_boundaries_preserved':len(expected&bounds)})
        if i<probe_limit:
            # Same-byte lexical symbol substitution; conservatively avoid text
            # and operator-name arguments. It is a local edit, not equivalence.
            if not re.search(r'\\(?:text|operatorname|mathrm|mbox)(?![A-Za-z])',text):
                unit=next(((a,b) for a,b in reference_lex(text) if b-a==1 and raw[a:b] in (b'x',b'y',b'z',b'a',b'b')),None)
                if unit:
                    a,b=unit;changed=(raw[:a]+b'q'+raw[b:]).decode();new=boundaries(model,changed)
                    far={p for p in bounds|new if p<a-16 or p>b+16}
                    edit.append(len((bounds^new)&far)/max(1,len(far)))
        if training_usage is not None:
            ids=model.encode(text);token_total+=len(ids);byte_total+=len(raw)
            for alpha in (.01,.1,.5,1.):
                denominator=denominators[str(alpha)]
                logbits[str(alpha)]+=-sum(math.log2((training_usage[id]+alpha)/denominator) for id in ids)
    return {'independent_parser':'pylatexenc2.11 strict source groups (not arithmetic semantic AST)',
        'independent_scope_boundary_recall':tp/total if total else None,
        'independent_scope_boundaries':total,'independent_scope_parser_coverage':parsed/max(1,len(records)),
        'independent_scope_nonempty_formulas':nonempty,
        'local_symbol_edit_far_boundary_change':sum(edit)/len(edit) if edit else None,'local_symbol_edit_probes':len(edit),
        'unigram_bpb_smoothing_sensitivity':{a:b/max(1,byte_total) for a,b in logbits.items()} if training_usage is not None else None},scope_rows
