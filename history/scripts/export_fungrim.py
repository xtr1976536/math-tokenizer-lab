"""Use Fungrim's unchanged official semantic-to-LaTeX renderer."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
repo=ROOT/'vendor/fungrim';sys.path.insert(0,str(repo))
import pygrim,pygrim.formulas
revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
locations={}
for path in (repo/'pygrim/formulas').glob('*.py'):
    import re
    text=path.read_text()
    for m in re.finditer(r'ID\("([0-9a-f]+)"\)',text):locations[m[1]]=(str(path.relative_to(repo)),1+text.count('\n',0,m.start()))
rows=[]
for id,entry in sorted(pygrim.formulas.entries_dict.items()):
    file,line=locations.get(id,('unknown',0))
    # Exclude definitions, prose entries and plot/table records.
    for argument in entry.args():
        if argument.head()!=pygrim.Formula:continue
        expression=argument.args()[0]
        domain='analysis'
        if any(x in file for x in ('prime','divisor','dirichlet','zeta','integer')):domain='number_theory'
        elif any(x in file for x in ('matrix','determinant','eigen')):domain='linear_algebra'
        elif any(x in file for x in ('binomial','factorial','bernoulli')):domain='combinatorics'
        elif any(x in file for x in ('logic','set','operators')):domain='logic'
        rows.append({'entry_id':id,'latex':expression.latex(),'semantic_expression':str(expression),
             'source_path':file,'source_line':line,'domain':domain,'source_group':'fungrim:'+file,
             'origin':'unknown','origin_evidence':'Official README says hand-compiled with possible computer generation. No formula-level human/AI certification; not inferred from MIT copyright year.',
             'license':'MIT','revision':revision,'repo':'fredrik-johansson/fungrim',
             'transformation':'Official pygrim Expr.latex renderer unchanged; only Formula entries exported',
             'academic_level':'university_advanced_or_research'})
path=ROOT/'data/raw/fungrim_formulas.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
registry={'id':'fungrim','repo':'fredrik-johansson/fungrim','revision':revision,'license':'MIT','url':'https://github.com/fredrik-johansson/fungrim','export_local':'data/raw/fungrim_formulas.jsonl','export_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'formula_entries':len(rows),'renderer_modified':False}
(ROOT/'data/fungrim_source.json').write_text(json.dumps(registry,indent=2));(ROOT/'references/fungrim_LICENSE').write_bytes((repo/'LICENSE').read_bytes());print(registry)
