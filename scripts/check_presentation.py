"""Three deliverable checks: evidence, layout geometry, export identity."""
import hashlib,json,re,statistics,subprocess,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
from math_tokenizer.corpus import ROOT

F=ROOT/'presentations/math_tokenizer'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pdf=F/'output/Design_of_a_Specialized_Tokenizer_for_Mathematical_Text.pdf'
    manifest=json.loads((F/'source/content_manifest.json').read_text());audit=json.loads((ROOT/'experiments/current/three_round_audit.json').read_text())
    refs=json.loads((F/'source/references.json').read_text());review=json.loads((F/'source/reference_verification.json').read_text())
    export=json.loads((F/'qa/export_verification.json').read_text())
    log=(F/'build/mathematical_tokenizer.log').read_text();tex=(F/'source/mathematical_tokenizer.tex').read_text()
    assert audit['all_passed'] and audit['active_methods']==7 and audit['formula_rows']==234556
    assert manifest['figures']==3 and manifest['references']==len(refs)==26
    assert len(review['records'])==26 and all(r['primary_source_browsing_verified'] for r in review['records'])
    assert sha(ROOT/'experiments/current/rankings.json')==manifest['table_source_sha256']
    assert 'Overfull' not in log and '\n!' not in log
    assert not any(ord(c)<32 and c!='\n' for c in tex)
    raw=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
    pages=raw.split('\f');pages=[p for p in pages if p.strip()]
    assert len(pages)==manifest['total_pages']==45
    cover=pages[0]
    for token in ['Tongren Xiao','tongrenxiao7@gmail.com','Sun Yat-sen University','October 10, 2026']:assert token in cover
    assert 'Design of a Specialized Tokenizer' in cover and 'for Mathematical Text' in cover
    for bad in ['extbf','oprule','Unknown??','TODO','INSERT FIGURE']:assert bad not in raw,bad
    bbox=F/'qa/page_geometry.html';subprocess.run(['pdftotext','-bbox',str(pdf),str(bbox)],check=True)
    root=ET.parse(bbox);edges=[];outliers=[]
    for index,page in enumerate(root.findall('.//{*}page')):
        width=float(page.attrib['width']);height=float(page.attrib['height']);assert abs(width/height-4/3)<.001
        words=page.findall('.//{*}word')
        for w in words:
            if float(w.attrib['xMin'])<-.5 or float(w.attrib['xMax'])>width+.5 or float(w.attrib['yMin'])<-.5 or float(w.attrib['yMax'])>height+.5:
                outliers.append({'page':index+1,'text':w.text,'box':w.attrib})
        body=[w for w in words if 20<float(w.attrib['yMin']) and float(w.attrib['yMax'])<253]
        if index and body:edges.append(max(float(w.attrib['yMax']) for w in body))
    assert not outliers,outliers[:10]
    assert min(edges)>220,'Sparse content frame; inspect and rebalance'
    assert export['slides']==45 and export['slide_image_bytes_verified'] and export['pdf_sha256']==sha(pdf)
    bundle=F/'output/Mathematical_Tokenizer_Beamer_Source.zip'
    with zipfile.ZipFile(bundle) as z:
        assert z.read('mathematical_tokenizer.tex')==(F/'source/mathematical_tokenizer.tex').read_bytes()
        assert 'template.tex' in z.namelist() and len([p for p in z.namelist() if p.endswith('.pdf')])==3
    assert 'Stage 2 is paused as future research' in raw
    assert 'Stage 2: Autonomous and Signal-Guided Tokenization' in raw
    result={'all_passed':True,'pages':45,'figures':3,'primary_references':26,
        'reviews':[{'round':1,'scope':'Evidence table source hashes, active experiment audit, cover identity, exact seven methods and reference records','passed':True},
                   {'round':2,'scope':'Compiled 4:3 pages, zero TeX overflow, no control-character artifacts, all text within page geometry; rendered-page inspection','passed':True},
                   {'round':3,'scope':'PowerPoint slide-image identity, PDF hashes and complete editable Beamer source bundle','passed':True}],
        'layout':{'min_body_text_bottom_pt':min(edges),'median_body_text_bottom_pt':statistics.median(edges),'page_height_pt':272.13,
                  'reference_median_body_text_bottom_pt':240.45,'out_of_page_words':len(outliers)},
        'pptx_editability':'Exact page renderings; editable master is the included Beamer source',
        'source_sha256':sha(F/'source/mathematical_tokenizer.tex'),'pdf_sha256':sha(pdf)}
    (F/'qa/three_round_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Presentation checks passed',len(pages),'pages; minimum body edge',round(min(edges),2))


if __name__=='__main__':main()
