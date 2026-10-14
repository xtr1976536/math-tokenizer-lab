"""Prepare Beamer pages and export PPTX with bundled Artifact Tool.

Python handles PDF/ZIP only. Editable content remains in the Beamer source.
"""
from datetime import datetime,timezone
import hashlib,json,subprocess,zipfile
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'presentations/math_tokenizer'
RUNTIME=Path('/Users/xiao/.cache/codex-runtimes/codex-primary-runtime/dependencies')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=FOLDER/'output';out.mkdir(exist_ok=True);name='Design_of_a_Specialized_Tokenizer_for_Mathematical_Text'
    data=(FOLDER/'build/mathematical_tokenizer.pdf').read_bytes();pdf=out/(name+'.pdf');pdf.write_bytes(data)
    (out/(name+'_Updated.pdf')).write_bytes(data)
    pages=FOLDER/'build/revision_pages';pages.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-png','-scale-to','2000',str(pdf),str(pages/'slide')],check=True)
    reader=PdfReader(pdf)
    expected=json.loads((FOLDER/'source/content_manifest.json').read_text())['total_pages']
    images=[p for p in sorted(pages.glob('slide-*.png')) if int(p.stem.split('-')[-1])<=expected]
    assert len(images)==len(reader.pages)==expected
    final=out/(name+'_Updated.pptx')
    if final.exists():final=out/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.pptx')
    links=[]
    destinations=reader.named_destinations
    for page in reader.pages:
        hotspots=[];width=float(page.mediabox.width);height=float(page.mediabox.height)
        for reference in page.get('/Annots',[]):
            annotation=reference.get_object();action=annotation.get('/A',{});rect=annotation.get('/Rect')
            if rect is None:continue
            x0,y0,x1,y1=map(float,rect)
            rect_fraction=[max(0.,x0/width),max(0.,(height-y1)/height),min(width,x1-x0)/width,min(height,y1-y0)/height]
            item={'rect_fraction':rect_fraction}
            if action.get('/S')=='/URI':item['uri']=str(action['/URI'])
            elif action.get('/S')=='/GoTo':
                target=action.get('/D')
                if isinstance(target,str) and target in destinations:
                    item['target_slide']=reader.get_destination_page_number(destinations[target])+1
                else:continue
            else:continue
            hotspots.append(item)
        links.append(hotspots)
    prep={'images':[str(p) for p in images],'notes':[p.extract_text() or '' for p in reader.pages],'finalPptx':str(final),'links':links}
    (FOLDER/'build/export_revision_input.json').write_text(json.dumps(prep,indent=2))
    subprocess.run([str(RUNTIME/'node/bin/node'),str(ROOT/'scripts/export_presentation_revision.mjs')],check=True,cwd=ROOT)
    with zipfile.ZipFile(final) as z:
        blobs=[z.read(n) for n in z.namelist() if n.startswith('ppt/media/') and n.endswith('.png')]
        assert len(blobs)==expected and {hashlib.sha256(b).hexdigest() for b in blobs}=={sha(p) for p in images}
    canonical=out/(name+'.pptx');canonical.write_bytes(final.read_bytes())
    bundle=out/'Mathematical_Tokenizer_Beamer_Source.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((FOLDER/'source').rglob('*')):
            if p.is_file():z.write(p,p.relative_to(FOLDER/'source'))
    record={'slides':expected,'aspect_ratio':'4:3','pptx_is_exact_page_rendition':True,'hyperlink_hotspots':sum(map(len,links)),
      'editable_master':'source/mathematical_tokenizer.tex','slide_image_bytes_verified':True,'exporter':'Artifact Tool',
      'pdf_sha256':sha(pdf),'pptx_sha256':sha(canonical),'source_bundle_sha256':sha(bundle),'future_research_added':True}
    (FOLDER/'qa/export_verification.json').write_text(json.dumps(record,indent=2))
    print('Exported',expected,'Beamer pages and source bundle.')
if __name__=='__main__':main()
