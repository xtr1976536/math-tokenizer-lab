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
    reader=PdfReader(pdf);images=sorted(pages.glob('slide-*.png'))
    expected=json.loads((FOLDER/'source/content_manifest.json').read_text())['total_pages']
    assert len(images)==len(reader.pages)==expected
    final=out/(name+'_Updated.pptx')
    if final.exists():final=out/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.pptx')
    prep={'images':[str(p) for p in images],'notes':[p.extract_text() or '' for p in reader.pages],'finalPptx':str(final)}
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
    record={'slides':expected,'aspect_ratio':'4:3','pptx_is_exact_page_rendition':True,
      'editable_master':'source/mathematical_tokenizer.tex','slide_image_bytes_verified':True,'exporter':'Artifact Tool',
      'pdf_sha256':sha(pdf),'pptx_sha256':sha(canonical),'source_bundle_sha256':sha(bundle),'future_research_added':True}
    (FOLDER/'qa/export_verification.json').write_text(json.dumps(record,indent=2))
    print('Exported',expected,'Beamer pages and source bundle.')
if __name__=='__main__':main()
