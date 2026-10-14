"""Preserve PDF reference/source hyperlinks as transparent native PPTX hotspots.

Page images keep the Beamer typography; native shapes carry clickable navigation.
This edits OOXML before the Artifact Tool finalizer validates the candidate.
"""
import argparse,json,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

P='http://schemas.openxmlformats.org/presentationml/2006/main'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
for name,uri in [('p',P),('a',A),('r',R)]:ET.register_namespace(name,uri)
def tag(ns,name):return '{'+ns+'}'+name


def main():
    parser=argparse.ArgumentParser();parser.add_argument('pptx',type=Path);parser.add_argument('input',type=Path);args=parser.parse_args()
    prep=json.loads(args.input.read_text());links=prep.get('links',[])
    with zipfile.ZipFile(args.pptx) as z:files={n:z.read(n) for n in z.namelist()}
    count=internal=0
    for page,hotspots in enumerate(links,1):
        if not hotspots:continue
        part=f'ppt/slides/slide{page}.xml';rels=f'ppt/slides/_rels/slide{page}.xml.rels'
        root=ET.fromstring(files[part]);tree=root.find('.//'+tag(P,'spTree'))
        relation=ET.fromstring(files[rels]) if rels in files else ET.Element(tag(REL,'Relationships'))
        maxid=max(int(x.attrib['id']) for x in root.findall('.//'+tag(P,'cNvPr')))
        used={x.attrib['Id'] for x in relation};ridindex=1
        for hotspot in hotspots:
            while 'rId'+str(ridindex) in used:ridindex+=1
            rid='rId'+str(ridindex);used.add(rid);maxid+=1
            attrs={'Id':rid}
            if 'target_slide' in hotspot:
                attrs.update(Type=R+'/slide',Target=f"slide{hotspot['target_slide']}.xml");action={'action':'ppaction://hlinksldjump'};internal+=1
            else:attrs.update(Type=R+'/hyperlink',Target=hotspot['uri'],TargetMode='External');action={}
            ET.SubElement(relation,tag(REL,'Relationship'),attrs)
            shape=ET.SubElement(tree,tag(P,'sp'));nv=ET.SubElement(shape,tag(P,'nvSpPr'))
            prop=ET.SubElement(nv,tag(P,'cNvPr'),{'id':str(maxid),'name':f'Link {maxid}'});action[tag(R,'id')]=rid
            ET.SubElement(prop,tag(A,'hlinkClick'),action);ET.SubElement(nv,tag(P,'cNvSpPr'));ET.SubElement(nv,tag(P,'nvPr'))
            sppr=ET.SubElement(shape,tag(P,'spPr'));transform=ET.SubElement(sppr,tag(A,'xfrm'))
            x,y,width,height=hotspot['rect_fraction']
            ET.SubElement(transform,tag(A,'off'),{'x':str(round(x*9144000)),'y':str(round(y*6858000))})
            ET.SubElement(transform,tag(A,'ext'),{'cx':str(round(width*9144000)),'cy':str(round(height*6858000))})
            geometry=ET.SubElement(sppr,tag(A,'prstGeom'),{'prst':'rect'});ET.SubElement(geometry,tag(A,'avLst'))
            fill=ET.SubElement(sppr,tag(A,'solidFill'));color=ET.SubElement(fill,tag(A,'srgbClr'),{'val':'FFFFFF'});ET.SubElement(color,tag(A,'alpha'),{'val':'0'})
            line=ET.SubElement(sppr,tag(A,'ln'));ET.SubElement(line,tag(A,'noFill'));count+=1
        files[part]=ET.tostring(root,encoding='utf-8',xml_declaration=True);files[rels]=ET.tostring(relation,encoding='utf-8',xml_declaration=True)
    temporary=args.pptx.with_suffix('.linked.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    temporary.replace(args.pptx)
    print('Preserved',count,'PPTX hyperlink hotspots;',internal,'internal reference jumps.')


if __name__=='__main__':main()
