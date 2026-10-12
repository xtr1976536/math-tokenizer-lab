from math_tokenizer.benchmark_v5 import template_key
from math_tokenizer.validation_v2 import validate
from math_tokenizer.corpus_v2 import tex_members
import io,tarfile

def test_template_renaming_shape():
 assert template_key('x+2=y')==template_key('a+3=b')
 assert template_key('x+x=y')!=template_key('a+b=c')
 assert template_key(r'\alpha+x=1')!=template_key(r'\beta+x=1')


def test_safe_archive_never_extracts_paths(tmp_path):
 stream=io.BytesIO()
 with tarfile.open(fileobj=stream,mode='w:gz') as t:
  member=tarfile.TarInfo('../../unsafe.tex');data=b'$x+1=2$';member.size=len(data);t.addfile(member,io.BytesIO(data))
 assert list(tex_members(stream.getvalue()))==[('../../unsafe.tex','$x+1=2$')]
 assert list(tmp_path.iterdir())==[]
