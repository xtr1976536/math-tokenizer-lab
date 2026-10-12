import gzip,json
import pytest
from math_tokenizer.result_rows import PerFormulaStore


def test_streaming_spool_preserves_all_rows_and_rejects_split_key(tmp_path):
    path=tmp_path/'results.gz'
    rows=[{'key':'a','suite':'x','tokens':1},{'key':'a','suite':'x','tokens':2},{'key':'b','suite':'y','tokens':3}]
    with gzip.open(path,'wt') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    store=PerFormulaStore(path)
    assert store[('a','x')]==rows[:2] and store[('b','y')]==rows[2:]
    assert store[('a','x')]==rows[:2]
    store.close()
    with gzip.open(path,'wt') as stream:
        for row in rows+[rows[0]]:stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError):PerFormulaStore(path)
