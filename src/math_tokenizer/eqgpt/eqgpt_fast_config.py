"""Frozen quick reward-search protocol and target-excluded compact tokenizers."""
import json
from pathlib import Path

from math_tokenizer.eqgpt.latex_tokenizers import LatexAdapter
from math_tokenizer.eqgpt.eqgpt_fast_tokenizers import train_or_load

CASES=('Wave_equation','Convection_diffusion_equation','KG_equation',
       'Burgers_equation','Eq_6_2_12','PDE_divide','KdV_equation',
       'Chaffee_Infante_equation')
SIZES=(280,315,350)
ALPHABET=list(range(32,127))+[9,10,13]
EPOCHS=25
LAYERS=3
ATTEMPTS=200
CYCLES=3
CONTINUATION=2
OBSERVATIONS=10000
SEED=525


def load_or_train_tokenizers(texts,folder):
    models=train_or_load(texts,Path(folder),SIZES,ALPHABET)
    return {f'{method}_{size}':LatexAdapter(method,model)
            for (method,size),model in models.items()}
