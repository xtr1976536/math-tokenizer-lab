"""Registered first-round scope; numerical and model settings stay fixed."""
MAIN_CASES = (
    'KdV_equation', 'Burgers_equation', 'Wave_equation',
    'Chaffee_Infante_equation', 'KG_equation', 'Allen_Cahn',
    'Convection_diffusion_equation', 'PDE_divide', 'Eq_6_2_12',
    'PDE_compound', 'Possion_x_y',
)
LATEX_METHODS = ('ast_bpe',)
METHODS = (*LATEX_METHODS, 'eqgpt_reference')
SEEDS = (525,)
VERSION = 'eqgpt-round1-main-v2'
