"""Eight fixed EqGPT comparison cases and their handbook exclusion labels."""

SHARED8_VERSION = 'eqgpt-shared8-v8-pde-vocab896-pos50-cleanup-100epoch'
SHARED8_EPOCHS = 100
AST_GENERATION_LIMIT = 50
SHARED_POSITION_CAPACITY = 50
PINN_ITERATIONS = 500
NUMERICAL_HIDDEN_LAYERS = 5
NUMERICAL_HIDDEN_WIDTH = 50
SHARED8_CASES = (
    'Wave_equation',
    'Convection_diffusion_equation',
    'KG_equation',
    'Burgers_equation',
    'Eq_6_2_12',
    'PDE_divide',
    'KdV_equation',
    'Chaffee_Infante_equation',
)
SHARED8_TARGETS = (
    'KdV equation',
    'Bateman-Burgers equation',
    '(1.2.10)',
    'Chafee–Infante equation',
    'Klein–Gordon_u',
    '6.2.13',
    '8.14.1d',
    '6.2.12',
)
