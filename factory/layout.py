"""Shared geometry of the factory cell (metres, world frame, floor at z=0)."""
import math

BELT_Z = 0.70                      # top surface of both conveyors
BELT_W = 0.30

# --- Conveyor A: raw products, flows +X ---------------------------------
A_Y = 0.25
A_X0, A_X1 = -2.60, 1.50           # start / end of belt
A_SPEED = 0.15
A_SPAWN_X = -2.45
QC_X = -1.50                       # inspection camera #1
A_END_X = 1.42                     # reject stop in front of Baxter

# --- Conveyor B: cartons, flows -X (counter-flow) -----------------------
B_Y = -0.25
B_X0, B_X1 = 1.10, -2.90
B_SPEED = 0.30
B_SPAWN_X = 1.00
B_FILL_X = 0.00                    # carton stops here under the delta
B_SEAL_X = -1.20                   # case sealer tunnel
B_END_X = -2.78                    # end stop / palletizer pick point
B_SPLIT_X = -0.11                  # indexing section (stops while a carton is filled) | outfeed section (always runs)
B_PITCH = 0.25                     # carton spacing set by the erector on the indexing section
B_QUEUE_PITCH = 0.20
B_QUEUE_MAX = 4

# --- Parts ---------------------------------------------------------------
PROD = (0.07, 0.05, 0.03)
CARTON = (0.17, 0.13, 0.06)        # outer, open top
CARTON_FLOOR = 0.005
LID_T = 0.004
CARTON_H = CARTON[2] + LID_T
SLOTS = [(sx * 0.04, sy * 0.03) for sx in (1, -1) for sy in (1, -1)]  # products per carton
PALLET = (0.30, 0.40, 0.045)       # runners along local x
PALLET_POCKET_Z = 0.03             # underside of deck (fork tine top)
PALLET_GRID = [(sx * 0.07, sy * 0.09) for sx in (1, -1) for sy in (1, -1)]  # carton centres, carton long axis along pallet y
PALLET_LAYERS = 3

# --- Robots --------------------------------------------------------------
DELTA_XY = (0.0, 0.0)
DELTA_TCP_HOME = (0.0, 0.0, 0.95)

UR_BASE = (-3.35, -0.85, 0.60)
UR_HOME = (-3.35 + 0.15, -0.85 + 0.65, 1.00)

STATION = (-4.10, -0.60, 0.40)     # palletizing stand top centre
IRB_BASE = (-5.60, -0.60)
IRB_R = 1.50                       # working radius of all three IRB targets
STACK = (-5.60, 0.90, 0.20)        # empty pallet stack stand top centre (IRB angle +90deg)
DOCK = (-5.60, -2.10)              # AMR loading position (IRB angle -90deg)
STACK_COUNT = 5

AMR_TOP = 0.645
AMR_UNLOAD_X = -1.75
SHIP_X0, SHIP_X1 = -1.05, 2.60     # outbound roller conveyor at y = DOCK[1]
SHIP_SPEED = 0.25

BAXTER = (2.02, 0.25)              # faces -X
BIN_REWORK = (1.68, 0.88, 0.62)    # top-centre of bin opening
BIN_SCRAP = (1.68, -0.38, 0.62)

CAM_H = 0.45                       # QC camera height above belt
CAM_FOV = math.radians(28.0)
CAM_RES = 128
CLASSES = ['good', 'rework', 'scrap']
N_VARIANTS = 60                    # product look-alikes per class (templates)
VIEWS = {                          # hidden render cameras: name -> (eye, look-at)
    'overview': ([-1.6, -8.0, 5.5], [-1.8, -0.7, 0.4]),
    'pack': ([1.6, -2.6, 2.3], [0.2, 0.0, 0.8]),
    'pallet': ([-2.2, -4.0, 2.8], [-4.6, -0.8, 0.6]),
    'baxter': ([0.6, 2.6, 2.2], [1.9, 0.25, 0.8]),
}
BELT_STRIPE_PITCH = 0.20
