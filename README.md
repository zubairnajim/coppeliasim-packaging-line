# Robotic packaging and palletizing line

A complete production line simulated in [CoppeliaSim](https://www.coppeliarobotics.com/) and driven
from Python: a delta robot packs products from one conveyor into cartons on another, a cobot
palletizes the cartons, a large industrial robot and a mobile robot move the pallets, a camera with
a neural network grades every product, and a humanoid robot sorts the rejects. An operator
dashboard in the browser shows the line live and controls it.

![Delta robot packing bars into cartons](docs/img/packing.gif)

![Operator dashboard](docs/img/dashboard.png)

## The line

| # | Station | Robot / equipment | What it does |
|---|---------|-------------------|--------------|
| 1 | Infeed | Feeder + conveyor A | Soap bars arrive at random position and angle; about 20 % carry a defect |
| 2 | Vision QC | Camera + CNN | Grades every bar on the fly: `good`, `rework` (small blemishes), `scrap` (crack / burn) |
| 3 | Packing | ABB IRB 360 delta | Tracks good bars on the moving conveyor, aligns them and places 4 per carton on conveyor B |
| 4 | Case sealing | Sealer tunnel | Closes full cartons; cartons then accumulate at the end stop |
| 5 | Palletizing | UR10 cobot (vacuum) | Stacks 12 cartons per pallet in 3 mirrored layers |
| 6 | Pallet handling | ABB IRB 4600 + fork | Takes the full pallet to the mobile robot and fetches an empty pallet from the stack |
| 7 | Intralogistics | KUKA Omnirob (AMR) | Carries the pallet to the outbound roller conveyor and returns to its dock |
| 8 | Reject sorting | Baxter humanoid + CNN | Re-inspects each rejected bar and bins it with the left (scrap) or right (rework) arm |

| | |
|---|---|
| ![Whole line](docs/img/overview.jpg) | ![Packing cell](docs/img/packing.jpg) |
| ![Cobot palletizing](docs/img/palletizing.jpg) | ![Pallet handling](docs/img/pallet-handling.jpg) |
| ![Pallet handling in motion](docs/img/pallet-handling.gif) | ![Baxter sorting rejects](docs/img/sorting.jpg) |

## Operator dashboard

`python -m factory.dashboard` serves a local web page (http://127.0.0.1:8050) that runs outside
the simulator and owns the line:

* **Key figures**: bars packed, bars still unpacked on the line, rejects (rework / scrap), cartons,
  pallets built and shipped, throughput, and inspection accuracy.
* **Line overview**: a live top view with every bar (shape and colour show its grade), carton fill
  levels, robot positions, the pallet being built, the AMR and the outbound pallets.
* **Vision inspection**: the pictures the cameras take, each with its grade and confidence; filter
  to defects or to misreads, click a thumbnail to inspect it.
* **Live camera**: four scene cameras rendered by the simulator.
* **Flow rate**: fed, packed and rejected bars per minute, with a hover readout and a table view.
* **Control**: Start, Hold / Resume, Stop and E-STOP; live setpoints for feed rate, conveyor speed,
  defect shares, pallet layers and the grading source; a cycle stop per station; simulation speed.
* **Event log**: pallets completed, AMR departures, misreads, operator actions.

The page uses no external libraries and only accepts commands from itself on `127.0.0.1`.

## Machine vision

A small CNN grades 128x128 top-down images taken by two cameras in the scene (one above the moving
infeed conveyor, one at the reject stop in front of Baxter).

![Training samples: good, rework, scrap](docs/img/qc-samples.png)

Every class has 60 different bars with random blemishes, cracks and burns. The split is by bar
identity (40 train, 8 validation, 12 test), so test bars are never seen in training. The shipped
model scores 100 % on the 576 test images (`factory/models/qc_metrics.json`). The defects are
synthetic and high-contrast, so this number says the pipeline works, not how it would do on real
products.

## Run it

Requirements: CoppeliaSim 4.10 (the Edu version is fine) and Python 3.10+.

```bash
pip install -r requirements.txt
```

Open `factory.ttt` in CoppeliaSim, then:

```bash
python -m factory.dashboard
```

Press **Start** on the page. Without the dashboard:

```bash
python -m factory.controller --realtime
```

Other entry points:

| Command | Purpose |
|---|---|
| `python -m factory.build_scene` | Rebuild the whole scene from code and save `factory.ttt` |
| `python -m factory.train_qc` | Collect a fresh image set from the cameras and retrain the network |
| `python -m factory.record --time 210 --view all --video out/line.mp4` | Record the scene cameras to a video |
| `python -m factory.record --time 60 --check` | Print how closely each robot follows its targets |

## How it works

* `factory/build_scene.py` creates every object through the remote API: conveyors, gantry, tools,
  stands, bins, cameras, product / carton / pallet masters, and inserts the stock robot models.
* `factory/controller.py` is the cell controller. Each station is a Python coroutine advanced once
  per simulation step. They coordinate through shared line state (carton queue, pallet station, AMR
  dock), which gives natural back-pressure when a station is busy or stopped.
* `factory/bridge.lua` is a small script inside the scene. It solves the inverse kinematics of all
  five arms, including the delta's closed chains, and applies all pose updates of a step in one
  remote call, which keeps the simulation faster than real time.
* `factory/dashboard.py` + `factory/dashboard.html` are the dashboard server (Python standard
  library only) and page.
* `factory/vision.py` + `factory/train_qc.py` hold the network and its training pipeline.
* `factory/layout.py` holds every dimension and position.

In a 400 s validation run the line packed 257 of 341 bars into 64 cartons, built 5 pallets, missed
no good bar, had no IK failure and kept every tool within 0.5 mm of its planned path, at about
twice real time.

## Limits

* The simulation is kinematic. Robots follow planned tool paths through IK and parts attach to
  grippers by re-parenting; contacts, friction and payload dynamics are not simulated.
* Robot paths are not collision-checked.
* The robot models come from CoppeliaSim's model library.
