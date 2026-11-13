# Tunnel M-P Interaction Check

Python tool that builds EC2 M-P interaction diagrams for RC tunnel sections, parses STAAD.Pro member forces, and checks demand/capacity (D/C) ratios for raft, wall and arch members.

## What it does
1. Builds the axial force (P) vs. moment (M) capacity envelope of a reinforced concrete section using strain compatibility and the EC2 equivalent rectangular stress block.
2. Reads member end forces from a STAAD.Pro `.ANL` analysis output file.
3. Plots the demand points (P, M) for every load combination against the capacity envelope.
4. Calculates a D/C ratio for each member and load combination, and flags each as OK or FAIL.

## Method
- Code basis: EN 1992-1-1 (Eurocode 2)
- Partial factors: γc = 1.5, γs = 1.15
- Ultimate concrete strain: εcu = 0.0035
- Capacity curve: neutral axis sweep with strain compatibility, symmetric reinforcement on both faces
- D/C ratio: a ray is drawn from the origin through the demand point, and D/C = demand distance / envelope boundary distance along that ray
- Load combinations are discovered automatically from the `.ANL` file

## Default section data
| Zone | Members | Thickness | Reinforcement | Cover |
|---|---|---|---|---|
| Raft | 1-4 | 800 mm | T16@200 each face | 60 mm |
| Walls | 5-10 | 600 mm | T16@200 each face | 60 mm |
| Arch | 11-16 | 600 mm | T16@200 each face | 60 mm |

Materials: fck = 25 MPa, fy = 500 MPa, strip width b = 1000 mm. Edit these values in the `__main__` block of the script.

## Requirements
- Python 3.9+
- numpy
- matplotlib
- scipy

```
pip install numpy matplotlib scipy
```

## How to run
1. Export the analysis results from STAAD.Pro as an `.ANL` file.
2. Set the file path in the script:
   ```python
   ANL_FILE = "sample_input.ANL"
   ```
3. Run:
   ```
   python mp_interaction_check.py
   ```

## Output
- `tunnel_portal_interaction.png`: M-P diagram with the capacity envelopes, demand points, section sketch and D/C summary table
- A D/C summary table printed in the console, listing each zone, member, load combination and OK/FAIL status

## Limitations
- Uniaxial bending only (My is not considered)
- Simplified section model for preliminary checks
- Results must be verified independently before use in any design

## Author
Ujjwal Lohani
