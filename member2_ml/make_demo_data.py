"""Create explicitly synthetic test data ONLY for checking wiring; NEVER use as hackathon evidence."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='data');a=parser.parse_args()
    rng=np.random.default_rng(123)
    candidates=[]; labels=[]
    for split,n in [('train',6),('student_test',2)]:
        for trip in range(n):
            trip_id=f'{split}_{trip:03d}'
            for observation in range(12):
                reference=str((trip+observation)%6)
                labels.append(dict(split=split,trip_id=trip_id,observation_idx=observation,reference_arc_id=reference))
                # Some observations lack the positive candidate, to test honest denominators.
                arcs=[str((trip+observation+k)%6) for k in range(3)]
                if observation%7==0: arcs=arcs[1:]
                for arc in arcs:
                    correct=int(arc==reference)
                    candidates.append(dict(split=split,trip_id=trip_id,observation_idx=observation,
                        arc_id=arc,distance_m=max(0,rng.normal(8 if correct else 38,8)),
                        heading_diff_deg=max(0,rng.normal(12 if correct else 80,25)),
                        gps_speed_mps=max(0,rng.normal(12,3)),time_gap_s=1.0))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(candidates).to_csv(out/'demo_candidates.csv',index=False)
    pd.DataFrame(labels).to_csv(out/'demo_labels.csv',index=False)
    print('SYNTHETIC demo CSVs generated. Not hackathon training results.')

if __name__=='__main__':main()
