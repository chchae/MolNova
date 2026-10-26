"""Generate a small frame-cycle example with a distant artificial receptor."""
import argparse
from pathlib import Path
from make_toy_inputs import create


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    ns=parser.parse_args()
    filename=create(ns.directory,1,2)
    text=filename.read_text().replace('growth-mode = "frontier" # legacy artificial frontier regression example',
        'growth-mode = "frame-cycles"\nreference-mcs = "c1([*:999])c([*:999])cccc1"\nelite-count = 20\nelite-rank-by = "qed" # synthetic example only; choose your real ranking criterion\nnext-attachment-count = 3\nattachment-distance = "bonds"\nmcs-timeout = 5')
    text=text.replace('sidechains-per-site = 3 # small artificial library; retain legacy ordered controls here','sidechains-per-site = 2 # two-site first cycle: 2^2 = 4 combinations')
    text=text.replace('sidechain-min-heavy-atoms = 1','sidechain-min-heavy-atoms = 5').replace('sidechain-max-heavy-atoms = 1000','sidechain-max-heavy-atoms = 10')
    text=text.replace('sidechain-selection = "ordered"','sidechain-selection = "random"').replace('sidechain-include-h = true','sidechain-include-h = false')
    text=text.replace('conformers = 3','conformers = 1').replace('torsion-trials = 2','torsion-trials = 0').replace('stop-mode = "time"','stop-mode = "cycles"')
    filename.write_text(text)
    (filename.parent/'sidechains.smi').write_text('\n'.join(['*CCCCC','*CCCCCC','*CCCCCCC','*CCCCCCCC','*CCCCCCCCC','*CCCCCCCCCC','*CCCCO','*CCCCN','*CCCCCO','*CCCCCN'])+'\n')
    print(filename)


if __name__=='__main__':main()
