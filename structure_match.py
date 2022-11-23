#!/usr/bin/python

"""
This module uses ase to read in two or more structures and then
pymatgen to compare the structural similarity of them.


"""
import numpy as np
from ase.io import read
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
import itertools



def match_structures(images, labels):
    structures = {}
    for i, atoms in enumerate(images):
        structure = AseAtomsAdaptor.get_structure(atoms)
        label = labels[i]
        structures[label] = structure

    a = StructureMatcher()
    results = []
    for n1, n2 in itertools.combinations(structures.keys(), 2):
        s1 = structures[n1]
        s2 = structures[n2]
        d = a.get_rms_anonymous(s1, s2)
        if d[0] is not None:
            r = {"rms": d[0], "match_found": True,
                "mapping": {str(k): str(v) for k, v in d[1].items()}}
        else:
            r = {"match_found": False, "mapping": None, "rms": None}
        r["files"] = [n1, n2]
        results.append(r)

    return results


# *****************************************
#             Main program
# *****************************************
if __name__ == "__main__":
    import sys

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--format", dest="format",
                    help="allowed ASE file format", action="store_true")



    # TODO : fix the format specifcation

    (options, args) = parser.parse_args()

    assert len(args) > 1, "Please provide the path to at least two structures"


    # list to hold the ASE atoms objects
    images =  []

    for arg in args:
        try:
            atoms = read(arg)
            images.append(atoms)
        except IOError:
            sys.stderr.write(
                "There was a problem opening the structure file. Does it exist at all / is it in a known format?")
            sys.exit(1)

    # match structures
    results = match_structures(images, args)
    print(" file 1  file 2   rms (Å)   mappings ")
    max_arglen = max([len(arg) for arg in args])
    for r in results:
        if r['match_found']:
            print(f"   {r['files'][0]: >{max_arglen}}  {r['files'][1]: >{max_arglen}}    {r['rms']:6.0e}  {r['mapping']}")
        else:
            print(f"   {r['files'][0]: >{max_arglen}}  {r['files'][1]: >{max_arglen}}    match not found")

