import numpy as np
import re
from ase.units import Ha, Bohr
def printme():
    print("me")
def parse_spins(filepath, natoms):
    re_spins = re.compile("Hirshfeld Charges")
    with open(filepath) as f:
        for line in f:
            if re_spins.search(line):
                spins = np.zeros(natoms)
                symbols = []
                next(f)
                next(f)
                for i in range(natoms):
                    fline = next(f)
                    spin = float(fline.split()[6])
                    spins[i] = spin
    return spins


def parse_cell(filepath):
    re_cell = re.compile("\&CELL\n")
    with open(filepath) as f:
        for line in f:
            if re_cell.search(line):
                aline = next(f)
                a = [float(x) for x in aline.split()[1:4]]
                bline = next(f)
                b = [float(x) for x in bline.split()[1:4]]
                cline = next(f)
                c = [float(x) for x in cline.split()[1:4]]
                cell = np.array([a, b, c])
    return cell


def parse_forces(filepath, natoms):
    re_force = re.compile(" ATOMIC FORCES in \[a.u.\]")

    with open(filepath) as f:
        for line in f:
            if re_force.search(line):
                forces = np.zeros((natoms, 3))
                symbols = []
                next(f)
                next(f)
                for i in range(natoms):
                    fline = next(f)
                    _, _, sym, x, y, z = fline.split()
                    symbols.append(sym)
                    forces[i] = np.array([float(x), float(y), float(z)])
    return forces, symbols


def get_natoms(filepath):
    re_natoms = re.compile("- Atoms:")
    with open(filepath) as f:
        for line in f:
            if re_natoms.search(line):
                natoms = int(line.split()[-1])
                return natoms


def get_total_energy(filepath, ionic_step=-1):
    re_energy = re.compile("ENERGY\| Total FORCE_EVAL")
    total_energies = []
    with open(filepath) as f:
        for line in f:
            if re_energy.search(line):
                energy = float(line.split()[-1])
                total_energies.append(energy)
    return total_energies[ionic_step]

def get_hf(filepath, natoms, atom_index, ionic_step=-1, verbose=False, return_components = False):
    re_hf = re.compile("Sca-Rel A_iso")

    with open(filepath) as f:
        iatom = 0
        all_Adip = []
        all_Acon = []
        all_gyro = []

        Adip = np.zeros((natoms, 3, 3))
        Acont = np.zeros(natoms)
        gyros = np.zeros(natoms)
        for line in f:
            if re_hf.search(line):

                # for i in range(natoms):
                idx = int(line.split()[0])
                species = line.split()[1]
                gyros[iatom] = float(line.split()[3]) * 2*np.pi
                Acont[iatom] = float(line.split()[8])
                t = next(f)  # skip non-relativistic ISO
                dipline1 = next(f)
                Adip[iatom][0] = [float(dipline1.split()[0]),
                                  float(dipline1.split()[1]),
                                  float(dipline1.split()[2])]

                dipline2 = next(f)
                Adip[iatom][1] = [float(dipline2.split()[2]),
                                  float(dipline2.split()[3]),
                                  float(dipline2.split()[4])]

                dipline3 = next(f)
                Adip[iatom][2] = [float(dipline3.split()[0]),
                                  float(dipline3.split()[1]),
                                  float(dipline3.split()[2])]
                iatom += 1
            if iatom == natoms:
                iatom = 0

                all_Adip.append(Adip)
                all_Acon.append(Acont)
                all_gyro.append(gyros)

                # reset arrays
                Adip = np.zeros((natoms, 3, 3))
                Acont = np.zeros(natoms)
                gyros = np.zeros(natoms)

    # take the [0,0,1] component of A_ani
    dip = all_Adip[ionic_step][atom_index]
    contact = all_Acon[ionic_step][atom_index]
    gyro_species = all_gyro[ionic_step][atom_index]
    gyro_mu = 851.616

    scale = 1.0  # scale = 0.5 in if A -> Bfield

    # correct units:
    contact = contact * scale * gyro_mu / gyro_species
    dip = dip * scale * gyro_mu / gyro_species

    # add contact to dipolar
    total = dip + np.eye(3) * contact

    # evecs/evals
    evals, evecs = np.linalg.eigh(total)

    # print(f" f_con  {np.linalg.norm(contact): 12.3f} MHz")
    # print(f"|f_dip| {np.linalg.norm(dip): 12.3f} MHz")
    # print(f"|f_tot| {np.linalg.norm(total): 12.3f} MHz")
    # print(f" f_tot  {total[0]:12.4f} {total[1]:12.4f} {total[2]:12.4f} MHz")
    if verbose:
        if ionic_step == -1:
            print(f"Values for the final ionic step (step {len(all_Acon)}):")
        else:
            print(f"Ionic step: {range(len(all_Acon))[ionic_step]}")
        print(f'Principle values and directions of the hyperfine tensor for atom (cp2k index) {range(natoms)[atom_index]+1}:')
        for i in range(3):
            print("{:10.3f} MHz, [{:8.5f}, {:8.5f}, {:8.5f}] (Rounded: [{:4.1f}, {:4.1f}, {:4.1f}])".format(evals[i], *evecs[:,i], *evecs[:,i]))
        

    if return_components:
        return {'contact' : contact,
                'dipolar' : dip,
                'total' : total}
    else:
        return total

def find_matching_directions(A):
    # evecs/evals
    evals, evecs = np.linalg.eigh(A)

    directions = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, -1],
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [1, 1, 1],
    ])
    for ievec, v in enumerate(evecs.T):
        best_match = 0
        best_theta = np.pi

        for i, d in enumerate(directions):
            theta = np.arccos(
                v.dot(d) / (np.linalg.norm(v) * np.linalg.norm(d)))
            if theta > np.pi/2:
                theta = np.pi - theta
                reverse = -1
            else:
                reverse = 1
            # print(180 * theta / np.pi)
            if abs(theta) < best_theta:
                best_match = i
                best_theta = theta
                best_reverse = reverse
        # print(180 * best_theta / np.pi, directions[best_match] * reverse)
        print("Evec {} has angle {:7.3f} deg wrt direction [{: 4}, {: 4}, {: 4}]".format(
            ievec, 180 * best_theta / np.pi, *
            directions[best_match] * best_reverse
        ))




 # *****************************************
 #             Main program
 # *****************************************
if __name__ == "__main__":
    from optparse import OptionParser
    import sys


    parser = OptionParser()
    parser.add_option("-v", "--verbose", dest="verbose",
                    help="Print extra info", action="store_true")

    parser.add_option("-f", "--file",
                    default='cp2k.out',
                    help="Path to CP2K output file")


    parser.add_option("-i", "--ionic_step",
                    default=-1,
                    type="int",
                    help="Ionic step")

    parser.add_option("-a", "--atom_index",
                    default=-1,
                    type="int",
                    help="Atom index (python)")

    (options, args) = parser.parse_args()


    ionic_step = options.ionic_step
    atom_index = options.atom_index


    try:
        output = open(options.file, "r")
    except IOError:
        sys.stderr.write(
            "There was a problem opening the output file. Does it exist at all?")
        sys.exit(1)

    if output != None:
        outputfile = options.file
        natoms = get_natoms(outputfile)
        get_hf(outputfile, natoms=natoms,
            atom_index=atom_index, ionic_step=ionic_step)
