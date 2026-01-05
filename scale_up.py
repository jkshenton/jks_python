'''
Created on 2022-05-26

Often I want to start defect calculations using a small-ish supercell
but then scale this up to check convergence / production.

This script takes in a supercell structure, a primitive structure and the mapping
between the supercell and the new supercell size.
For example, let's say we initially used a 2x2x2 supercell but now we want to
scale up to a 4x4x4 supercell.


TODO: break up the center_on_defect code into distance finder and centering
use the distance finder to make a new function to find translation vector that best aligns the 
bulk bits of the structure. 
ie we'll run this reordering thing 3 times! -- maybe be more efficient about that

FIX translation and drift calculations

@author: jkshenton
'''
# use click to get the arguments

import click
from ase.io import read, write
import sys
import numpy as np
# import default dict
from collections import defaultdict
from ase.geometry import find_mic


def reorder_atoms(atoms_ref, atoms_to_reorder, check_species=True, tol=0.5):
    """
    Takes in a reference ASE atoms object and another ASE atoms object to reorder
    
    Loops over reference atoms and, for each one, finds it's closest counterpart in atoms_to_reorder
    (taking into account the periodic boundaries via the minimum image convention).
    
    It finally reorders atoms_to_reorder such that they follow the order in the refernce one. 

    Parameters
    ----------
    atoms_ref : ASE atoms object
        The reference structure
    atoms_to_reorder : ASE atoms object
        The structure to be reordered
    check_species : bool
        If True, will check that the species of the atoms are the same
        Default is True
    tol : float (Angstroms)
        The tolerance for the minimum image convention distance between possible matches.
        Default: 10 Angstroms (i.e. basically anything goes!)
    
    Returns
    -------
    mapping : defaultdict (i, j) mapping index i in atoms_ref to j in atoms_to_reorder


    """
    atoms_to_reorder_new_indices = []
    mapping = defaultdict(list)
    # loop over reference atoms
    for i, atom in enumerate(atoms_ref):

        # temporary copy of atoms_to_reorder
        temp = atoms_to_reorder.copy()
        
        # add atom to temp:
        temp.append(atom)

        # which atom is closest to the newly added atom in atoms_to_reorder
        distances = temp.get_distances(-1, range(len(atoms_to_reorder)), mic=True)
        
        # # filter those outside the tolerance:
        # distances = distances[distances < tol]

        order = np.argsort(distances)
        
        if check_species:
            # loop over closest atoms to atom (in order of closeness)
            for closest_index in order:
                if atom.symbol == atoms_to_reorder[closest_index].symbol:
                    # if the atom closest is of the same species, then
                    # we're done, otherwise keep going until you find 
                    # one that does match in species
                    atoms_to_reorder_index = closest_index
                    break
                else:
                    # if the atom closest is of a different species, then
                    # there isn't an equivalent atom in ref:
                    # in_ref_only.append(i)
                    pass
            
        else:
            # get index of closest atom
            atoms_to_reorder_index = order[0]

        if distances[atoms_to_reorder_index] < tol:
            # append to array of new order
            mapping[i] = atoms_to_reorder_index
            atoms_to_reorder_new_indices.append(atoms_to_reorder_index)
        else:
            # this atom is not in ref, so we don't need to do anything
            pass

    return mapping


def scale_up(primitive_atoms, initial_supercell, newsupercell, check_species=True, tol=0.5):
    '''
    Scales up a structure by a given mapping.


    TODO: Remove any net translation between prim and distorted supercell
    
    Parameters
    ----------
    primitive_atoms : ASE atoms object
        The reference structure "== (1,1,1) supercell"
    initial_supercell : ASE atoms object
        The structure to be scaled up
    newsupercell : list/tuple/array
        e.g. [3,3,3]
    check_species : bool
        If True, will check that the species of the atoms are the same
        Default is True
    tol : float (Angstroms)
        The tolerance for the minimum image convention distance between possible matches.
        Default: 0.5 Angstroms
    
    Returns
    -------
    The scaled up atoms  : ASE atoms object
    '''
    
    initial_supercell_size = initial_supercell.cell.cellpar()[:3] / primitive_atoms.cell.cellpar()[:3]
    initial_supercell_size = np.round(initial_supercell_size).astype(int)


    # primitive_supercell_initial = primitive_atoms * initial_supercell_size
    primitive_supercell_new     = primitive_atoms * newsupercell


    new_supercell = initial_supercell.copy()
    # now let's scale up the box, keeping the initial supercell atoms in the bottom corner
    new_supercell.set_cell(primitive_supercell_new.cell, scale_atoms=False)


    # reorder new_supercell
    initial_supercell = reorder_atoms(primitive_supercell_new, new_supercell, check_species=check_species, tol=tol)

    # scale up
    mapping = reorder_atoms(primitive_supercell_new, 
                                        new_supercell,
                                        check_species=check_species,
                                        tol=tol)
    atoms_to_reorder_new_indicies = list(mapping.values())
    not_in_ref = list(set(range(len(new_supercell))) - set(atoms_to_reorder_new_indicies))

    atoms = primitive_supercell_new.copy()

    for i, atom in enumerate(atoms):
        if mapping[i]:
            atoms[i].position = new_supercell[mapping[i]].position
            atoms[i].charge = new_supercell[mapping[i]].charge
            atoms[i].magmom = new_supercell[mapping[i]].magmom
            atoms[i].tag = 1


    # add in atoms that weren't in the ref (e.g. interstitial defects):
    for i in not_in_ref:
        atoms.append(new_supercell[i])
        atoms[-1].tag = 2
        
    return atoms
def get_drift(distorted_structure, primitive, fraction=0.9):
    '''
    Finds the translation vector to minimize the drift between the distorted and primitive structures.
    It considers the mean displacements of the atoms that move the least. 
    
    Parameters
    ----------
    distorted_structure : ASE atoms object
        The distorted structure
    primitive : ASE atoms object
        The primitive structure
    fraction : float
        The fraction of the atoms to consider when calculating the drift. i.e. the default of 0.9 means 
        that the drift is calculated from excluding the 10% of atoms that move the most... 
    
    Returns
    -------
    translation_vector : array
        The translation vector to center the distorted structure on the distortion.
    '''
    cell, distorted_structure_new, displacements, distances = get_matched_displacements(distorted_structure, primitive)
    number_to_include = int(len(displacements) * fraction)
    print(f'Including {number_to_include} atoms in the drift calculation')
    least_displaced_atoms = np.argsort(distances)[:number_to_include]
    # mean displacement of the least displaced atoms
    mean_displacement = np.mean(displacements[least_displaced_atoms], axis=0)

    return mean_displacement



def center_on_defect(distorted_structure, primitive, fraction):
    '''
    Finds the translation vector to center the distorted structure on the distortion
    with respect to primitive.
    
    Parameters
    ----------
    distorted_structure : ASE atoms object
        The distorted structure
    primitive : ASE atoms object
        The primitive structure
    fraction : float
        The fraction of the atoms to consider when calculating the defect center of mass.
    
    Returns
    -------
    translation_vector : array
        The translation vector to center the distorted structure on the distortion.
    '''
    cell, distorted_structure_new, displacements, distances = get_matched_displacements(distorted_structure, primitive)
    # get max 5% distances:
    number_to_include = int(len(displacements) * fraction)
    print(f'Using the {number_to_include} most distorted atoms to calculate the centre of the defect area.')

    most_displaced_atoms = np.argsort(distances)[-number_to_include:]

    # center of mass of most displaced atoms:
    # center_of_mass = np.median(distorted_structure_new.get_positions()[most_displaced_atoms], axis=0)
    center_of_mass = distorted_structure_new[most_displaced_atoms].get_center_of_mass()
    # translate center of mass to cell center:
    f = [0.5, 0.5, 0.5]
    # f = 1/ relative_supercell_scale
    print(f'The centre of the defect area is: {center_of_mass}')
    translation_vector = cell.T.dot(f) - center_of_mass
    print(f'The translation vector to move the defect to the centre is: {translation_vector} (A)')

    return translation_vector

def get_matched_displacements(distorted_structure, primitive, tol=50):
    # find how much to scale the primitive structure up to get the distorted structure
    scale_factor = distorted_structure.cell.cellpar()[:3] / primitive.cell.cellpar()[:3]
    scale_factor = np.round(scale_factor).astype(int)
    primitive_sc = primitive * scale_factor

    cell = primitive_sc.cell

    # find the mapping between the distorted and primitive structures
    mapping = reorder_atoms(primitive_sc, distorted_structure, tol=tol)

    # reorder the distorted structure
    distorted_structure_new = primitive_sc.copy()
    for i, atom in enumerate(distorted_structure_new):
        if mapping[i]:
            distorted_structure_new[i].position = distorted_structure[mapping[i]].position
            distorted_structure_new[i].tag = 1
    
    # get displacement from one to the other
    displacements = distorted_structure_new.get_positions() - primitive_sc.get_positions()

    # get minimum image convention displacements:
    displacements, distances = find_mic(displacements, cell)
    return cell,distorted_structure_new,displacements,distances

    




@click.command()
@click.argument('primitive', type=click.Path(exists=True))
@click.argument('distorted_structure', type=click.Path(exists=True))
@click.argument('supercell', nargs=3, type=click.INT)
@click.option('--center', '-c', type=bool, default=True, help='If True, will attempt to centre defect within its cell before scaling up. Set e.g. "-c False" to disable.')
@click.option('--centerfrac', '-cf', type=float, default=0.1, help='Fraction of atoms to use when centering the defect (based on the amount of displacement from the pristine structure). Default is 0.1.')
@click.option('--check_species', type=bool, default=True, help='If True, will check that the species of the atoms are the same')
@click.option('--tol', '-t', type=float, default=0.5, help='The tolerance for the minimum image convention distance between possible matches')
@click.option('--output_file', '-o', type=click.Path(exists=False), default=None, help='The name of the output file. Can be any ASE supported format.')
# add option flag to view the structure
@click.option('--view', '-v', is_flag=True, default=False, help='If True, will view the structure')

def main(primitive, distorted_structure, supercell, center, centerfrac, check_species, tol, output_file, view):
    '''
    Scales up a structure by a given mapping.

    For example, if you relaxed an interstitial defect in a 2x2x2 supercell and want to scale it up to a 4x4x4 supercell,
    you might do:
    
    scale_up.py primitive.cif distorted_sc222.cif 4 4 4 --view

    Note that the final structure in the ASE gui is the one scaled up one. Using the ASE gui, select colour by tag to see the
    region of the distorted structure that is not in the pristine structure. 
    
    To output to file:
    
    scale_up.py primitive.cif distorted_sc222.cif 4 4 4 -o distorted_sc444.cif


    We try to center the defective part of the structure first and then 'pad' the structure with the pristine part.
    To prevent or tweak this centering see the --center and --centerfrac options. 

    
    Parameters\n
    ----------\n
    primitive : filepath
        The reference structure "== (1,1,1) supercell"

    distorted_structure : filepath
        The structure to be scaled up

    supercell : list/tuple/array
        e.g. [3,3,3]

    center : bool
        If True, will attempt to centre defect within its cell before scaling up.
        This is useful if the defect is not centered in the middle of the cell to start with. It might not always work though, 
        so you can always disable it if you want to.

    centerfrac : float
        Fraction of atoms to use when centering the defect (based on the amount of displacement from the pristine structure).
        Default is 0.1.

    check_species : bool
        If True, will check that the species of the atoms are the same
        Default is True

    tol : float (Angstroms)
        The tolerance for the minimum image convention distance between possible matches.
        Default: 0.5 Angstroms

    output_file : filepath
        The name of the output file. Can be any ASE supported format.
    
    Returns\n
    -------\n    
    The scaled up atoms  : ASE atoms object
    '''

    images = []
    # read in the input file
    primitive_atoms = read(primitive)
    primitive_atoms.calc = None # this sometimes confuses things...
    distorted_atoms = read(distorted_structure)
    distorted_atoms.calc = None # this sometimes confuses things...
    images.append(distorted_atoms.copy())
    # what is the distorted strcutrue's supercell relative to the primitive one?
    original_supercell = distorted_atoms.cell.cellpar()[:3] / primitive_atoms.cell.cellpar()[:3]
    original_supercell = np.round(original_supercell).astype(int)
    
    nprisitine = np.prod(original_supercell) * len(primitive_atoms)
    delta_atoms = len(distorted_atoms) - nprisitine
    print(f'The distorted structure has {delta_atoms} different number of atoms wrt primitive structure.')


    expected_final_number_of_atoms = np.prod(supercell) * len(primitive_atoms) + delta_atoms
    


    if center:
        # center the main distortion in the cell
        relative_supercell_scale = np.array(supercell) / original_supercell
        v = center_on_defect(distorted_atoms, primitive_atoms, fraction=centerfrac)
        distorted_atoms.translate(v)
        images.append(distorted_atoms.copy())

        distorted_atoms.wrap()
        images.append(distorted_atoms.copy())
        
        # remove net drift from the distorted structure
        drift = get_drift(distorted_atoms, primitive_atoms, fraction=1-centerfrac)
        distorted_atoms.translate(-drift)
        images.append(distorted_atoms.copy())
        
        print(f'drift: {drift}')

        # wrap structure
        distorted_atoms.wrap()
        images.append(distorted_atoms.copy())

    

    # scale up
    atoms = scale_up(primitive_atoms, distorted_atoms, supercell, check_species=check_species, tol=tol)
    final_number_of_atoms = len(atoms)
    if expected_final_number_of_atoms != final_number_of_atoms:
        # raise error
        raise ValueError(f'The final number of atoms is {final_number_of_atoms} but it should be {expected_final_number_of_atoms}\n'
        'Try changing the centerfrac parameter or setting "--center False". You can also try changing the tolerance (-t).')
    images.append(primitive_atoms * supercell)
    images.append(atoms.copy())

    # write out
    if output_file:
        write(output_file, atoms)

    # View the structure using the ASE gui?
    if view:
        import ase.visualize as viz
        viz.view(images)





# main function
if __name__ == '__main__':   
    sys.exit(main()) 