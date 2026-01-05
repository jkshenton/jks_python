#!/usr/bin/env python
'''
A command line script using ASE povray to generate nice looking
atoms images.

Uses click for handling command line arguments.

'''
import click
from ase.io import read, write
from ase.io.pov import get_bondpairs, pc
from ase.data.colors import jmol_colors
from ase.data import atomic_numbers

import numpy as np


@click.command(context_settings={'show_default': True})
@click.argument('inputfile', type=click.Path(exists=True))
@click.option('--index', '-i', type=str, default=-1, help='Index of the image to be rendered. Use -1 for the last image; use ":" for all images.')
@click.option('--prefix', '-p', type=str, default='niceatoms')
@click.option('--width', '-w', type=int, default=800)
@click.option('--background', '-b', type=click.Choice(['White', 'Black']), default='White')
@click.option('--transparent', '-t', is_flag=True)
@click.option('--dont-render', is_flag=True)
@click.option('--rotation', '-r', type=str, default="0x,0y,0z", help='Rotation of the image. Get this from the ASE GUI.')
@click.option('--texture', type=click.Choice([
                                    'simple',
                                    'pale',
                                    'intermediate',
                                    'vmd',
                                    'jmol',
                                    'ase2',
                                    'ase3',
                                    'glass',
                                    'glass2',
                                    ]),
                            default='ase3')
@click.option('--radii', type=float, default=0.7)
@click.option('--camera-dist', type=float, default=50)
@click.option('--depth-cue', is_flag=True, default=False)
@click.option('--camera-type', type=click.Choice(['perspective', 'orthographic']), default='orthographic')
@click.option('--hide-bonds', is_flag=True)
@click.option('--bond-width', type=float, default=0.165)
@click.option('--vdw-scale', type=float, default=1.1, help='Scale the van der Waals radii by this factor when calculating bonds.')
@click.option('--celllinewidth', '--cw', type=float, default=0.015)
# key, value pairs of colors for each element
@click.option('--colors', type=str, default='', help='Comma separated list of colors for each element. E.g. "C:blue,O:red". Colors can be in any format that povray accepts, e.g. "rgb 0 0 1" or "Blue".')
def niceatoms(
    inputfile,
    index,
    prefix,
    width,
    background,
    transparent,
    dont_render,
    rotation,
    texture,
    radii,
    camera_dist,
    depth_cue,
    camera_type,
    hide_bonds,
    bond_width,
    vdw_scale,
    celllinewidth,
    colors,
    ):
    '''A command line script using ASE povray to generate nice looking
    atoms images.

    Uses click for handling command line arguments.

    '''
    # if index can be converted to a single int, assume it's a single image
    # if index can be converted to a slice, assume it's a slice
    try :
        index = int(index)
    except ValueError:
        pass

    # convert colors to a dictionary
    colors = dict([c.split(':') for c in colors.split(',') if c])
    # remove any whitespace from the keys
    colors = {k.strip():v for k,v in colors.items()}
    # convert all colors to rgb
    for k, v in colors.items():
        v_array = v.split()
        if len(v_array) == 1:
            # assume it's just a color name
            # pass
            continue
        elif len(v_array) == 3:
            # assume it's rgb
            colors[k] = np.asarray(v_array, dtype=float)
        else:
            raise ValueError(f'Could not parse color {v}')
    




    if isinstance(index, int):
        images = [read(inputfile, index=index)]
    else:
        images = read(inputfile, index=index)

    for iat, atoms in enumerate(images):
        if len(images) > 1:
            fname = f'{prefix}-{iat:04d}.pov'
        else:
            fname = f'{prefix}.pov'

        elements = set(atoms.get_chemical_symbols())
        # For any elements that are not in the colors dictionary, use the jmol colors
        default_elements = elements.difference(colors.keys())
        colors.update({e:jmol_colors[atomic_numbers[e]] for e in default_elements})


        textures = [texture for i in range(len(atoms))]
        colors_list = [colors.get(atoms[i].symbol) for i in range(len(atoms))]
        # convert to povray colors
        # colors_list = [pc(c) for c in colors_list]
        print(colors_list)
        # Define the atomic bonds to show
        bondpairs = []
        if not hide_bonds:
            bondpairs = get_bondpairs(atoms, radius=vdw_scale)
        # Create nice-looking image using povray
        renderer = write(
            fname,
            atoms,
            rotation=rotation,
            radii=radii,
            povray_settings=dict(transparent=transparent,
                                 textures=textures,
                                 camera_dist=camera_dist,
                                 depth_cueing=depth_cue,
                                 cue_density=7e-2,
                                 camera_type=camera_type,
                                 canvas_width=width,
                                 bondlinewidth=bond_width,
                                 bondatoms=bondpairs,
                                 background=background,
                                 celllinewidth=celllinewidth,
                                 colors = colors_list,
                                 ))
        if not dont_render:
            renderer.render()
            print(f'Rendered image {iat}')


if __name__ == '__main__':
    niceatoms()
