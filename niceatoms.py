#!/usr/bin/env python
'''
A command line script using ASE povray to generate nice looking
atoms images.

Uses click for handling command line arguments.

'''
import click
from ase.io import read, write
from ase.io.pov import get_bondpairs
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



    if isinstance(index, int):
        images = [read(inputfile, index=index)]
    else:
        images = read(inputfile, index=index)

    for iat, atoms in enumerate(images):
        if len(images) > 1:
            fname = f'{prefix}-{iat:04d}.pov'
        else:
            fname = f'{prefix}.pov'

        textures = [texture for i in range(len(atoms))]
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
                                 ))
        if not dont_render:
            renderer.render()
            print(f'Rendered image {iat}')


if __name__ == '__main__':
    niceatoms()
