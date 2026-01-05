#!/usr/bin/env python3
"""
CLI tool to convert Extended XYZ files to MAGRES format.
Uses ASE to read atomic structures and export magnetic properties.
"""

import argparse
import sys
from pathlib import Path
from ase.io import read, write
import numpy as np


def prepare_atoms_for_magres(atoms, ms_tag=None, efg_tag=None, 
                               ms_units='ppm', efg_units='au', metadata=None):
    """
    Prepare atoms object for MAGRES export by renaming arrays and setting metadata.
    
    ASE's magres writer expects:
    - Magnetic shielding tensors in an array named 'ms' 
    - EFG tensors in an array named 'efg'
    - Units specified in atoms.info['magres_units'] dict
    - Calculation metadata in atoms.info['magresblock_calculation'] dict
    
    If arrays contain scalar values (isotropic components only), they will be
    converted to diagonal 3x3 tensors.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        The atomic structure with magnetic properties
    ms_tag : str, optional
        Name of the source array containing magnetic shielding tensors
    efg_tag : str, optional
        Name of the source array containing EFG tensors
    ms_units : str
        Units for magnetic shielding (default: 'ppm')
    efg_units : str
        Units for EFG tensors (default: 'au')
    metadata : dict, optional
        Custom metadata to add to the calculation block
    
    Returns:
    --------
    ase.Atoms : Modified atoms object ready for magres export
    """
    atoms_copy = atoms.copy()
    
    # Initialize magres_units dict if not present
    if 'magres_units' not in atoms_copy.info:
        atoms_copy.info['magres_units'] = {}
    
    # Initialize metadata dict if not present
    if metadata is None:
        metadata = {}
    
    # Handle magnetic shielding tensors
    if ms_tag and ms_tag in atoms.arrays:
        ms_array = atoms.arrays[ms_tag]
        
        # Check if we have scalar values (isotropic components only)
        if ms_array.ndim == 1 or (ms_array.ndim == 2 and ms_array.shape[1] == 1):
            # Convert scalar to diagonal tensor
            scalars = ms_array.flatten()
            ms_array = np.array([np.diag([s, s, s]) for s in scalars])
            if 'warning' not in metadata:
                metadata['warning'] = []
            metadata['warning'].append(['MS: Scalar values converted to diagonal tensors'])
            print(f"Note: Converting {len(scalars)} scalar shielding values to diagonal 3x3 tensors")
        # Ensure shielding tensors are in correct shape (N, 3, 3)
        # extxyz may store them as (N, 9) (flat)
        elif ms_array.ndim == 2 and ms_array.shape[1] == 9:
            ms_array = ms_array.reshape((-1, 3, 3))
        
        # Copy and rename to 'ms' if it's not already named that
        if ms_tag != 'ms':
            atoms_copy.new_array('ms', ms_array)
        else:
            atoms_copy.arrays['ms'] = ms_array
        atoms_copy.info['magres_units']['ms'] = ms_units
    
    # Handle EFG tensors
    if efg_tag and efg_tag in atoms.arrays:
        efg_array = atoms.arrays[efg_tag]
        
        # Check if we have scalar values (isotropic components only)
        if efg_array.ndim == 1 or (efg_array.ndim == 2 and efg_array.shape[1] == 1):
            # Convert scalar to diagonal tensor
            scalars = efg_array.flatten()
            efg_array = np.array([np.diag([s, s, s]) for s in scalars])
            if 'warning' not in metadata:
                metadata['warning'] = []
            metadata['warning'].append(['EFG: Scalar values converted to diagonal tensors'])
            print(f"Note: Converting {len(scalars)} scalar EFG values to diagonal 3x3 tensors")
        # Ensure EFG tensors are in correct shape (N, 3, 3)
        # extxyz may store them as (N, 9) (flat)
        elif efg_array.ndim == 2 and efg_array.shape[1] == 9:
            efg_array = efg_array.reshape((-1, 3, 3))
        
        # Copy and rename to 'efg' if it's not already named that
        if efg_tag != 'efg':
            atoms_copy.new_array('efg', efg_array)
        else:
            atoms_copy.arrays['efg'] = efg_array
        atoms_copy.info['magres_units']['efg'] = efg_units
    
    # Add metadata to calculation block
    if metadata:
        atoms_copy.info['magresblock_calculation'] = metadata
    
    return atoms_copy


def parse_metadata(metadata_strings):
    """
    Parse metadata strings in format 'key=value' into a dictionary compatible with magres format.
    
    For the magres calculation block, metadata should be formatted as:
    {key: [value]} where value is a list.
    
    Parameters:
    -----------
    metadata_strings : list of str
        List of 'key=value' strings
    
    Returns:
    --------
    dict : Parsed metadata in magres format
    """
    metadata = {}
    if metadata_strings:
        for item in metadata_strings:
            if '=' in item:
                key, value = item.split('=', 1)
                # Store as list of lists for magres format compatibility
                # Each inner list represents tokens on one line
                metadata[key.strip()] = [[value.strip()]]
            else:
                print(f"Warning: Ignoring invalid metadata format: {item}", file=sys.stderr)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert Extended XYZ file to MAGRES format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (first structure by default)
  %(prog)s input.extxyz
  
  # Specify output file
  %(prog)s input.extxyz -o output.magres
  
  # Convert with shielding data from custom array
  %(prog)s input.extxyz --ms-tag dft_ms
  
  # Convert with both shielding and EFG data
  %(prog)s input.extxyz --ms-tag ms --efg-tag efg
  
  # Specify custom units
  %(prog)s input.extxyz --ms-tag sigma --ms-units ppm
  
  # Add custom metadata to calculation block
  %(prog)s input.extxyz --metadata calc_code=CASTEP calc_code_version=20.1
  
  # Add warning metadata
  %(prog)s input.extxyz --ms-tag ms --metadata warning="Check convergence"
  
  # Convert specific structure by index
  %(prog)s input.extxyz --ms-tag ms --index 5
  
  # Convert last structure
  %(prog)s input.extxyz --ms-tag ms --index -1
  
  # Convert all structures (creates multiple output files)
  %(prog)s input.extxyz --ms-tag ms --index :
  
  # Convert range of structures
  %(prog)s input.extxyz --ms-tag ms --index 0:10
  
  # Convert every other structure
  %(prog)s input.extxyz --ms-tag ms --index ::2

Notes:
  - At least one of --ms-tag or --efg-tag must be specified
  - Structures without the requested data will be skipped
  - Scalar values are automatically converted to diagonal 3x3 tensors
  - Multiple structures get indexed filenames (e.g., output_000.magres)
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input Extended XYZ file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output MAGRES file (default: input file with .magres extension)'
    )
    
    parser.add_argument(
        '--ms-tag',
        type=str,
        default=None,
        help='Name of the array containing magnetic shielding tensors (e.g., "ms", "dft_ms", "shielding")'
    )
    
    parser.add_argument(
        '--efg-tag',
        type=str,
        default=None,
        help='Name of the array containing EFG tensors (e.g., "efg", "efg_tensor")'
    )
    
    parser.add_argument(
        '--ms-units',
        type=str,
        default='ppm',
        help='Units for magnetic shielding tensors (default: ppm)'
    )
    
    parser.add_argument(
        '--efg-units',
        type=str,
        default='au',
        help='Units for EFG tensors (default: au)'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        nargs='+',
        help='Custom metadata for calculation block in format key=value (can specify multiple)'
    )
    
    parser.add_argument(
        '--index',
        type=str,
        default='0',
        help='Index or slice of structures to convert (e.g., "0", ":", "-1", "0:5"). Default: "0"'
    )
    
    args = parser.parse_args()
    
    # Determine base output filename
    if args.output is None:
        input_path = Path(args.input)
        output_base = input_path.with_suffix('.magres')
    else:
        output_base = Path(args.output)
    
    # Read the Extended XYZ file with index
    try:
        atoms_list = read(args.input, index=args.index)
        # Ensure we have a list for consistent handling
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]
            indices = [int(args.index) if args.index.lstrip('-').isdigit() else 0]
        else:
            # Determine the actual indices from the slice
            # First, read all to get total count
            all_atoms = read(args.input, index=':')
            total_count = len(all_atoms)
            # Parse the slice to get actual indices
            if args.index == ':':
                indices = list(range(total_count))
            elif ':' in args.index:
                # Handle slice notation - convert "0:3" to slice(0, 3)
                parts = args.index.split(':')
                start = int(parts[0]) if parts[0] else None
                stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                slice_obj = slice(start, stop, step)
                indices = list(range(*slice_obj.indices(total_count)))
            else:
                # Single negative index
                idx = int(args.index)
                indices = [idx if idx >= 0 else total_count + idx]
        
        print(f"Read {len(atoms_list)} structure(s) from {args.input}")
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine zero-padding for indices
    if len(atoms_list) > 1:
        # Get the maximum index to determine padding
        total_structures = read(args.input, index=':')
        max_index = len(total_structures) - 1
        zero_pad = len(str(max_index))
    else:
        zero_pad = 0
    
    # Process each structure
    for struct_idx, (atoms, orig_idx) in enumerate(zip(atoms_list, indices)):
        # Parse metadata (fresh copy for each structure)
        metadata = parse_metadata(args.metadata)
        
        # Add conversion source information
        metadata['converted_from'] = [[args.input]]
        # Always include structure_index when index is explicitly specified
        if args.index != '0' or len(atoms_list) > 1:
            metadata['structure_index'] = [[str(orig_idx)]]
        
        # Check if requested tags exist
        has_ms = args.ms_tag and args.ms_tag in atoms.arrays
        has_efg = args.efg_tag and args.efg_tag in atoms.arrays
        
        if args.ms_tag and not has_ms:
            print(f"Warning: MS tag '{args.ms_tag}' not found in arrays.", file=sys.stderr)
            print(f"Available arrays: {list(atoms.arrays.keys())}", file=sys.stderr)
        
        if args.efg_tag and not has_efg:
            print(f"Warning: EFG tag '{args.efg_tag}' not found in arrays.", file=sys.stderr)
            print(f"Available arrays: {list(atoms.arrays.keys())}", file=sys.stderr)
        
        # Skip if no magnetic data is present
        if not has_ms and not has_efg:
            print(f"Skipping structure {orig_idx}: No MS or EFG data found", file=sys.stderr)
            continue
        
        # Prepare atoms for magres export
        try:
            atoms_prepared = prepare_atoms_for_magres(
                atoms, 
                ms_tag=args.ms_tag,
                efg_tag=args.efg_tag,
                ms_units=args.ms_units,
                efg_units=args.efg_units,
                metadata=metadata
            )
        except Exception as e:
            print(f"Error preparing atoms for magres format: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Determine output filename with index if multiple structures
        if len(atoms_list) > 1:
            stem = output_base.stem
            suffix = output_base.suffix
            parent = output_base.parent
            idx_str = str(orig_idx).zfill(zero_pad)
            output_path = parent / f"{stem}_{idx_str}{suffix}"
        else:
            output_path = output_base
        
        # Write the MAGRES file using ASE's writer
        try:
            write(output_path, atoms_prepared, format='magres')
            print(f"Successfully wrote MAGRES file to {output_path}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
