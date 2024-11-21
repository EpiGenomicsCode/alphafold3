
import numpy as np
from typing import Tuple, Dict, Optional
from alphafold3.common import residue_constants
import jax.numpy as jnp

def get_sequence_from_pdb(pdb_fn: str) -> str:
    """
    Extract the amino acid sequence from a PDB file.

    Args:
        pdb_fn (str): Path to the PDB file.

    Returns:
        str: Amino acid sequence.
    """
    to1letter = {
        "ALA": 'A', "ARG": 'R', "ASN": 'N', "ASP": 'D', "CYS": 'C',
        "GLN": 'Q', "GLU": 'E', "GLY": 'G', "HIS": 'H', "ILE": 'I',
        "LEU": 'L', "LYS": 'K', "MET": 'M', "PHE": 'F', "PRO": 'P',
        "SER": 'S', "THR": 'T', "TRP": 'W', "TYR": 'Y', "VAL": 'V'
    }

    sequence = []
    with open(pdb_fn, 'r') as fp:
        for line in fp:
            if line.startswith("ATOM"):
                if line[12:16].strip() != "CA":
                    continue
                res_name = line[17:20]
                if res_name in to1letter:
                    sequence.append(to1letter[res_name])
                else:
                    sequence.append('X')  # Unknown amino acid

    return ''.join(sequence)

def get_atom_positions_and_masks(pdb_fn: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract atom positions and masks from a PDB file.

    Args:
        pdb_fn (str): Path to the PDB file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - all_atom_positions: [num_residues, num_atoms_per_residue, 3]
            - all_atom_masks: [num_residues, num_atoms_per_residue]
    """
    atom_order = residue_constants.atom_order
    num_atoms = residue_constants.atom_type_num  # e.g., 14 for standard residues

    all_positions = []
    all_masks = []
    residue_positions = {}
    residue_masks = {}

    with open(pdb_fn, 'r') as fp:
        for line in fp:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_num = int(line[22:26])
            res_name = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

            if res_num not in residue_positions:
                residue_positions[res_num] = np.zeros((num_atoms, 3), dtype=np.float32)
                residue_masks[res_num] = np.zeros(num_atoms, dtype=np.float32)

            if atom_name in atom_order:
                atom_idx = atom_order[atom_name]
                residue_positions[res_num][atom_idx] = [x, y, z]
                residue_masks[res_num][atom_idx] = 1.0
            elif atom_name.upper() == 'SE' and res_name == 'MSE':
                # Map 'SE' to 'SD' for selenomethionine
                atom_idx = atom_order['SD']
                residue_positions[res_num][atom_idx] = [x, y, z]
                residue_masks[res_num][atom_idx] = 1.0

    # Sort residues by residue number
    sorted_res_nums = sorted(residue_positions.keys())
    for res_num in sorted_res_nums:
        all_positions.append(residue_positions[res_num])
        all_masks.append(residue_masks[res_num])

    all_atom_positions = np.array(all_positions)  # Shape: [num_residues, num_atoms, 3]
    all_atom_masks = np.array(all_masks)        # Shape: [num_residues, num_atoms]

    return all_atom_positions, all_atom_masks

def parse_pdb_to_predefined_positions(pdb_fn: str) -> jnp.ndarray:
    """
    Parse a PDB file and convert atom positions to JAX array.

    Args:
        pdb_fn (str): Path to the PDB file.

    Returns:
        jnp.ndarray: Predefined atom positions [1, num_residues, num_atoms, 3]
    """
    all_atom_positions, _ = get_atom_positions_and_masks(pdb_fn)
    predefined_positions = jnp.array(all_atom_positions)[np.newaxis, ...]  # Add batch dimension
    return predefined_positions