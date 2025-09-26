from __future__ import annotations
import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem


def _iter_smiles(csv_path: Path) -> Iterable[Tuple[str, str, str]]:
    """Yield (index, num_c, smiles) rows from a CSV with header."""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["Index"], row["Num_c"], row["SMILES"].strip()


def _build_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    """Build a 3D molecule from SMILES with ETKDG + MMFF/UFF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Try ETKDGv3, then ETKDGv2 as fallback.
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params2 = AllChem.ETKDGv2()
        params2.randomSeed = 0xF00D
        status = AllChem.EmbedMolecule(mol, params2)
        if status != 0:
            return None

    # Optimize geometry: prefer MMFF94, fallback to UFF when parameters missing.
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            return None

    return mol


def _write_single_sdf(mol: Chem.Mol, output_path: Path) -> bool:
    """Write a single molecule to an SDF file."""
    writer = Chem.SDWriter(str(output_path))
    try:
        writer.write(mol)
        return True
    finally:
        writer.close()


def _process_molecule_and_save(idx: str, num_c: str, smi: str, output_dir: Path) -> bool:
    """Process a single molecule and save it as a separate SDF file."""
    mol = _build_3d_mol(smi)
    if mol is None:
        return False

    # Set name and properties. Materials Studio reads SDF data fields.
    mol.SetProp("_Name", f"R4N_{idx}_C{num_c}")
    mol.SetProp("Index", str(idx))
    mol.SetProp("Num_c", str(num_c))
    mol.SetProp("SMILES", smi)

    # Define output path for this molecule
    output_file = output_dir / f"R4N_{idx}_C{num_c}.sdf"

    return _write_single_sdf(mol, output_file)


def main() -> None:
    """Entry point: convert CSV SMILES to 3D SDF files."""
    root = Path(__file__).resolve().parent
    csv_path = root / "data" / "r4n_smiles_c20.csv"
    output_dir = root / "data" / "3d_molecules"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 3D molecules from SMILES (RDKit)...")
    print(f"Input:  {csv_path}")
    print(f"Output Directory: {output_dir}")

    rows = list(_iter_smiles(csv_path))
    print(f"Total molecules to process: {len(rows)}")

    successful_count = 0
    failed_count = 0

    # Process each row and save to individual files
    for idx, num_c, smi in rows:
        success = _process_molecule_and_save(idx, num_c, smi, output_dir)
        if success:
            successful_count += 1
        else:
            failed_count += 1

    print("Done.")
    print(f"Successfully written: {successful_count}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
