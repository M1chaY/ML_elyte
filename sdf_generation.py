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
    # 从SMILES生成3D分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 加入氢原子
    mol = Chem.AddHs(mol)

    # 生成3D构象
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    AllChem.EmbedMolecule(mol, params)

    embed_result = AllChem.EmbedMolecule(mol, params)
    if embed_result == -1:
        # If ETKDG fails, try basic embedding
        embed_result = AllChem.EmbedMolecule(mol)
        if embed_result == -1:
            print(f"Warning: Could not generate 3D coordinates for SMILES: {smiles}")
            return None

    # Try MMFF optimization first
    try:
        mmff_result = AllChem.MMFFOptimizeMolecule(mol)
        if mmff_result == 1:  # 1 indicates failure, 0 indicates success
            # If MMFF fails, try UFF as fallback
            AllChem.UFFOptimizeMolecule(mol)
    except ValueError as e:
        # If MMFF completely fails, try UFF
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except ValueError:
            print(f"Warning: Could not optimize molecule for SMILES: {smiles}")
            # Return the molecule anyway - it has 3D coordinates even if not optimized
            pass

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
    mol.SetProp("_Name", f"{idx}_{smi}")
    mol.SetProp("Index", str(idx))
    mol.SetProp("Num_c", str(num_c))
    mol.SetProp("SMILES", smi)

    # Define output path for this molecule
    output_file = output_dir / f"{idx}_{smi}.sdf"

    return _write_single_sdf(mol, output_file)


def main() -> None:
    """Entry point: convert CSV SMILES to 3D SDF files."""
    root = Path(__file__).resolve().parent
    csv_path = root / "data" / "r4n_smiles_c20.csv"
    output_dir = root / "data" / "sdf"

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
