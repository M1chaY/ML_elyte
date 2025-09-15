"""Mock RDKit module for testing when RDKit is not available."""

import warnings

# Mock RDKit classes and functions
class MockMol:
    def __init__(self):
        pass

class MockChem:
    Mol = MockMol  # Add Mol as a class attribute
    
    @staticmethod
    def MolFromSmiles(smiles):
        if smiles and isinstance(smiles, str):
            return MockMol()
        return None
    
    @staticmethod
    def AddHs(mol):
        return mol
    
    @staticmethod
    def RemoveHs(mol):
        return mol
    
    @staticmethod
    def SanitizeMol(mol):
        pass
    
    @staticmethod
    def MolToSmiles(mol, canonical=True):
        return "CCO"  # Dummy SMILES

class MockDescriptors:
    @staticmethod
    def MolWt(mol):
        return 46.07  # Ethanol molecular weight
    
    @staticmethod
    def MolLogP(mol):
        return -0.31
    
    @staticmethod
    def NumHDonors(mol):
        return 1
    
    @staticmethod
    def NumHAcceptors(mol):
        return 1
    
    @staticmethod
    def NumRotatableBonds(mol):
        return 1
    
    @staticmethod
    def TPSA(mol):
        return 20.23
    
    @staticmethod
    def NumAromaticRings(mol):
        return 0
    
    @staticmethod
    def NumSaturatedRings(mol):
        return 0
    
    @staticmethod
    def NumHeavyAtoms(mol):
        return 3
    
    @staticmethod
    def NumValenceElectrons(mol):
        return 20
    
    @staticmethod
    def BalabanJ(mol):
        return 0.0
    
    @staticmethod
    def BertzCT(mol):
        return 0.0
    
    @staticmethod
    def Chi0n(mol):
        return 0.0
    
    @staticmethod
    def HallKierAlpha(mol):
        return 0.0
    
    @staticmethod
    def Kappa1(mol):
        return 0.0
    
    @staticmethod
    def LabuteASA(mol):
        return 0.0
    
    @staticmethod
    def PEOE_VSA1(mol):
        return 0.0
    
    @staticmethod
    def SMR_VSA1(mol):
        return 0.0
    
    @staticmethod
    def SlogP_VSA1(mol):
        return 0.0
    
    @staticmethod
    def VSA_EState1(mol):
        return 0.0
    
    @staticmethod
    def FractionCsp3(mol):
        return 1.0
    
    @staticmethod
    def NumAliphaticCarbocycles(mol):
        return 0
    
    @staticmethod
    def NumAliphaticHeterocycles(mol):
        return 0
    
    @staticmethod
    def RingCount(mol):
        return 0
    
    @staticmethod
    def Ipc(mol):
        return 0.0

class MockCrippen:
    @staticmethod
    def CrippenClogPAndMR(mol):
        return (-0.31, 1.0)  # logP, MR

class MockMolDescriptors:
    @staticmethod
    def GetMorganFingerprintAsBitVect(mol, radius, nBits=1024):
        return MockBitVect(nBits)
    
    @staticmethod
    def CalcMolFormula(mol):
        return "C2H6O"

class MockBitVect:
    def __init__(self, size):
        self.size = size
    
    def ToBitString(self):
        return '0' * self.size

# Create mock module structure
Chem = MockChem()
Descriptors = MockDescriptors() 
Crippen = MockCrippen()
rdMolDescriptors = MockMolDescriptors()

print("Using mock RDKit module (RDKit not installed)")