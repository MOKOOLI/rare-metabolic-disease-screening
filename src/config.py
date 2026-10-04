"""
Domain knowledge layer: metabolites, exact masses, pathways and disorder signatures.

All m/z values are monoisotopic [M+H]+ ions (positive ESI), computed from
elemental formulas. Reference medians are approximate dried-blood-spot (DBS)
concentrations in umol/L and are used ONLY to drive the simulator; they are not
clinical cut-offs.
"""
from dataclasses import dataclass

PROTON = 1.007276


@dataclass(frozen=True)
class Metabolite:
    key: str            # short id used as column name
    name: str
    formula: str
    mono_mass: float    # neutral monoisotopic mass (Da)
    rt: float           # expected retention time on a HILIC-like method (min)
    ref_median: float   # approx DBS median, umol/L
    ref_gsd: float      # geometric SD of the healthy distribution
    pathway: str

    @property
    def mz(self) -> float:
        return round(self.mono_mass + PROTON, 4)


METABOLITES = [
    # --- Amino acids ---
    Metabolite("Phe", "Phenylalanine", "C9H11NO2", 165.0790, 4.10, 55.0, 1.25, "phenylalanine_tyrosine"),
    Metabolite("Tyr", "Tyrosine", "C9H11NO3", 181.0739, 5.20, 75.0, 1.35, "phenylalanine_tyrosine"),
    Metabolite("Xle", "Leucine + Isoleucine", "C6H13NO2", 131.0946, 4.60, 125.0, 1.28, "bcaa"),
    Metabolite("aIle", "allo-Isoleucine", "C6H13NO2", 131.0946, 4.85, 1.0, 1.40, "bcaa"),
    Metabolite("Val", "Valine", "C5H11NO2", 117.0790, 5.60, 145.0, 1.28, "bcaa"),
    Metabolite("Met", "Methionine", "C5H11NO2S", 149.0510, 5.05, 22.0, 1.30, "sulfur_aa"),
    Metabolite("Cit", "Citrulline", "C6H13N3O3", 175.0957, 8.30, 16.0, 1.35, "urea_cycle"),
    Metabolite("Arg", "Arginine", "C6H14N4O2", 174.1117, 9.40, 12.0, 1.45, "urea_cycle"),
    Metabolite("Orn", "Ornithine", "C5H12N2O2", 132.0899, 9.10, 85.0, 1.35, "urea_cycle"),
    Metabolite("Gln", "Glutamine", "C5H10N2O3", 146.0691, 7.40, 520.0, 1.25, "urea_cycle"),
    # --- Tyrosine catabolism marker ---
    Metabolite("SA", "Succinylacetone", "C7H10O4", 158.0579, 2.30, 0.5, 1.50, "tyrosine_catabolism"),
    # --- Carnitine / acylcarnitines ---
    Metabolite("C0", "Free carnitine", "C7H15NO3", 161.1052, 6.80, 26.0, 1.35, "carnitine_shuttle"),
    Metabolite("C2", "Acetylcarnitine", "C9H17NO4", 203.1158, 6.10, 21.0, 1.35, "carnitine_shuttle"),
    Metabolite("C3", "Propionylcarnitine", "C10H19NO4", 217.1314, 5.70, 1.6, 1.40, "propionate_metabolism"),
    Metabolite("C5", "Isovalerylcarnitine", "C12H23NO4", 245.1627, 4.90, 0.15, 1.40, "leucine_catabolism"),
    Metabolite("C5DC", "Glutarylcarnitine", "C12H21NO6", 275.1369, 6.40, 0.05, 1.45, "lysine_catabolism"),
    Metabolite("C8", "Octanoylcarnitine", "C15H29NO4", 287.2097, 3.60, 0.05, 1.45, "fatty_acid_oxidation"),
    Metabolite("C10", "Decanoylcarnitine", "C17H33NO4", 315.2410, 3.20, 0.07, 1.45, "fatty_acid_oxidation"),
]
MET_BY_KEY = {m.key: m for m in METABOLITES}
METABOLITE_KEYS = [m.key for m in METABOLITES]

PATHWAYS = sorted({m.pathway for m in METABOLITES})

# Clinically used diagnostic ratios (numerator, denominator)
RATIOS = {
    "Phe/Tyr": ("Phe", "Tyr"),
    "Xle/Phe": ("Xle", "Phe"),
    "Cit/Arg": ("Cit", "Arg"),
    "C3/C2": ("C3", "C2"),
    "C8/C10": ("C8", "C10"),
    "C8/C2": ("C8", "C2"),
    "C5/C2": ("C5", "C2"),
    "C5DC/C8": ("C5DC", "C8"),
    "Met/Phe": ("Met", "Phe"),
}

# Disorder signatures: metabolite -> (min fold, max fold) relative to healthy median.
DISORDERS = {
    "PKU": {"label": "Phenylketonuria", "markers": {"Phe": (6, 25), "Tyr": (0.6, 0.9)}},
    "MSUD": {"label": "Maple syrup urine disease", "markers": {"Xle": (4, 15), "aIle": (15, 60), "Val": (2.5, 6)}},
    "TYR1": {"label": "Tyrosinemia type I", "markers": {"SA": (15, 120), "Tyr": (1.2, 3.0), "Met": (1.0, 2.5)}},
    "PA_MMA": {"label": "Propionic / methylmalonic acidemia", "markers": {"C3": (4, 12), "C2": (0.7, 1.0), "C0": (0.5, 0.9)}},
    "MCADD": {"label": "MCAD deficiency", "markers": {"C8": (8, 40), "C10": (1.5, 4.0), "C0": (0.5, 0.9)}},
    "CIT1": {"label": "Citrullinemia type I", "markers": {"Cit": (10, 50), "Arg": (0.3, 0.8), "Gln": (1.2, 2.0)}},
    "GA1": {"label": "Glutaric acidemia type I", "markers": {"C5DC": (5, 25)}},
    "IVA": {"label": "Isovaleric acidemia", "markers": {"C5": (6, 40)}},
}
DISORDER_KEYS = list(DISORDERS)

# Primary single-analyte marker per disorder (used for the classic cut-off baseline)
PRIMARY_MARKER = {"PKU": "Phe", "MSUD": "Xle", "TYR1": "SA", "PA_MMA": "C3",
                  "MCADD": "C8", "CIT1": "Cit", "GA1": "C5DC", "IVA": "C5"}

# Annotation tolerances
PPM_TOL = 5.0
RT_TOL_MIN = 0.20
