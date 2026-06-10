import pandas as pd
from mp_api.client import MPRester

import pandas as pd
from mp_api.client import MPRester
import unicodedata
import re


# CONFIG

API_KEY = "mVceo1qJ84xj9TJk2fl3az4YWXyXvPAu"
INPUT_FILE = "cleaned_osano.csv"
OUTPUT_FILE = "cleaned_osano.csv"


# STRONG SPACEGROUP NORMALIZER

def normalize_sg(symbol):
    if pd.isna(symbol):
        return ""

    # Normalize unicode (handles combined characters properly)
    symbol = unicodedata.normalize("NFKD", str(symbol))

    # Remove combining overbars
    symbol = symbol.replace("̅", "")

    # Convert subscripts to normal digits
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    symbol = symbol.translate(subscript_map)

    # Remove hyphens, underscores, and spaces
    symbol = re.sub(r"[-_\s]", "", symbol)

    return symbol.lower().strip()


# LOAD CSV

df = pd.read_csv(INPUT_FILE)

df["Compound"] = df["Compound"].astype(str).str.strip()
df["spacegroup"] = df["spacegroup"].astype(str).str.strip()
df["spacegroup_number"] = pd.to_numeric(df["spacegroup_number"], errors="coerce")

df["mp_material_id"] = None
df["mp_formula"] = None


# SEARCH AND MATCH

with MPRester(API_KEY) as mpr:

    for idx, row in df.iterrows():

        compound = row["Compound"]
        sg_number_csv = row["spacegroup_number"]
        sg_symbol_csv = row["spacegroup"]

        if pd.isna(sg_number_csv) or compound == "":
            continue

        sg_symbol_csv_norm = normalize_sg(sg_symbol_csv)

        print(f"\nSearching: {compound}")
        print(f"CSV SG#: {sg_number_csv} | CSV Symbol: {sg_symbol_csv}")

        try:
            results = mpr.materials.search(
                formula=compound,
                fields=["material_id", "formula_pretty", "symmetry"]
            )

            for doc in results:

                mp_sg_number = int(doc.symmetry.number)
                mp_sg_symbol = doc.symmetry.symbol
                mp_sg_symbol_norm = normalize_sg(mp_sg_symbol)

                print("Found:",
                      doc.material_id,
                      doc.formula_pretty,
                      mp_sg_symbol,
                      mp_sg_number)

                # Strict match: number + normalized symbol
                if (
                    mp_sg_number == int(sg_number_csv)
                    and mp_sg_symbol_norm == sg_symbol_csv_norm
                ):
                    print(" FULL MATCH FOUND")

                    df.at[idx, "mp_material_id"] = doc.material_id
                    df.at[idx, "mp_formula"] = doc.formula_pretty
                    break

        except Exception as e:
            print("Error:", e)

# SAVE OUTPUT

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFinished. Output saved as: {OUTPUT_FILE}")
