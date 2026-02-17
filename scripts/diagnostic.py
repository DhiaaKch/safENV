"""
quick_http_check.py - Diagnostic rapide http.csv
"""

import pandas as pd
from pathlib import Path

fp = Path("data/00_raw/r4.2/http.csv")

print("🔍 DIAGNOSTIC HTTP.CSV")
print("=" * 60)

if not fp.exists():
    print(f"❌ Fichier introuvable: {fp}")
    exit(1)

print(f"✅ Fichier: {fp}")
print(f"   Taille: {fp.stat().st_size / 1024 / 1024:.1f} MB\n")

# Lire 20 premières lignes
df = pd.read_csv(fp, nrows=20, low_memory=False)

print("📋 COLONNES:")
print(df.columns.tolist())
print()

print("📅 COLONNES DATE/TIME:")
date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time'])]
print(f"   Trouvées: {date_cols}")
print()

if date_cols:
    for col in date_cols:
        print(f"📊 Colonne '{col}':")
        print(f"   Exemples:")
        for i, val in enumerate(df[col].head(5), 1):
            print(f"      {i}. {val}")
        print()
        
        # Test formats
        formats = [
            "%m/%d/%Y %H:%M:%S",     # CERT standard
            "%Y-%m-%d %H:%M:%S",     # ISO
            "%m/%d/%Y %H:%M",        # Sans secondes
            "%m-%d-%Y %H:%M:%S",     # Tirets
            "%d/%m/%Y %H:%M:%S",     # Jour d'abord
        ]
        
        for fmt in formats:
            try:
                parsed = pd.to_datetime(df[col].head(), format=fmt)
                print(f"   ✅ FORMAT DÉTECTÉ: {fmt}")
                print(f"      Min: {parsed.min()}")
                print(f"      Max: {parsed.max()}")
                break
            except:
                continue
        else:
            print(f"   ⚠️  Format non-standard, essai auto-détection...")
            try:
                parsed = pd.to_datetime(df[col].head(), infer_datetime_format=True)
                print(f"   ✅ AUTO-DÉTECTION OK")
                print(f"      Min: {parsed.min()}")
                print(f"      Max: {parsed.max()}")
            except Exception as e:
                print(f"   ❌ Impossible de parser: {e}")
        print()

else:
    print("⚠️  AUCUNE colonne date/time trouvée!")
    print("   Colonnes disponibles:", df.columns.tolist())
    print()
    print("🔍 PREMIÈRES LIGNES:")
    print(df.head())

print("\n📈 STATISTIQUES TOTALES:")
try:
    total = len(pd.read_csv(fp, usecols=[0]))  # Lire juste 1 colonne
    print(f"   Total lignes: {total:,}")
except Exception as e:
    print(f"   ⚠️  Erreur comptage: {e}")