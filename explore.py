import pandas as pd
import json
from collections import Counter

# Load all datasets
cso    = pd.read_csv('data/real_data/cso_employment.csv')
cso_ed = pd.read_csv('data/real_data/cso_employment_ed.csv')
rtb    = pd.read_csv('data/real_data/rtb_rent.csv')
rtb_ed = pd.read_csv('data/real_data/rtb_rent_ed.csv')
seai   = pd.read_csv('data/real_data/seai_ber.csv')
seai_ed= pd.read_csv('data/real_data/seai_ber_ed.csv')
tii    = pd.read_csv('data/real_data/tii_traffic.csv')
tii_ed = pd.read_csv('data/real_data/tii_traffic_ed.csv')

datasets = {
    'cso_employment':    cso,
    'cso_employment_ed': cso_ed,
    'rtb_rent':          rtb,
    'rtb_rent_ed':       rtb_ed,
    'seai_ber':          seai,
    'seai_ber_ed':       seai_ed,
    'tii_traffic':       tii,
    'tii_traffic_ed':    tii_ed,
}

# ── 1. Shape & Columns ──────────────────────────────────────────────────────
print("=" * 60)
print("1. SHAPE & COLUMNS")
print("=" * 60)
for name, df in datasets.items():
    print(f"\n[{name}]  shape={df.shape}")
    print(f"  columns: {list(df.columns)}")

# ── 2. Describe (numeric stats) ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DESCRIBE (numeric columns)")
print("=" * 60)
for name, df in datasets.items():
    print(f"\n[{name}]")
    print(df.describe().to_string())

# ── 3. Null counts ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. NULL COUNTS")
print("=" * 60)
for name, df in datasets.items():
    nulls = df.isnull().sum()
    total_nulls = nulls.sum()
    print(f"[{name}]  total_nulls={total_nulls}")
    if total_nulls > 0:
        print(nulls[nulls > 0])

# ── 4. Time ranges ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TIME RANGES (monthly datasets)")
print("=" * 60)
for name, df in [('cso', cso), ('rtb', rtb), ('tii', tii),
                 ('cso_ed', cso_ed), ('rtb_ed', rtb_ed), ('tii_ed', tii_ed)]:
    months = sorted(df['month'].unique())
    print(f"[{name}]  {len(months)} months  |  {months[0]}  ->  {months[-1]}")

# ── 5. County & ED unique counts ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. UNIQUE COUNTIES & EDs")
print("=" * 60)
for name, df in [('cso', cso), ('rtb', rtb), ('seai', seai), ('tii', tii)]:
    print(f"[{name}]  {df['county'].nunique()} counties")
print()
for name, df in [('cso_ed', cso_ed), ('rtb_ed', rtb_ed), ('seai_ed', seai_ed), ('tii_ed', tii_ed)]:
    print(f"[{name}]  {df['ed_id'].nunique()} EDs  |  {df['county'].nunique()} counties")

# ── 6. County list ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. COUNTY LIST")
print("=" * 60)
print(sorted(cso['county'].unique()))

# ── 7. Sample ED IDs ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SAMPLE ED IDs (first 10)")
print("=" * 60)
print(list(cso_ed['ed_id'].unique()[:10]))

# ── 8. Join key consistency ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. JOIN KEY CONSISTENCY (ed_id across ED files)")
print("=" * 60)
ids = {
    'cso_ed':  set(cso_ed['ed_id'].unique()),
    'rtb_ed':  set(rtb_ed['ed_id'].unique()),
    'seai_ed': set(seai_ed['ed_id'].unique()),
    'tii_ed':  set(tii_ed['ed_id'].unique()),
}
for name, s in ids.items():
    print(f"  {name}: {len(s)} unique ed_ids")
intersection = ids['cso_ed'] & ids['rtb_ed'] & ids['seai_ed'] & ids['tii_ed']
print(f"  All 4 identical: {ids['cso_ed'] == ids['rtb_ed'] == ids['seai_ed'] == ids['tii_ed']}")
print(f"  Intersection size: {len(intersection)}")

# ── 9. GeoJSON structure ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. GEOJSON STRUCTURE")
print("=" * 60)
for geo_file, label in [('data/real_data/ireland_counties.geojson', 'counties'),
                         ('data/real_data/ireland_eds.geojson', 'EDs')]:
    with open(geo_file) as f:
        geo = json.load(f)
    features = geo['features']
    props = features[0]['properties']
    print(f"\n[{label}]  features={len(features)}")
    print(f"  property keys: {list(props.keys())}")
    if 'ed_type' in props:
        ed_types = [ft['properties'].get('ed_type', 'N/A') for ft in features]
        print(f"  ed_type distribution: {dict(Counter(ed_types))}")

# ── 10. Key value ranges (spot-check) ────────────────────────────────────────
print("\n" + "=" * 60)
print("10. KEY VALUE RANGES")
print("=" * 60)
print("\n[Rent - county level]")
print(f"  min={rtb['avg_monthly_rent'].min():.0f}  max={rtb['avg_monthly_rent'].max():.0f}  mean={rtb['avg_monthly_rent'].mean():.0f}")
print(f"  rent_growth_pct unique values: {sorted(rtb['rent_growth_pct'].unique())}")

print("\n[Rent - ED level]")
print(f"  min={rtb_ed['avg_monthly_rent'].min():.0f}  max={rtb_ed['avg_monthly_rent'].max():.0f}  mean={rtb_ed['avg_monthly_rent'].mean():.0f}")

print("\n[Employment rate - county]")
print(f"  min={cso['employment_rate'].min():.3f}  max={cso['employment_rate'].max():.3f}  mean={cso['employment_rate'].mean():.3f}")

print("\n[Congestion delay - county]")
print(f"  min={tii['congestion_delay_minutes'].min():.1f}  max={tii['congestion_delay_minutes'].max():.1f}  mean={tii['congestion_delay_minutes'].mean():.1f}")

print("\n[BER avg score - county]")
print(f"  min={seai['ber_avg_score'].min():.2f}  max={seai['ber_avg_score'].max():.2f}  mean={seai['ber_avg_score'].mean():.2f}")
print(f"  pct_a_rated unique: {seai['pct_a_rated'].unique()}")
print(f"  pct_bcd_rated unique: {seai['pct_bcd_rated'].unique()}")
print(f"  avg_speed_kph unique (tii): {tii['avg_speed_kph'].unique()}")
print(f"  avg_speed_kph unique (tii_ed): {tii_ed['avg_speed_kph'].unique()}")

print("\nDone.")
