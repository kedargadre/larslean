"""Léarslán V3 — Configuration & Constants (Electoral Division Level)"""
import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
USE_REAL_DATA = True

if USE_REAL_DATA:
    DATA_DIR = PROJECT_DIR / "data" / "real_data"
else:
    DATA_DIR = PROJECT_DIR / "data"

# County-level data (backward compat)
CSO_FILE = DATA_DIR / "cso_employment.csv"
TII_FILE = DATA_DIR / "tii_traffic.csv"
SEAI_FILE = DATA_DIR / "seai_ber.csv"
RTB_FILE = DATA_DIR / "rtb_rent.csv"
GEOJSON_FILE = DATA_DIR / "ireland_counties.geojson"

# Electoral Division-level data
CSO_ED_FILE = DATA_DIR / "cso_employment_ed.csv"
TII_ED_FILE = DATA_DIR / "tii_traffic_ed.csv"
SEAI_ED_FILE = DATA_DIR / "seai_ber_ed.csv"
RTB_ED_FILE = DATA_DIR / "rtb_rent_ed.csv"
ED_GEOJSON_FILE = DATA_DIR / "ireland_eds.geojson"

# ── GADM GeoJSON Download URLs ────────────────────────────────
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IRL_1.json"
GADM_ED_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IRL_2.json"

# ── Spatial Level Toggle ──────────────────────────────────────
# "county" for legacy 26-county view, "ed" for Electoral Division level
DEFAULT_SPATIAL_LEVEL = "county"

# ── 26 Counties of the Republic of Ireland ─────────────────────
IRISH_COUNTIES = [
    "Carlow", "Cavan", "Clare", "Cork", "Donegal", "Dublin",
    "Galway", "Kerry", "Kildare", "Kilkenny", "Laois", "Leitrim",
    "Limerick", "Longford", "Louth", "Mayo", "Meath", "Monaghan",
    "Offaly", "Roscommon", "Sligo", "Tipperary", "Waterford",
    "Westmeath", "Wexford", "Wicklow",
]

# ── County Centroids (lat, lon) ────────────────────────────────
COUNTY_CENTROIDS = {
    "Carlow": (52.84, -6.93), "Cavan": (53.99, -7.36),
    "Clare": (52.84, -8.98), "Cork": (51.90, -8.47),
    "Donegal": (54.83, -7.78), "Dublin": (53.35, -6.26),
    "Galway": (53.27, -8.86), "Kerry": (52.06, -9.99),
    "Kildare": (53.22, -6.77), "Kilkenny": (52.65, -7.25),
    "Laois": (52.99, -7.56), "Leitrim": (54.12, -8.00),
    "Limerick": (52.66, -8.63), "Longford": (53.73, -7.79),
    "Louth": (53.92, -6.49), "Mayo": (53.90, -9.30),
    "Meath": (53.65, -6.65), "Monaghan": (54.25, -6.97),
    "Offaly": (53.23, -7.60), "Roscommon": (53.76, -8.27),
    "Sligo": (54.28, -8.47), "Tipperary": (52.67, -7.85),
    "Waterford": (52.26, -7.11), "Westmeath": (53.53, -7.35),
    "Wexford": (52.47, -6.57), "Wicklow": (52.98, -6.37),
}

# ── Baseline Statistics (seeded from known Irish data) ─────────
COUNTY_BASELINES = {
    "Dublin":     {"rent": 2100, "income": 55000, "employment": 0.76, "traffic": 85000, "congestion": 28, "ber": 3.2, "energy_cost": 2100, "rent_growth": 0.18},
    "Cork":       {"rent": 1500, "income": 45000, "employment": 0.72, "traffic": 45000, "congestion": 18, "ber": 3.5, "energy_cost": 2300, "rent_growth": 0.12},
    "Galway":     {"rent": 1350, "income": 42000, "employment": 0.71, "traffic": 30000, "congestion": 15, "ber": 3.6, "energy_cost": 2400, "rent_growth": 0.14},
    "Limerick":   {"rent": 1200, "income": 40000, "employment": 0.70, "traffic": 28000, "congestion": 14, "ber": 3.7, "energy_cost": 2500, "rent_growth": 0.11},
    "Waterford":  {"rent": 1100, "income": 38000, "employment": 0.68, "traffic": 22000, "congestion": 12, "ber": 3.8, "energy_cost": 2600, "rent_growth": 0.10},
    "Kerry":      {"rent": 1000, "income": 36000, "employment": 0.66, "traffic": 18000, "congestion": 10, "ber": 4.0, "energy_cost": 2800, "rent_growth": 0.08},
    "Clare":      {"rent": 950,  "income": 37000, "employment": 0.67, "traffic": 20000, "congestion": 11, "ber": 3.9, "energy_cost": 2700, "rent_growth": 0.07},
    "Tipperary":  {"rent": 900,  "income": 35000, "employment": 0.66, "traffic": 19000, "congestion": 10, "ber": 4.1, "energy_cost": 2900, "rent_growth": 0.06},
    "Kilkenny":   {"rent": 950,  "income": 36000, "employment": 0.67, "traffic": 17000, "congestion": 9,  "ber": 3.9, "energy_cost": 2700, "rent_growth": 0.07},
    "Wexford":    {"rent": 1000, "income": 35000, "employment": 0.65, "traffic": 18000, "congestion": 10, "ber": 4.0, "energy_cost": 2800, "rent_growth": 0.09},
    "Wicklow":    {"rent": 1400, "income": 48000, "employment": 0.73, "traffic": 35000, "congestion": 20, "ber": 3.3, "energy_cost": 2200, "rent_growth": 0.15},
    "Kildare":    {"rent": 1350, "income": 47000, "employment": 0.74, "traffic": 38000, "congestion": 22, "ber": 3.2, "energy_cost": 2100, "rent_growth": 0.16},
    "Meath":      {"rent": 1300, "income": 46000, "employment": 0.73, "traffic": 36000, "congestion": 21, "ber": 3.3, "energy_cost": 2200, "rent_growth": 0.14},
    "Louth":      {"rent": 1200, "income": 39000, "employment": 0.69, "traffic": 25000, "congestion": 15, "ber": 3.6, "energy_cost": 2500, "rent_growth": 0.13},
    "Westmeath":  {"rent": 900,  "income": 35000, "employment": 0.66, "traffic": 16000, "congestion": 8,  "ber": 4.0, "energy_cost": 2800, "rent_growth": 0.06},
    "Offaly":     {"rent": 800,  "income": 33000, "employment": 0.63, "traffic": 14000, "congestion": 7,  "ber": 4.2, "energy_cost": 3000, "rent_growth": 0.05},
    "Laois":      {"rent": 850,  "income": 34000, "employment": 0.64, "traffic": 15000, "congestion": 8,  "ber": 4.1, "energy_cost": 2900, "rent_growth": 0.06},
    "Carlow":     {"rent": 950,  "income": 34000, "employment": 0.65, "traffic": 14000, "congestion": 7,  "ber": 4.0, "energy_cost": 2800, "rent_growth": 0.07},
    "Longford":   {"rent": 750,  "income": 31000, "employment": 0.61, "traffic": 10000, "congestion": 5,  "ber": 4.3, "energy_cost": 3100, "rent_growth": 0.04},
    "Roscommon":  {"rent": 700,  "income": 32000, "employment": 0.62, "traffic": 11000, "congestion": 5,  "ber": 4.4, "energy_cost": 3200, "rent_growth": 0.03},
    "Sligo":      {"rent": 800,  "income": 33000, "employment": 0.63, "traffic": 13000, "congestion": 7,  "ber": 4.2, "energy_cost": 3000, "rent_growth": 0.05},
    "Leitrim":    {"rent": 650,  "income": 30000, "employment": 0.60, "traffic": 8000,  "congestion": 4,  "ber": 4.5, "energy_cost": 3300, "rent_growth": 0.02},
    "Mayo":       {"rent": 750,  "income": 32000, "employment": 0.62, "traffic": 12000, "congestion": 6,  "ber": 4.3, "energy_cost": 3100, "rent_growth": 0.04},
    "Donegal":    {"rent": 700,  "income": 31000, "employment": 0.60, "traffic": 11000, "congestion": 5,  "ber": 4.4, "energy_cost": 3200, "rent_growth": 0.03},
    "Cavan":      {"rent": 750,  "income": 32000, "employment": 0.62, "traffic": 12000, "congestion": 6,  "ber": 4.3, "energy_cost": 3100, "rent_growth": 0.04},
    "Monaghan":   {"rent": 750,  "income": 32000, "employment": 0.62, "traffic": 11000, "congestion": 6,  "ber": 4.3, "energy_cost": 3100, "rent_growth": 0.04},
}

# ══════════════════════════════════════════════════════════════
# ELECTORAL DIVISION (ED) REGISTRY
# Real ED names per county, with variance modifiers for synthetic data.
# Each ED has a "type" that drives its baseline modifier:
#   "urban_core"    → +30% rent, +20% income, +40% traffic, −15% BER
#   "suburban"      → +10% rent, +10% income, +20% traffic, −5% BER
#   "town"          → ±0% (baseline)
#   "village"       → −15% rent, −10% income, −20% traffic, +10% BER
#   "rural"         → −30% rent, −20% income, −50% traffic, +20% BER
# ══════════════════════════════════════════════════════════════

ED_TYPE_MODIFIERS = {
    "urban_core": {"rent": 1.30, "income": 1.20, "employment": 1.05, "traffic": 1.40, "congestion": 1.35, "ber": 0.85, "energy_cost": 0.85, "rent_growth": 1.20},
    "suburban":   {"rent": 1.10, "income": 1.10, "employment": 1.02, "traffic": 1.20, "congestion": 1.15, "ber": 0.95, "energy_cost": 0.95, "rent_growth": 1.10},
    "town":       {"rent": 1.00, "income": 1.00, "employment": 1.00, "traffic": 1.00, "congestion": 1.00, "ber": 1.00, "energy_cost": 1.00, "rent_growth": 1.00},
    "village":    {"rent": 0.85, "income": 0.90, "employment": 0.97, "traffic": 0.80, "congestion": 0.80, "ber": 1.10, "energy_cost": 1.10, "rent_growth": 0.85},
    "rural":      {"rent": 0.70, "income": 0.80, "employment": 0.93, "traffic": 0.50, "congestion": 0.60, "ber": 1.20, "energy_cost": 1.20, "rent_growth": 0.70},
}

# ── Electoral Division Registry ────────────────────────────────
# Format: {county: [(ed_name, ed_type), ...]}
# Uses real Irish Electoral Division names
ED_REGISTRY = {
    "Dublin": [
        ("Rathmines East A", "urban_core"),
        ("Rathmines West C", "urban_core"),
        ("Pembroke East A", "urban_core"),
        ("Ballsbridge", "urban_core"),
        ("Drumcondra South A", "urban_core"),
        ("Cabra East A", "suburban"),
        ("Clontarf East A", "suburban"),
        ("Finglas South A", "suburban"),
        ("Tallaght-Fettercairn", "suburban"),
        ("Lucan-Esker", "suburban"),
        ("Blanchardstown-Blakestown", "suburban"),
        ("Swords-Forrest", "town"),
        ("Howth", "suburban"),
        ("Dalkey", "suburban"),
        ("Blackrock-Carysfort", "urban_core"),
    ],
    "Cork": [
        ("Cork City South Central", "urban_core"),
        ("Cork City North Central", "urban_core"),
        ("Blackrock (Cork)", "suburban"),
        ("Douglas", "suburban"),
        ("Ballincollig", "suburban"),
        ("Carrigaline", "town"),
        ("Cobh Urban", "town"),
        ("Midleton", "town"),
        ("Mallow Urban", "town"),
        ("Fermoy Urban", "town"),
        ("Bandon", "village"),
        ("Kinsale", "village"),
        ("Skibbereen", "rural"),
        ("Bantry", "rural"),
    ],
    "Galway": [
        ("Galway City East", "urban_core"),
        ("Galway City West", "urban_core"),
        ("Salthill", "suburban"),
        ("Knocknacarra", "suburban"),
        ("Oranmore", "town"),
        ("Tuam Urban", "town"),
        ("Loughrea", "town"),
        ("Ballinasloe Urban", "town"),
        ("Athenry", "village"),
        ("Clifden", "village"),
        ("Oughterard", "rural"),
        ("Spiddal", "rural"),
    ],
    "Limerick": [
        ("Limerick City East", "urban_core"),
        ("Limerick City West", "urban_core"),
        ("Castletroy", "suburban"),
        ("Raheen", "suburban"),
        ("Dooradoyle", "suburban"),
        ("Newcastle West", "town"),
        ("Adare", "town"),
        ("Kilmallock", "village"),
        ("Abbeyfeale", "village"),
        ("Askeaton", "rural"),
        ("Bruff", "rural"),
    ],
    "Waterford": [
        ("Waterford City Centre", "urban_core"),
        ("Waterford City South", "urban_core"),
        ("Tramore Urban", "suburban"),
        ("Dunmore East", "suburban"),
        ("Dungarvan Urban", "town"),
        ("Lismore", "village"),
        ("Cappoquin", "village"),
        ("Ardmore", "rural"),
        ("Portlaw", "rural"),
        ("Kilmacthomas", "rural"),
    ],
    "Kerry": [
        ("Tralee Urban", "urban_core"),
        ("Killarney Urban", "urban_core"),
        ("Listowel Urban", "town"),
        ("Dingle", "town"),
        ("Kenmare", "village"),
        ("Cahirciveen", "village"),
        ("Killorglin", "village"),
        ("Castleisland", "town"),
        ("Sneem", "rural"),
        ("Waterville", "rural"),
    ],
    "Clare": [
        ("Ennis Urban", "urban_core"),
        ("Shannon", "suburban"),
        ("Kilrush Urban", "town"),
        ("Ennistymon", "village"),
        ("Kilkee", "village"),
        ("Newmarket-on-Fergus", "town"),
        ("Sixmilebridge", "village"),
        ("Scarriff", "rural"),
        ("Lisdoonvarna", "rural"),
        ("Miltown Malbay", "rural"),
    ],
    "Tipperary": [
        ("Clonmel East Urban", "urban_core"),
        ("Clonmel West Urban", "urban_core"),
        ("Thurles Urban", "town"),
        ("Nenagh Urban", "town"),
        ("Carrick-on-Suir Urban", "town"),
        ("Cashel Urban", "town"),
        ("Roscrea", "town"),
        ("Tipperary Urban", "village"),
        ("Cahir", "village"),
        ("Templemore", "rural"),
    ],
    "Kilkenny": [
        ("Kilkenny No. 1 Urban", "urban_core"),
        ("Kilkenny No. 2 Urban", "urban_core"),
        ("Castlecomer", "town"),
        ("Callan", "village"),
        ("Thomastown", "village"),
        ("Graiguenamanagh", "village"),
        ("Freshford", "rural"),
        ("Urlingford", "rural"),
        ("Piltown", "rural"),
    ],
    "Wexford": [
        ("Wexford Urban", "urban_core"),
        ("Wexford Rural", "suburban"),
        ("New Ross Urban", "town"),
        ("Enniscorthy Urban", "town"),
        ("Gorey", "town"),
        ("Bunclody", "village"),
        ("Ferns", "village"),
        ("Rosslare", "village"),
        ("Courtown", "rural"),
        ("Kilmore Quay", "rural"),
    ],
    "Wicklow": [
        ("Bray No. 1", "urban_core"),
        ("Bray No. 2", "urban_core"),
        ("Greystones", "suburban"),
        ("Arklow Urban", "town"),
        ("Wicklow Urban", "town"),
        ("Rathnew", "suburban"),
        ("Blessington", "town"),
        ("Aughrim", "village"),
        ("Baltinglass", "village"),
        ("Roundwood", "rural"),
    ],
    "Kildare": [
        ("Naas Urban", "urban_core"),
        ("Newbridge", "suburban"),
        ("Maynooth", "suburban"),
        ("Celbridge", "suburban"),
        ("Leixlip", "suburban"),
        ("Athy Urban", "town"),
        ("Kildare Urban", "town"),
        ("Monasterevin", "village"),
        ("Kilcullen", "village"),
        ("Clane", "town"),
    ],
    "Meath": [
        ("Navan Urban", "urban_core"),
        ("Trim Urban", "town"),
        ("Ashbourne", "suburban"),
        ("Ratoath", "suburban"),
        ("Dunboyne", "suburban"),
        ("Kells Urban", "town"),
        ("Dunshaughlin", "town"),
        ("Bettystown", "suburban"),
        ("Laytown", "town"),
        ("Enfield", "village"),
    ],
    "Louth": [
        ("Drogheda East Urban", "urban_core"),
        ("Drogheda West Urban", "urban_core"),
        ("Dundalk Urban", "urban_core"),
        ("Dundalk Rural", "suburban"),
        ("Ardee", "town"),
        ("Dunleer", "village"),
        ("Carlingford", "village"),
        ("Blackrock (Louth)", "village"),
        ("Clogherhead", "rural"),
        ("Cooley", "rural"),
    ],
    "Westmeath": [
        ("Athlone East Urban", "urban_core"),
        ("Athlone West Urban", "urban_core"),
        ("Mullingar North", "suburban"),
        ("Mullingar South", "suburban"),
        ("Moate", "town"),
        ("Kilbeggan", "village"),
        ("Kinnegad", "village"),
        ("Castlepollard", "rural"),
        ("Fore", "rural"),
    ],
    "Offaly": [
        ("Tullamore Urban", "urban_core"),
        ("Birr Urban", "town"),
        ("Edenderry", "town"),
        ("Clara", "village"),
        ("Ferbane", "village"),
        ("Banagher", "village"),
        ("Kilcormac", "rural"),
        ("Shinrone", "rural"),
        ("Cloghan", "rural"),
    ],
    "Laois": [
        ("Portlaoise Urban", "urban_core"),
        ("Mountmellick", "town"),
        ("Portarlington", "town"),
        ("Mountrath", "village"),
        ("Abbeyleix", "village"),
        ("Durrow", "village"),
        ("Rathdowney", "rural"),
        ("Stradbally", "rural"),
        ("Ballinakill", "rural"),
    ],
    "Carlow": [
        ("Carlow Urban", "urban_core"),
        ("Carlow Rural", "suburban"),
        ("Tullow", "town"),
        ("Muine Bheag (Bagenalstown)", "village"),
        ("Hacketstown", "village"),
        ("Borris", "rural"),
        ("Leighlinbridge", "rural"),
        ("Myshall", "rural"),
    ],
    "Longford": [
        ("Longford Urban", "urban_core"),
        ("Longford Rural", "suburban"),
        ("Edgeworthstown", "town"),
        ("Ballymahon", "village"),
        ("Granard", "village"),
        ("Lanesborough", "village"),
        ("Drumlish", "rural"),
        ("Newtownforbes", "rural"),
    ],
    "Roscommon": [
        ("Roscommon Urban", "urban_core"),
        ("Boyle Urban", "town"),
        ("Castlerea", "town"),
        ("Ballaghaderreen", "village"),
        ("Strokestown", "village"),
        ("Elphin", "rural"),
        ("Athleague", "rural"),
        ("Frenchpark", "rural"),
    ],
    "Sligo": [
        ("Sligo East", "urban_core"),
        ("Sligo West", "urban_core"),
        ("Strandhill", "suburban"),
        ("Rosses Point", "suburban"),
        ("Ballymote", "village"),
        ("Tubbercurry", "village"),
        ("Collooney", "village"),
        ("Easkey", "rural"),
        ("Enniscrone", "rural"),
    ],
    "Leitrim": [
        ("Carrick-on-Shannon", "urban_core"),
        ("Manorhamilton", "town"),
        ("Ballinamore", "village"),
        ("Mohill", "village"),
        ("Drumshanbo", "village"),
        ("Kinlough", "rural"),
        ("Dromahair", "rural"),
        ("Drumkeerin", "rural"),
    ],
    "Mayo": [
        ("Castlebar Urban", "urban_core"),
        ("Westport Urban", "urban_core"),
        ("Ballina Urban", "town"),
        ("Claremorris", "town"),
        ("Ballinrobe", "village"),
        ("Swinford", "village"),
        ("Belmullet", "village"),
        ("Knock", "village"),
        ("Achill", "rural"),
        ("Louisburgh", "rural"),
    ],
    "Donegal": [
        ("Letterkenny", "urban_core"),
        ("Donegal Urban", "town"),
        ("Buncrana", "town"),
        ("Bundoran", "town"),
        ("Carndonagh", "village"),
        ("Ballybofey", "village"),
        ("Milford", "village"),
        ("Dungloe", "rural"),
        ("Glenties", "rural"),
        ("Falcarragh", "rural"),
    ],
    "Cavan": [
        ("Cavan Urban", "urban_core"),
        ("Virginia", "town"),
        ("Bailieborough", "town"),
        ("Kingscourt", "village"),
        ("Cootehill", "village"),
        ("Belturbet", "village"),
        ("Ballyconnell", "rural"),
        ("Ballyjamesduff", "rural"),
    ],
    "Monaghan": [
        ("Monaghan Urban", "urban_core"),
        ("Carrickmacross", "town"),
        ("Castleblayney", "town"),
        ("Clones", "village"),
        ("Ballybay", "village"),
        ("Emyvale", "rural"),
        ("Glaslough", "rural"),
        ("Scotstown", "rural"),
    ],
}


def get_ed_id(county: str, ed_name: str) -> str:
    """Generate a stable ED identifier from county and ED name."""
    clean = ed_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    clean = clean.replace("-", "_").replace("'", "").replace(".", "")
    county_prefix = county[:3].upper()
    return f"{county_prefix}_{clean}"


def get_all_eds() -> list:
    """Return flat list of (ed_id, ed_name, county, ed_type) tuples."""
    eds = []
    for county, ed_list in ED_REGISTRY.items():
        for ed_name, ed_type in ed_list:
            ed_id = get_ed_id(county, ed_name)
            eds.append((ed_id, ed_name, county, ed_type))
    return eds


def get_county_eds(county: str) -> list:
    """Return list of (ed_id, ed_name, ed_type) for a given county."""
    eds = []
    for ed_name, ed_type in ED_REGISTRY.get(county, []):
        ed_id = get_ed_id(county, ed_name)
        eds.append((ed_id, ed_name, ed_type))
    return eds


def get_ed_baseline(county: str, ed_type: str) -> dict:
    """Get adjusted baseline for an ED based on county baseline + type modifier."""
    county_base = COUNTY_BASELINES.get(county, COUNTY_BASELINES["Dublin"])
    modifiers = ED_TYPE_MODIFIERS.get(ed_type, ED_TYPE_MODIFIERS["town"])
    return {k: v * modifiers.get(k, 1.0) for k, v in county_base.items()}


# ── Feature Columns ────────────────────────────────────────────
FEATURE_COLS = [
    "avg_monthly_rent", "rent_growth_pct", "avg_income",
    "employment_rate", "traffic_volume", "congestion_delay_minutes",
    "ber_avg_score", "est_annual_energy_cost",
    "commute_to_rent_ratio", "energy_tax", "true_cost_index",
]

# ── Color Palette ──────────────────────────────────────────────
COLORS = {
    "bg_dark": "#0a1628",
    "bg_card": "#111a2e",
    "bg_card_hover": "#1a2744",
    "accent_green": "#10b981",
    "accent_emerald": "#34d399",
    "accent_amber": "#f59e0b",
    "accent_red": "#ef4444",
    "accent_blue": "#3b82f6",
    "accent_purple": "#8b5cf6",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "border": "#1e293b",
    "gradient_start": "#10b981",
    "gradient_end": "#3b82f6",
}

# ── Map Settings ───────────────────────────────────────────────
MAP_CENTER = [53.5, -7.5]
MAP_ZOOM = 7

# ── Score Thresholds ───────────────────────────────────────────
RISK_THRESHOLDS = {"Low": (0, 33), "Medium": (34, 66), "High": (67, 100)}

# ── OpenAI Settings ────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# ── Gemini Settings ────────────────────────────────────────────
from dotenv import load_dotenv as _ld
_ld(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Forecasting Settings ──────────────────────────────────────
FORECAST_PERIODS = 6  # months ahead
