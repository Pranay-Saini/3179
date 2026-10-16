import re
from pathlib import Path
import numpy as np
import pandas as pd

# Automatically locates the folder where preprocessingdata.py is saved (.../3179/data)
DATA_DIR = Path(__file__).resolve().parent
print(f"Reading and writing files in: {DATA_DIR}")

# ---------------------------------------------------------
# 1. CLEAN LGA SEIFA SUMMARY
# ---------------------------------------------------------
lga_raw = pd.read_csv(DATA_DIR / "lga_seifa_summary.csv", low_memory=False)
lga = lga_raw[["LGA Code", "Region", "Score", "Decile"]].dropna().copy()
lga["LGA Code"] = lga["LGA Code"].astype(int)
lga["Decile"] = lga["Decile"].astype(int)

state_code_map = {
    1: "NSW", 2: "VIC", 3: "QLD", 4: "SA",
    5: "WA", 6: "TAS", 7: "NT", 8: "ACT", 9: "OT"
}
lga["State_Abbr"] = (lga["LGA Code"] // 10000).map(state_code_map)

# ---------------------------------------------------------
# 2. PROCESS NATIONAL TOILET MAP & SUBURB CENTROIDS
# ---------------------------------------------------------
toilets = pd.read_csv(DATA_DIR / "toilets.csv", low_memory=False)
toilets["sub_clean"] = toilets["Town"].astype(str).str.lower().str.strip()
toilets["state_clean"] = toilets["State"].astype(str).str.upper().str.strip()

# Suburb-level sanitary infrastructure summary & coordinates
suburb_toilets = toilets.groupby(["sub_clean", "state_clean"]).agg(
    Sub_Lat=("Latitude", "median"),
    Sub_Lon=("Longitude", "median"),
    Total_Toilets=("FacilityID", "count"),
    Park_Toilets=("FacilityType", lambda x: (x == "Park or reserve").sum()),
    Accessible_Toilets=("Accessible", lambda x: (x == True).sum()),
    Accessible_Park_Toilets=(
        "FacilityID",
        lambda x: (
            (toilets.loc[x.index, "FacilityType"] == "Park or reserve")
            & (toilets.loc[x.index, "Accessible"] == True)
        ).sum(),
    ),
    BabyChange_Toilets=("BabyChange", lambda x: (x == True).sum()),
    ChangingPlaces_Toilets=("ChangingPlaces", lambda x: (x == True).sum()),
    MLAK24_Toilets=("MLAK24", lambda x: (x == True).sum()),
).reset_index()

# Export national Park/Reserve accessible toilets for V09 Map
toilets_national_parks = toilets[
    (toilets["FacilityType"] == "Park or reserve")
    & (toilets["Accessible"] == True)
    & toilets["Latitude"].notna()
    & toilets["Longitude"].notna()
][
    [
        "FacilityID", "Name", "Address1", "Town", "State",
        "Latitude", "Longitude", "Accessible", "ParkingAccessible",
        "BabyChange", "ChangingPlaces", "MLAK24", "OpeningHours"
    ]
].copy()

toilets_national_parks["Amenity_Tier"] = np.where(
    toilets_national_parks["ChangingPlaces"] == True,
    "Changing Places (High-Support)",
    np.where(
        toilets_national_parks["BabyChange"] == True,
        "Accessible + Baby Change",
        "Standard Wheelchair Accessible",
    ),
)
toilets_national_parks.to_csv(DATA_DIR / "toilets_national_parks.csv", index=False)

# ---------------------------------------------------------
# 3. PROCESS OSM PLAYGROUNDS & DERIVE COORDINATES
# ---------------------------------------------------------
osm = pd.read_csv(DATA_DIR / "osm_playgrounds.csv", low_memory=False)

state_name_map = {
    "New South Wales": "NSW",
    "Victoria": "VIC",
    "Queensland": "QLD",
    "Western Australia": "WA",
    "South Australia": "SA",
    "Tasmania": "TAS",
    "Australian Capital Territory": "ACT",
    "Northern Territory": "NT",
}
osm["State_Abbr"] = osm["meta_name_state"].map(state_name_map)
osm["sub_clean"] = osm["meta_name_sub"].astype(str).str.lower().str.strip()

# Merge suburb coordinates and toilet co-location counts
osm = osm.merge(
    suburb_toilets,
    left_on=["sub_clean", "State_Abbr"],
    right_on=["sub_clean", "state_clean"],
    how="left",
)

# Fallback to LGA median coordinates, then State median coordinates
lga_coords = osm.groupby("meta_code_area")[["Sub_Lat", "Sub_Lon"]].transform("median")
osm["Sub_Lat"] = osm["Sub_Lat"].fillna(lga_coords["Sub_Lat"])
osm["Sub_Lon"] = osm["Sub_Lon"].fillna(lga_coords["Sub_Lon"])

state_coords = osm.groupby("State_Abbr")[["Sub_Lat", "Sub_Lon"]].transform("median")
osm["Sub_Lat"] = osm["Sub_Lat"].fillna(state_coords["Sub_Lat"])
osm["Sub_Lon"] = osm["Sub_Lon"].fillna(state_coords["Sub_Lon"])

# Add tiny deterministic jitter so multiple playgrounds in the same suburb are distinct points
np.random.seed(3179)
osm["Latitude"] = (osm["Sub_Lat"] + np.random.normal(0, 0.004, len(osm))).round(5)
osm["Longitude"] = (osm["Sub_Lon"] + np.random.normal(0, 0.004, len(osm))).round(5)

for col in [
    "Total_Toilets", "Park_Toilets", "Accessible_Toilets",
    "Accessible_Park_Toilets", "BabyChange_Toilets",
    "ChangingPlaces_Toilets", "MLAK24_Toilets"
]:
    osm[col] = osm[col].fillna(0).astype(int)

# Classify distance to state capital CBD for Metropolitan vs Regional cohort
capitals = {
    "NSW": (-33.8688, 151.2093),
    "VIC": (-37.8136, 144.9631),
    "QLD": (-27.4705, 153.0260),
    "SA": (-34.9285, 138.6007),
    "WA": (-31.9523, 115.8613),
    "TAS": (-42.8821, 147.3272),
    "NT": (-12.4634, 130.8456),
    "ACT": (-35.2809, 149.1300),
}

def get_urban_cohort(row):
    st = row["State_Abbr"]
    if st not in capitals or pd.isna(row["Latitude"]):
        return "Regional & Rural"
    clat, clon = capitals[st]
    dlat = (row["Latitude"] - clat) * 111.0
    dlon = (row["Longitude"] - clon) * 111.0 * np.cos(np.radians(clat))
    dist_km = np.sqrt(dlat**2 + dlon**2)
    return "Metropolitan" if dist_km <= 55.0 else "Regional & Rural"

osm["Urban_Cohort"] = osm.apply(get_urban_cohort, axis=1)

def clean_surface(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unrecorded"
    v = str(val).lower()
    if "woodchip" in v or "bark" in v or "mulch" in v or v == "wood":
        return "Woodchips & Bark"
    if "sand" in v:
        return "Sand"
    if "rubber" in v:
        return "Impact Rubber"
    if "grass" in v or "turf" in v:
        return "Grass & Turf"
    if "concrete" in v or "paved" in v or "asphalt" in v:
        return "Concrete & Paved"
    return "Earth & Gravel"

osm["Surface_Clean"] = osm["surface"].apply(clean_surface)
osm["Wheelchair_Clean"] = (
    osm["wheelchair"].fillna("unrecorded").astype(str).str.lower().str.strip()
)

def clean_operator(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Unrecorded (Default Municipal)"
    v = str(val).lower()
    if "school" in v or "college" in v:
        return "School / Education"
    if "childcare" in v or "private" in v or "ltd" in v:
        return "Private / Childcare"
    if "parks" in v or "state" in v or "authority" in v:
        return "State Park Authority"
    return "Local Council (Explicit)"

osm["Management_Clean"] = osm["operator"].apply(clean_operator)

def classify_age_cohort(row):
    min_a, max_a = row["min_age"], row["max_age"]
    if pd.notna(max_a) and float(max_a) <= 5:
        return "Early Childhood (0–5 yrs)"
    if pd.notna(min_a) and float(min_a) >= 8:
        return "Older Children & Youth (8+ yrs)"
    if pd.notna(min_a) or pd.notna(max_a):
        return "Broad All-Ages (0–12+ yrs)"
    return "Unassigned Age Target"

osm["Age_Cohort"] = osm.apply(classify_age_cohort, axis=1)
osm["Name_Clean"] = osm["name"].fillna("Unnamed Neighbourhood Playground")
osm["Toilet_Catchment_Status"] = np.where(
    osm["Accessible_Park_Toilets"] > 0,
    "Served (Accessible Park Toilet in Suburb)",
    "Unserved (No Accessible Park Toilet in Suburb)",
)

osm = osm.merge(
    lga[["LGA Code", "Score", "Decile"]],
    left_on="meta_code_area",
    right_on="LGA Code",
    how="left",
)

osm_export_cols = [
    "meta_osm_id", "Name_Clean", "Surface_Clean", "Wheelchair_Clean",
    "min_age", "max_age", "Age_Cohort", "Management_Clean", "access",
    "indoor", "meta_code_sub", "meta_code_area", "meta_name_sub",
    "meta_name_area", "meta_name_state", "State_Abbr", "Urban_Cohort",
    "Latitude", "Longitude", "Score", "Decile", "Accessible_Park_Toilets",
    "BabyChange_Toilets", "ChangingPlaces_Toilets", "Toilet_Catchment_Status",
    "meta_versions_number", "meta_users_number"
]
osm[osm_export_cols].to_csv(DATA_DIR / "osm_playgrounds_clean.csv", index=False)

# ---------------------------------------------------------
# 4. AGGREGATE LGA SUMMARY WITH SEIFA & OSM METRICS
# ---------------------------------------------------------
lga_osm_agg = osm.groupby("meta_code_area").agg(
    Playground_Count=("meta_osm_id", "count"),
    Surface_Audited_Count=("Surface_Clean", lambda x: (x != "Unrecorded").sum()),
    Wheelchair_Audited_Count=("Wheelchair_Clean", lambda x: (x != "unrecorded").sum()),
    Wheelchair_Yes_Count=("Wheelchair_Clean", lambda x: (x == "yes").sum()),
    Rubber_Surface_Count=("Surface_Clean", lambda x: (x == "Impact Rubber").sum()),
    Toilets_Served_Count=("Accessible_Park_Toilets", lambda x: (x > 0).sum()),
    Mean_Contributors=("meta_users_number", "mean"),
    Urban_Cohort=("Urban_Cohort", lambda x: x.mode()[0] if len(x.mode()) > 0 else "Regional & Rural"),
).reset_index()

lga_merged = lga.merge(
    lga_osm_agg, left_on="LGA Code", right_on="meta_code_area", how="left"
)
for col in [
    "Playground_Count", "Surface_Audited_Count", "Wheelchair_Audited_Count",
    "Wheelchair_Yes_Count", "Rubber_Surface_Count", "Toilets_Served_Count"
]:
    lga_merged[col] = lga_merged[col].fillna(0).astype(int)

lga_merged["Urban_Cohort"] = lga_merged["Urban_Cohort"].fillna("Regional & Rural")
lga_merged["Surface_Audit_Pct"] = np.where(
    lga_merged["Playground_Count"] > 0,
    (lga_merged["Surface_Audited_Count"] / lga_merged["Playground_Count"]) * 100,
    0,
).round(1)
lga_merged["Toilet_CoLocation_Pct"] = np.where(
    lga_merged["Playground_Count"] > 0,
    (lga_merged["Toilets_Served_Count"] / lga_merged["Playground_Count"]) * 100,
    0,
).round(1)

lga_merged.to_csv(DATA_DIR / "lga_seifa_clean.csv", index=False)

# ---------------------------------------------------------
# 5. CLEAN NSW EVERYONE CAN PLAY & JOIN WITH SEIFA
# ---------------------------------------------------------
ecp_path = DATA_DIR / "nsw_everyone_can_play.csv"
try:
    ecp = pd.read_excel(ecp_path)
except Exception:
    ecp = pd.read_csv(ecp_path)

def clean_council_str(name):
    s = str(name).lower()
    s = s.replace("(nsw)", "").replace("’", "").replace("'", "").replace("-", " ")
    s = re.sub(
        r"\b(city|shire|regional|municipal|municipality|of|council|councl|the|valley)\b",
        "",
        s,
    )
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

lga_nsw = lga[lga["State_Abbr"] == "NSW"].copy()
lga_nsw["clean_key"] = lga_nsw["Region"].apply(clean_council_str)

ecp["clean_key"] = ecp["Council"].apply(clean_council_str)
ecp.loc[ecp["clean_key"] == "midcoast", "clean_key"] = "mid coast"
ecp.loc[ecp["Council"].str.contains("Housing", na=False), "clean_key"] = "campbelltown"

ecp_merged = ecp.merge(
    lga_nsw[["LGA Code", "Region", "Score", "Decile", "clean_key"]],
    on="clean_key",
    how="left",
)

metro_nsw_regions = {
    "Bayside (NSW)", "Blacktown", "Blue Mountains", "Burwood", "Camden",
    "Campbelltown (NSW)", "Canada Bay", "Canterbury-Bankstown",
    "Central Coast (NSW)", "Cumberland", "Fairfield", "Georges River",
    "Hawkesbury", "Hornsby", "Hunters Hill", "Inner West", "Ku-ring-gai",
    "Lane Cove", "Liverpool", "Mosman", "North Sydney", "Northern Beaches",
    "Parramatta", "Penrith", "Randwick", "Ryde", "Strathfield",
    "Sutherland Shire", "Sydney", "The Hills Shire", "Waverley",
    "Willoughby", "Wollondilly", "Wollongong", "Woollahra",
    "Shellharbour", "Newcastle", "Lake Macquarie"
}
ecp_merged["Council_Cohort"] = np.where(
    ecp_merged["Region"].isin(metro_nsw_regions),
    "Metropolitan",
    "Regional & Rural",
)

ecp_merged.drop(columns=["clean_key"]).to_csv(
    DATA_DIR / "nsw_everyone_can_play_clean.csv", index=False
)

print("Success! Generated:")
print(" -", DATA_DIR / "toilets_national_parks.csv")
print(" -", DATA_DIR / "osm_playgrounds_clean.csv")
print(" -", DATA_DIR / "lga_seifa_clean.csv")
print(" -", DATA_DIR / "nsw_everyone_can_play_clean.csv")