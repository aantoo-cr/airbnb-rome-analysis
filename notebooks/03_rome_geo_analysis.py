# ==========================================
# STEP 1: IMPORT CORE LIBRARIES & CONFIG
# ==========================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style for the business report
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("✅ Step 1 Success: Core libraries successfully imported!")


# ==========================================
# STEP 2: DATA LOADING AND CLEANING
# ==========================================
print("\n🔄 Step 2: Loading Rome Airbnb listings data...")

# Match columns perfectly with the standard summary schema (using 'neighbourhood')
geo_cols = [
    'id', 'name', 'neighbourhood', 'latitude', 'longitude', 
    'room_type', 'price', 'number_of_reviews'
]

# Load dataset
try:
    df = pd.read_csv('data/listings.csv', low_memory=False)
    df_geo = df[geo_cols].copy()
    print(f"✅ Data successfully loaded. Shape: {df_geo.shape}")
except FileNotFoundError:
    print("❌ Error: 'data/listings.csv' not found. Please check your file name and path.")
    exit()

# Clean price data (handling potential string formats from Airbnb)
print(" sweep: Cleaning price data...")
df_geo['price'] = df_geo['price'].astype(str).str.replace('$', '').str.replace(',', '')
df_geo['price'] = pd.to_numeric(df_geo['price'], errors='coerce')

# Filter out invalid prices and missing spatial coordinates
df_geo = df_geo[df_geo['price'] > 0].dropna(subset=['price', 'latitude', 'longitude'])

print(f"✅ Step 2 Success: Filtered down to {df_geo.shape[0]} valid listings with clean prices.")


# ==========================================
# STEP 3: NEIGHBORHOOD ANALYSIS & VISUALIZATION
# ==========================================
print("\n📊 Step 3: Analyzing Rome neighborhoods and generating maps...")

# Group by neighborhood and compute median price and counts
neighborhood_stats = df_geo.groupby('neighbourhood')['price'].agg(['median', 'count'])
top_10_expensive = neighborhood_stats.sort_values(by='median', ascending=False).head(10)

print("\n💵 --- TOP 10 MOST EXPENSIVE NEIGHBORHOODS IN ROME (Median Price) ---")
print(top_10_expensive)

# Plot 1: Top 10 Neighborhoods Barplot
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_expensive['median'], y=top_10_expensive.index, palette='viridis')
plt.title('Top 10 Most Expensive Neighborhoods in Rome (Median Price per Night)', fontsize=14)
plt.xlabel('Median Price per Night ($)')
plt.ylabel('Neighborhood')
plt.tight_layout()
plt.savefig('roma_top_neighborhoods.png', dpi=300)
plt.close()
print("💾 Saved plot: 'roma_top_neighborhoods.png'")

# Plot 2: Geographic Price Distribution (95th Percentile to avoid luxury outliers)
price_limit = df_geo['price'].quantile(0.95)
df_map = df_geo[df_geo['price'] <= price_limit]

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_map['longitude'], 
    df_map['latitude'], 
    c=df_map['price'], 
    cmap='plasma', 
    alpha=0.5, 
    s=12
)
plt.colorbar(scatter, label='Price per Night ($)')
plt.title('Geographic Distribution of Airbnb Prices in Rome (95th Percentile)', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('roma_price_map.png', dpi=300)
plt.close()
print("💾 Saved plot: 'roma_price_map.png'")

print("✅ Step 3 Success: Visualizations generated and saved to your project folder!")

# ==========================================
# STEP 4: BUSINESS-DRIVEN MARKET SEGMENTATION
# ==========================================
print("\n🎯 Step 4: Clustering listings using business logic (Proximity to Colosseum)...")

# 1. Coordinate definitions for the Colosseum (Tourism Hub)
COLOSSEUM_LAT = 41.8902
COLOSSEUM_LON = 12.4922

# 2. Calculate spatial Euclidean distance to the historical core
df_geo['dist_to_center'] = np.sqrt(
    (df_geo['latitude'] - COLOSSEUM_LAT)**2 + 
    (df_geo['longitude'] - COLOSSEUM_LON)**2
)

# 3. Apply rule-based grouping logic to define 5 macro business clusters
def assign_business_cluster(row):
    if row['dist_to_center'] <= 0.015:
        return 0  # Zone 0: Premium Historical Center
    elif row['dist_to_center'] <= 0.04:
        return 1  # Zone 1: Urban Core Ring (Trastevere, Vatican, etc.)
    else:
        # Segment the remaining periphery into 3 geographic sectors
        if row['latitude'] > COLOSSEUM_LAT and row['longitude'] > COLOSSEUM_LON:
            return 2  # Zone 2: Northeast Periphery
        elif row['latitude'] <= COLOSSEUM_LAT and row['longitude'] > COLOSSEUM_LON:
            return 3  # Zone 3: Southeast Periphery
        else:
            return 4  # Zone 4: West Periphery / Infrastructure Corridor

df_geo['geo_cluster'] = df_geo.apply(assign_business_cluster, axis=1)

# 4. Plot the Strategic Business Zones
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_geo, 
    x='longitude', 
    y='latitude', 
    hue='geo_cluster', 
    palette='Set1', 
    alpha=0.6, 
    s=15
)
# Highlight the Colosseum on the map
plt.scatter(COLOSSEUM_LON, COLOSSEUM_LAT, color='black', marker='X', s=250, label='The Colosseum')

plt.title('Rome Airbnb Market Segmentation via Proximity to Tourism Core', fontsize=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Business Zones', labels=[
    'Zone 0: Premium Historical Center',
    'Zone 1: Urban Core Ring',
    'Zone 2: Northeast Periphery',
    'Zone 3: Southeast Periphery',
    'Zone 4: West Periphery',
    'The Colosseum'
])
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('roma_business_segments.png', dpi=300)
plt.close()
print("💾 Saved plot: 'roma_business_segments.png'")


# ==========================================
# STEP 5: BUSINESS IMPLICATIONS AND PROFILING
# ==========================================
print("\n📈 Step 5: Profiling the business performance of each zone...")

# Group by the new strategic cluster and aggregate metrics
cluster_profile = df_geo.groupby('geo_cluster').agg(
    total_listings=('id', 'count'),
    mean_price=('price', 'mean'),
    median_price=('price', 'median'),
    total_reviews=('number_of_reviews', 'sum'),
    avg_reviews_per_listing=('number_of_reviews', 'mean')
).reset_index()

print("\n📊 --- STRATEGIC BUSINESS ZONE PROFILES ---")
print(cluster_profile.to_string(index=False))

print("\n✅ Step 4 & 5 Success: Market segmentation script fully executed!")