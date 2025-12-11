# knowledge_base.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from A_model_loader import model_manager
from fuzzywuzzy import fuzz
import re

embedder = model_manager.get_embedder()
client = model_manager.get_chroma_client()


# Load your CSVs
df_property = pd.read_csv("dataset_clean/VoiceBot_Dataset.csv")
df_locality = pd.read_csv("dataset_clean/Locality_Info.csv")

print(f"✅ Loaded {len(df_property)} property records")
print(f"✅ Loaded {len(df_locality)} locality records")

# ========================
# Mapping Short Abbreviations
PRIMARY_USE_MAP = {
    "Res": "Residential Area",
    "PremRes": "Premium Residential Area",
    "Comm": "Commercial Zone",
    "Mixed": "Mixed Residential & Commercial Zone",
    "EduComm": "Educational + Commercial Cluster",
    "Indus": "Industrial Zone"
}


INVEST_FOCUS_MAP = {
    "StableRental": "Stable and consistent rental demand",
    "HighRental": "High rental returns and strong tenant demand",
    "ResGrowth": "Strong residential growth potential",
    "EmergingRes": "Emerging residential micro-market",
    "MidRes": "Mid-range, balanced residential locality",
    "FutureGrowth": "Future-ready high growth corridor",
    "Premium": "Premium high-value residential zone",
    "CommHub": "Major commercial and office hub",
    "Corporate": "Corporate business district with office clusters",
    "Established": "Well-developed, mature residential locality",
    "EduHub": "Education-focused neighbourhood with institutions",
    "GovtFocus": "Government-supported development zone",
    "IndusZone": "Industrial and manufacturing zone",
    "Logistic": "Strategic logistical and transport hub",
    "Heritage": "Historic and cultural heritage locality"
}


GEO_QUADRANT_MAP = {
    "West": "West Ahmedabad",
    "WestCent": "West-Central Ahmedabad",
    
    "SW": "South-West Ahmedabad",
    "South West": "South-West Ahmedabad",
    
    "CentSouth": "Central-South Ahmedabad",
    "Cent": "Central Ahmedabad",
    "Central": "Central Ahmedabad",
    
    "North": "North Ahmedabad",
    "NorthWest": "North-West Ahmedabad",
    "NW": "North-West Ahmedabad",
    "North West": "North-West Ahmedabad",
    
    "East": "East Ahmedabad"
}

# ========================

# ChromaDB setup --> all main model loading done in A_model_loader.py file so from there we will get the chroma client
collection = client.get_or_create_collection(name="ahmedabad_real_estate")


# ============================================================================
# BUILD KNOWLEDGE BASE
# ============================================================================

def expand_locality_csv_row(row):
    """Convert short CSV codes into human-friendly meaning."""
    
    primary = PRIMARY_USE_MAP.get(row.get("Primary_Use"), row.get("Primary_Use", "Unknown"))
    invest = INVEST_FOCUS_MAP.get(row.get("Invest_Focus"), row.get("Invest_Focus", "Unknown"))
    geo = GEO_QUADRANT_MAP.get(row.get("Geo_Quadrant"), row.get("Geo_Quadrant", "Unknown"))
    
    return {
        "Primary_Use_Expanded": primary,
        "Invest_Focus_Expanded": invest,
        "Geo_Quadrant_Expanded": geo
    }

def build_knowledge_texts():
    """Convert CSV data into searchable text chunks"""
    
    texts = []
    metadata = []
    
    # 1. Property Price Information
    for idx, row in df_property.iterrows():
        # Calculate price in lakhs from PSF and size
        min_price_lakh = (row['Min_Price_PSF'] * row['Avg_Size_SqFt']) / 100000
        max_price_lakh = (row['Max_Price_PSF'] * row['Avg_Size_SqFt']) / 100000
        avg_price_lakh = row['apt_price_lpa']  # This is already in lakhs per apartment
        sqft_price = (row['Min_Price_PSF'] + row['Max_Price_PSF']) / 2  # keeping this way Avg sqft_price
        
        # Main price chunk
        text = f"A {row['Property_Type']} in {row['Locality_Name']} area of Ahmedabad "
        text += f"costs between ₹{min_price_lakh:.2f} lakhs to ₹{max_price_lakh:.2f} lakhs. "
        text += f"The average price is ₹{avg_price_lakh} lakhs. "
        text += f"Average rent is ₹{row['Avg_Rent']} per month. "
        text += f"Price trend is {row['Price_Trend_YoY']}% year-on-year. "
        text += f"Average size is {row['Avg_Size_SqFt']} sq ft. "
        text += f"Price per sq ft is ₹{(row['Min_Price_PSF'] + row['Max_Price_PSF']) / 2:.0f}."
        
        texts.append(text)
        metadata.append({
            'type': 'property_price',
            'locality': row['Locality_Name'],
            'property_type': row['Property_Type'],
            'min_price': min_price_lakh,
            'max_price': max_price_lakh,
            'avg_price': avg_price_lakh
        })

        # RENT chunks (NEW)
        rent_text = (
            f"In {row['Locality_Name']}, the average monthly rent for a "
            f"{row['Property_Type']} is ₹{row['Avg_Rent']}."
        )
        
        texts.append(rent_text)
        metadata.append({
             "type": "property_rent",
             "locality": row["Locality_Name"],
             "property_type": row["Property_Type"],
             "avg_rent": row["Avg_Rent"]
        })
        

        # Alternative phrasing for better matching
        alt_text = f"In {row['Locality_Name']}, a {row['Property_Type']} is priced at approximately ₹{avg_price_lakh} lakhs."
        texts.append(alt_text)
        metadata.append({
            'type': 'property_price_short',
            'locality': row['Locality_Name'],
            'property_type': row['Property_Type'],
            'avg_price': avg_price_lakh
        })
    
    # 2. Locality Information - Fix column names
    for idx, row in df_locality.iterrows():
        expanded = expand_locality_csv_row(row)
        
        text = f"{row['Locality_Name']} is a {expanded['Primary_Use_Expanded']} area in Ahmedabad. "
        text += f"It has a connectivity rating of {row['Connect_Rate']}/5. "
        text += f"Key landmark: {row['Key_Landmark']}. "
        text += f"Nearest metro station: {row['Nearest_Metro_Station']}."
        
        texts.append(text)
        metadata.append({
            'type': 'locality_info',
            'locality': row['Locality_Name'],
            'primary_use': row['Primary_Use'],
            'connectivity': row['Connect_Rate']
        })
    
    # 3. Comparison chunks (for "cheapest 2BHK" type queries)
    for prop_type in df_property['Property_Type'].unique():
        prop_data = df_property[df_property['Property_Type'] == prop_type].nsmallest(5, 'apt_price_lpa')
        
        if len(prop_data) > 0:
            text = f"For {prop_type} properties in Ahmedabad, affordable options are: "
            options = []
            for _, row in prop_data.iterrows():
                options.append(f"{row['Locality_Name']} (₹{row['apt_price_lpa']} lakhs)")
            text += ", ".join(options) + "."
            
            texts.append(text)
            metadata.append({
                'type': 'comparison',
                'property_type': prop_type,
                'query_intent': 'affordable'
            })
    
    return texts, metadata


# Initialize vector DB
if collection.count() == 0:
    print("🔄 Building vector database...")
    knowledge_texts, knowledge_metadata = build_knowledge_texts()
    
    print(f"🔄 Generating embeddings for {len(knowledge_texts)} chunks...")
    embeddings = embedder.encode(knowledge_texts, show_progress_bar=True).tolist()
    
    print("🔄 Storing in ChromaDB...")
    collection.add(
        ids=[f"doc_{i}" for i in range(len(knowledge_texts))],
        embeddings=embeddings,
        documents=knowledge_texts,
        metadatas=knowledge_metadata
    )
    print(f"✅ Vector DB created with {len(knowledge_texts)} documents")
else:
    print(f"✅ Vector DB loaded ({collection.count()} documents)")



#=============================================================================
# forming functions for basically doing dynamic retrieval for queries --> basically for comparison based queries of comparing 2-3 localities & all
#=============================================================================
def extract_multiple_localities(query, all_localities):
    """Return ALL locality names mentioned in a query with fuzzy matching."""
    found = []
    for loc in all_localities:
        if loc.lower() in query.lower():
            found.append(loc)

    # If user says Thaltej, Bopal, Vastrapur → extract properly
    if ',' in query or 'and' in query:
        tokens = re.split(r'[,\sand]+', query)
        for t in tokens:
            loc = fuzzy_match_locality(t.strip())
            if loc and loc not in found:
                found.append(loc)

    return list(set(found))

# weighted scoring for each locality -----------------------------------
ROI_MAP = {
    'High': 3,
    'Medium-High': 2,
    'Medium': 1,
    'Low': 0
}

WEIGHTS = {
    'investment': 0.35,
    'roi': 0.20,
    'yield': 0.15,
    'appreciation': 0.20,
    'connect': 0.10
}

def compute_score(row):
    return (
        WEIGHTS['investment'] * row['Investment_Score'] +
        WEIGHTS['roi'] * ROI_MAP.get(row['ROI_Potential'], 1) +
        WEIGHTS['yield'] * row['Rental_Yield_Pct'] +
        WEIGHTS['appreciation'] * row['Appreciation_Rate'] +
        WEIGHTS['connect'] * row['Connect_Rate']
    )

# main multi locality comparison function ------------------------------
''' Works for 2, 3, 5, 10 localities
- User can say:
      - “Compare Thaltej vs Bopal”
      - “Which is better between Thaltej, Bopal, Gota, Vastrapur?”
      - “Rank these areas for investment: Bopal, Prahlad Nagar, SG Highway, Ghatlodia” '''

''' Can speak natural reasoning with real estate logic
-> ✔ Extensible
-> Add more fields later (crime score, water supply, commercial mix, etc.)
-> ✔ Transparent and auditable '''


def compare_localities_multi(localities):
    matched = [fuzzy_match_locality(l) for l in localities]
    matched = [m for m in matched if m]

    if len(matched) < 2:
        return {'found': False, 'reason': 'Need at least 2 valid localities'}

    comparison_details = []

    for loc in matched:
        row = df_locality[df_locality['Locality_Name'] == loc].iloc[0]
        score = compute_score(row)
        
        comparison_details.append({
            'name': loc,
            'investment_score': row['Investment_Score'],
            'roi': row['ROI_Potential'],
            'rental_yield': row['Rental_Yield_Pct'],
            'appreciation': row['Appreciation_Rate'],
            'connectivity': row['Connect_Rate'],
            'final_score': round(score, 2)
        })

    # Sort descending
    comparison_details.sort(key=lambda x: x['final_score'], reverse=True)

    # Winner
    winner = comparison_details[0]

    return {
        'found': True,
        'winner': winner,
        'all_localities': comparison_details,
        'response_type': 'comparison_multi'
    }



# ============================================================================
# RETRIEVAL FUNCTIONS
# ============================================================================

def search_csv_direct(locality, property_type, intent=None):
    """Direct CSV lookup with intent awareness - NON-AGGRESSIVE"""
    
    locality_clean = locality.lower().strip() if locality else None
    prop_type_clean = property_type.lower().strip() if property_type else None
    
    matches = df_property.copy()
    
    if locality_clean:
        matches = matches[matches['Locality_Name'].str.lower().str.strip() == locality_clean]
    
    if prop_type_clean:
        matches = matches[
            matches['Property_Type'].str.lower().str.contains(prop_type_clean, na=False) |
            matches['Property_Type'].str.lower().str.contains(prop_type_clean.replace(' apartment', ''), na=False)
        ]
    
    if len(matches) > 0:
        row = matches.iloc[0]
        
        # ✅ ALWAYS include ALL data - let formatter decide what to show
        min_price_lakh = (row['Min_Price_PSF'] * row['Avg_Size_SqFt']) / 100000
        max_price_lakh = (row['Max_Price_PSF'] * row['Avg_Size_SqFt']) / 100000
        
        result = {
            'found': True,
            'locality': row['Locality_Name'],
            'property_type': row['Property_Type'],
            'method': 'csv_direct',
            
            # ✅ ALWAYS include both price AND rent
            'min_price': round(min_price_lakh, 2),
            'max_price': round(max_price_lakh, 2),
            'avg_price': row['apt_price_lpa'],
            'price_trend': f"{row['Price_Trend_YoY']}% YoY",
            'avg_size': row['Avg_Size_SqFt'],
            'avg_rent': row['Avg_Rent'],
            
            # ✅ NEW: Pass intent hint (not enforcement)
            'intent_hint': intent  # Just a hint for formatter
        }
        
        return result
    
    return None

# ---- doing the Price Square foor code to get ---> square foot price
# Add to knowledge_base.py after search_csv_direct()

def get_price_per_sqft(locality, property_type):
    """Direct PSF lookup - bypasses vector search ambiguity"""
    
    locality_clean = locality.lower().strip() if locality else None
    prop_type_clean = property_type.lower().strip() if property_type else None
    
    matches = df_property.copy()
    
    if locality_clean:
        matches = matches[matches['Locality_Name'].str.lower() == locality_clean]
    
    if prop_type_clean:
        matches = matches[matches['Property_Type'].str.lower().str.contains(prop_type_clean, na=False)]
    
    if len(matches) > 0:
        row = matches.iloc[0]
        avg_psf = (row['Min_Price_PSF'] + row['Max_Price_PSF']) / 2
        
        return {
            'found': True,
            'locality': row['Locality_Name'],
            'property_type': row['Property_Type'],
            'min_psf': row['Min_Price_PSF'],
            'max_psf': row['Max_Price_PSF'],
            'avg_psf': round(avg_psf, 2),
            'method': 'psf_direct',
            'response_type': 'psf'  # Flag for formatter
        }
    
    return None

def format_psf_response(result):
    """Format price per square foot response for voice output"""
    
    locality = result['locality']
    property_type = result['property_type']
    min_psf = result['min_psf']
    max_psf = result['max_psf']
    avg_psf = result['avg_psf']
    
    # Natural language response
    response = f"In {locality}, the price per square foot for a {property_type} "
    response += f"ranges from ₹{min_psf:,.0f} to ₹{max_psf:,.0f}. "
    response += f"The average rate is ₹{avg_psf:,.0f} per square foot."
    
    return {
        'text': response,
        'data': result,
        'response_type': 'psf'
    }


def search_vector_db(query, top_k=3):
    """Semantic search in vector database"""
    
    query_embedding = embedder.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    if results['documents'][0]:
        return {
            'found': True,
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'best_match': results['documents'][0][0],
            'best_metadata': results['metadatas'][0][0],
            'confidence': 1 - results['distances'][0][0],
            'method': 'vector_search'
        }
    
    return None

def retrieve_property_info(query, entities, intent=None):
    """Hybrid retrieval with BETTER fallback logic"""

    print(f"   🔍 Searching knowledge base...")

    locality = entities.get('locality')
    property_type = entities.get('property_type')
    
    csv_result = None

    # 1. ✅ Try direct CSV lookup first (STRICT match on property_type)
    if locality and property_type:
        csv_result = search_csv_direct(locality, property_type, intent)
        if csv_result:
            print(f"   ✅ Found via CSV direct lookup")
            # ✅ ADD intent hint to result
            csv_result['intent_hint'] = intent
            return csv_result  # ✅ RETURN IMMEDIATELY - don't override with vector search
    
    # 2. ✅ Try with just locality (any property type) - ONLY if property_type not specified
    if not csv_result and locality and not property_type:
        csv_result = search_csv_direct(locality, None, intent)
        if csv_result:
            print(f"   ✅ Found via CSV (locality only)")
            csv_result['intent_hint'] = intent
            return csv_result

    # 3. Fallback to vector search (ONLY if CSV found nothing)
    vector_result = search_vector_db(query, top_k=5)
    
    if vector_result and vector_result['confidence'] > 0.55:
        print(f"   ✅ Found via vector search (confidence: {vector_result['confidence']:.2f})")
        return vector_result
    
    # 4. No confident match found
    print(f"   ⚠️  No confident match found")
    return None

# ---------------------------------------------
# fuzzy match locality info with attribute
# ---------------------------------------------
def fuzzy_match_locality(user_input, threshold=75):
    """
    Handle misspellings: "Satelite" → "Satellite"
    Returns: matched locality name or None
    """
    user_input_lower = user_input.lower().strip()
    
    # Get all unique localities from both datasets
    all_localities = pd.concat([
        df_property['Locality_Name'],
        df_locality['Locality_Name']
    ]).unique()
    
    best_match = None
    best_score = 0
    
    for locality in all_localities:
        score = fuzz.ratio(user_input_lower, locality.lower())
        if score > best_score:
            best_score = score
            best_match = locality
    
    if best_score >= threshold:
        print(f"   🔍 Fuzzy match score: {best_score} for '{user_input}' → '{best_match}'")
        return best_match
    
    return None

# LOCALITY INFO RETRIEVAL WITH ATTRIBUTE --------------------------------------------------------

def get_locality_info(locality_name, attribute='all'):  # ✅ Add attribute parameter
    
    # ✅ ADD FUZZY MATCHING
    matched_locality = fuzzy_match_locality(locality_name)
    
    if not matched_locality:
        return {"found": False}
    
    # ✅ USE matched_locality instead of locality_name
    row = df_locality[df_locality['Locality_Name'] == matched_locality]
    
    if row.empty:
        return {"found": False}
    
    row = row.iloc[0]
    expanded = expand_locality_csv_row(row)
    
    # ✅ ADD ATTRIBUTE-SPECIFIC RETURNS
    if attribute == 'metro':
        return {
            "found": True,
            "locality": row["Locality_Name"],
            "metro": row["Nearest_Metro_Station"],
            "response_type": "specific"
        }
    
    elif attribute == 'connectivity':
        return {
            "found": True,
            "locality": row["Locality_Name"],
            "connectivity": row["Connect_Rate"],
            "response_type": "specific"
        }
    
    elif attribute == 'landmark':
        return {
            "found": True,
            "locality": row["Locality_Name"],
            "landmark": row["Key_Landmark"],
            "response_type": "specific"
        }
    
    # Default: return all info (existing behavior)
    return {
        "found": True,
        "locality": row["Locality_Name"],
        "primary_use": expanded["Primary_Use_Expanded"],
        "invest_focus": expanded["Invest_Focus_Expanded"],
        "geo_quadrant": expanded["Geo_Quadrant_Expanded"],
        "landmark": row["Key_Landmark"],
        "metro": row["Nearest_Metro_Station"],
        "connectivity": row["Connect_Rate"]
    }


# Add this at the very end of knowledge_base.py

def rebuild_vector_db():
    """Force rebuild with rent-optimized chunks"""
    
    print("\n🔄 REBUILDING VECTOR DATABASE...")

    # 1. Delete all existing data
    try:
        existing_ids = collection.get()['ids']
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"✅ Deleted {len(existing_ids)} old documents")
    except:
        print("⚠️  No existing documents to delete")

    # 2. Build new knowledge base
    knowledge_texts, knowledge_metadata = build_knowledge_texts()
    print(f"📊 Generated {len(knowledge_texts)} text chunks")

    # 3. Generate embeddings
    print("🔄 Generating embeddings (this may take 30–60 seconds)...")
    embeddings = embedder.encode(knowledge_texts, show_progress_bar=True).tolist()

    # 4. Store in ChromaDB
    print("💾 Storing in ChromaDB...")
    collection.add(
        ids=[f"doc_{i}" for i in range(len(knowledge_texts))],
        embeddings=embeddings,
        documents=knowledge_texts,
        metadatas=knowledge_metadata
    )

    print(f"✅ Vector DB rebuilt with {len(knowledge_texts)} documents")

    # ---------------------------------------------------------
    # 5. Test rent query (SAFE)
    # ---------------------------------------------------------
    print("\n🧪 Testing rent query...")
    test_query = "monthly rent for 2 BHK in Satellite"
    test_embedding = embedder.encode(test_query).tolist()

    # NO WHERE FILTER → safe on all DBs
    results = collection.query(
        query_embeddings=[test_embedding],
        n_results=3
    )

    # Safety checks before accessing results
    if (not results.get("documents") or
        not results["documents"][0] or
        len(results["documents"][0]) == 0):
        
        print("⚠️ No vector results found for rent query.")
        return

    print("\nTop result:")
    print(results["documents"][0][0][:150], "...")

    print("\nMetadata:")
    print(results["metadatas"][0][0])

    print("\nConfidence Score:", 1 - results["distances"][0][0])


