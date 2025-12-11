# intent_router.py

from groq import Groq
import re
import json
import os
from knowledge_base import (
    retrieve_property_info, 
    get_locality_info,
    df_property,
    df_locality,
    fuzzy_match_locality,
    extract_multiple_localities,  
    compare_localities_multi      
)
from difflib import get_close_matches

# Initialize Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================================
# ENTITY EXTRACTION
# ============================================================================

COMPARISON_KEYWORDS = [
    'compare', 'vs', 'versus', 'better', 'between', 
    'which is best', 'rank', 'which area', 'comparison',
    'investment comparison', 'should i choose', 'investment', 'investment wise', 'investment options',
    'which locality', 'which location', 'which area is better', 'which location is better', 'which is better'
]

def extract_locality_attributes(query):
    """
    Extract WHAT aspect of locality user wants
    """
    query_lower = query.lower()
    
    # Map keywords to CSV column names
    attribute_map = {
        'metro': ['metro', 'station', 'nearest metro'],
        'connectivity': ['connectivity', 'connected', 'connect rate'],
        'landmark': ['landmark', 'famous place', 'key landmark'],
        'primary_use': ['residential', 'commercial', 'type of area'],
        'invest_focus': ['investment', 'growth', 'rental demand']
    }
    
    for attribute, keywords in attribute_map.items():
        if any(kw in query_lower for kw in keywords):
            return attribute
    
    return 'all'  # Default: return everything

def extract_entities_regex(query):
    """Extract property type and locality using regex"""
    
    query_lower = query.lower()
    
    # ✅ ENHANCED BHK EXTRACTION
    property_type = None
    
    # Pattern 1: "2 bhk", "2bhk", "two bhk"
    bhk_match = re.search(r'(\d+|one|two|three|four)\s*bhk', query_lower)
    if bhk_match:
        bhk_num = bhk_match.group(1)
        # Convert word to number
        word_to_num = {'one': '1', 'two': '2', 'three': '3', 'four': '4'}
        bhk_num = word_to_num.get(bhk_num, bhk_num)
        property_type = f"{bhk_num} BHK Apartment"
    
    # ✅ Pattern 2: "house", "flat", "apartment" without BHK
    if not property_type:
        if any(word in query_lower for word in ['house', 'flat', 'apartment']):
            property_type = "Apartment"  # Generic fallback
    
    # Extract locality from known areas
    known_localities = [
        'satellite', 'bopal', 'sg highway', 'bodakdev', 'vastrapur',
        'prahlad nagar', 'prahladnagar', 'thaltej', 'maninagar', 
        'navrangpura', 'paldi', 'chandkheda', 'gota', 'naranpura',
        'science city', 'shilaj', 'naroda', 'ghatlodia', 'sarkhej',
        'ellis bridge', 'ashram road'
    ]
    
    locality = None
    # ✅ SORT BY LENGTH (longest first to avoid partial matches)
    known_localities.sort(key=len, reverse=True)
    
    for area in known_localities:
        if area in query_lower:
            # Capitalize properly
            locality = area.title()
            if locality == 'Sg Highway':
                locality = 'SG Highway'
            elif locality == 'Prahladnagar':
                locality = 'Prahlad Nagar'
            break
    
    return {
        'property_type': property_type,
        'locality': locality,
        'method': 'regex'
    }




def extract_entities_groq(query):
    """Fallback: Use Groq for complex extraction with typo tolerance"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "system",
                "content": """You extract property details from Indian English queries with typos.

Common localities (match even with typos):
- Satellite, Bopal, SG Highway, Bodakdev, Vastrapur, Thaltej, Maninagar, 
  Prahlad Nagar, Navrangpura, Paldi, Chandkheda, Gota, Science City

Return ONLY JSON: {"property_type": "X BHK Apartment", "locality": "Name"}
If unclear, use null. NO other text."""
            }, {
                "role": "user",
                "content": f'Extract from: "{query}"'
            }],
            temperature=0.0,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        result = re.sub(r'```json\s*|\s*```', '', result)
        
        json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
        
        if json_match:
            entities = json.loads(json_match.group())
            
            # ✅ Apply fuzzy matching on extracted locality
            if entities.get('locality'):
                from knowledge_base import fuzzy_match_locality
                matched = fuzzy_match_locality(entities['locality'], threshold=75)
                if matched:
                    entities['locality'] = matched
                    print(f"   ✅ Groq+Fuzzy: '{entities['locality']}' → '{matched}'")
            
            entities['method'] = 'groq'
            return entities
            
    except Exception as e:
        print(f"   ⚠️  Groq extraction error: {e}")
    
    return {'property_type': None, 'locality': None, 'method': 'failed'}



def extract_entities(query):
    """Hybrid entity extraction with multi-locality support + fuzzy + Groq."""

    # -----------------------------------------------------------------------------
    # INITIAL STRUCTURE (EXTENDED)
    # -----------------------------------------------------------------------------
    entities = {
        'locality': None,          # single locality
        'localities': [],          # multiple localities (comparison)
        'property_type': None,
        'budget': None,
        'method': None,
        'locality_attribute': 'all'
    }

    query_lower = query.lower().strip()

    # -----------------------------------------------------------------------------
    # LOAD ALL LOCALITIES FROM BOTH DATASETS
    # -----------------------------------------------------------------------------
    try:
        import pandas as pd
        from knowledge_base import df_property, df_locality, fuzzy_match_locality
    except Exception as e:
        print("Error loading datasets:", e)
        return entities

    all_localities = pd.concat([
        df_property['Locality_Name'],
        df_locality['Locality_Name']
    ]).dropna().unique().tolist()

    # -----------------------------------------------------------------------------
    # 1️⃣ MULTI-LOCALITY DETECTION (NEW)
    # -----------------------------------------------------------------------------
    from knowledge_base import extract_multiple_localities

    comp = detect_comparison_intent(query)

    # A) for comparison intent made
    if comp:
        # User wrote: "compare bopal vs thaltej vs vastrapur"
        detected = extract_multiple_localities(query, all_localities)

        if len(detected) >= 2:
            entities['localities'] = detected
            entities['method'] = 'multi_locality'
            print(f"   ✅ Multi-locality detected: {detected}")
            return entities  # STOP HERE → skip normal single locality flow
        

    # ADD: Check for multiple localities in PSF queries --> like if query comes for psf sq ft price of 2bhk in Navrangpura & in Paldi -> but if like they say instead of [&] they say [ or also ]
    if not entities['localities']:
        connectors = [" & ", " and ", " or ", " or also ", ",", "/"]

    if any(connector in query_lower for connector in connectors):
        detected = extract_multiple_localities(query, all_localities)
        if len(detected) >= 2:
            entities['localities'] = detected
            entities['locality'] = None  # Clear single locality
            entities['method'] = 'multi_locality'
            print(f"   ✅ Multi-locality detected in PSF query: {detected}")

    # -----------------------------------------------------------------------------
    # 2️⃣ EXISTING: FIRST PASS (Regex-based single locality extraction)
    # -----------------------------------------------------------------------------
    for loc in all_localities:
        if loc.lower() in query_lower:
            entities['locality'] = fuzzy_match_locality(loc)
            entities['method'] = 'regex'
            print(f"   ✅ Regex locality matched → {entities['locality']}")
            break

    # -----------------------------------------------------------------------------
    # 3️⃣ FUZZY MATCH if single locality still missing (OLD LOGIC)
    # -----------------------------------------------------------------------------
    if not entities['locality']:
        words = query_lower.split()

        for word in words:
            if len(word) <= 3:
                continue

            matched = fuzzy_match_locality(word, threshold=75)

            if matched:
                entities['locality'] = matched
                entities['method'] = 'fuzzy'
                print(f"   ✅ Fuzzy matched locality: '{word}' → '{matched}'")
                break

    # -----------------------------------------------------------------------------
    # 4️⃣ PROPERTY TYPE EXTRACTION (unchanged)
    # -----------------------------------------------------------------------------
    property_keywords = {
        "1 bhk": "1 BHK", "1bhk": "1 BHK",
        "2 bhk": "2 BHK", "2bhk": "2 BHK",
        "3 bhk": "3 BHK", "3bhk": "3 BHK",
        "flat": "flat", "apartment": "flat", "house": "house",
        "villa": "villa"
    }

    for k, v in property_keywords.items():
        if k in query_lower:
            entities['property_type'] = v
            print(f"   🏠 Property type detected: {v}")
            break


    # -----------------------------------------------------------------------------
    # 5️⃣ CALL GROQ IF ANYTHING STILL MISSING (OLD HYBRID LOGIC of calling groq if something goes wrong )
    # -----------------------------------------------------------------------------
    if not entities['property_type'] or not entities['locality']:
        print("   → Incomplete extraction, calling Groq...")

        try:
            groq_entities = extract_entities_groq(query)

            if not entities['property_type'] and groq_entities.get('property_type'):
                entities['property_type'] = groq_entities['property_type']

            if not entities['locality'] and groq_entities.get('locality'):
                # fuzzy correct Groq locality output
                corrected = fuzzy_match_locality(groq_entities['locality'])
                if corrected:
                    entities['locality'] = corrected

            entities['method'] = 'hybrid'

        except Exception as e:
            print("Groq extraction failed:", e)

    return entities


# ============================================================================
# INTENT DETECTION
# ============================================================================

INTENT_PRIORITY = {
    'comparison': 0,      # Highest - user comparing multiple things --> highest because it's multi-dimensional (needs multiple data points)
    'price': 1,           # Most common - buying     -> high bcos they're transactional
    'rent': 2,            # Second most common
    'psf_price': 3,       # Specialized price query
    'property_size': 4,   # Area/size info
    'locality_info': 5,   # Area details
    'availability': 6,    # Stock/availability
    'greeting': 7,        # Casual
    'unknown': 99
}

CUSTOM_CORRECTIONS = {
    "prize": "price",
    "prise": "price",
    "prce": "price",
    "pice": "price",
    "prie": "price",
    
    "ren": "rent",
    "rente": "rent",
    "rantal": "rental",

    "lokation": "location",
    "locaton": "location",
    "conectivity": "connectivity",

    "sqft": "square feet",
    "sq ft": "square feet",
    "sqt": "square feet",
    "skft": "square feet",
    "square fit": "square feet",
    "squar feet": "square feet",
    "per sk fit": "per square feet",
    "psf": "per square feet",
    "rate per": "per square feet",
    "per square": "per square feet",
}

# fuzzy fallback dictionary
FUZZY_WORDS = [
    "price", "rent", "availability", "location", "connectivity",
    "metro", "safety", "area", "landmark", "property", "bhk"
]

# intent classification for price per square feet
PSF_KEYWORDS = [
    'per square', 'psf', 'sq ft price', 'square feet price', 'rate per',
    'per sq ft', 'per sqft', 'square foot rate', 'square feet rate', 
    'price per square', 'rate per square', 'per square foot'          
]

def correct_spelling(query):
    """Apply custom corrections + fuzzy matching."""
    words = query.split()
    corrected = []

    for w in words:
        wl = w.lower()

        # Custom dictionary correction
        if wl in CUSTOM_CORRECTIONS:
            corrected.append(CUSTOM_CORRECTIONS[wl])
            continue

        # Fuzzy correction for STT noise
        close = get_close_matches(wl, FUZZY_WORDS, n=1, cutoff=0.82)
        if close:
            corrected.append(close[0])
        else:
            corrected.append(w)

    return " ".join(corrected).lower()

def detect_psf_query(query):
    """Detect if user wants price per sq ft"""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PSF_KEYWORDS)

def detect_comparison_intent(query):
    """Detect if user wants to compare multiple localities"""
    query_lower = query.lower()
    
    # Check for comparison keywords
    has_comparison_word = any(keyword in query_lower for keyword in COMPARISON_KEYWORDS)
    
    # Check for multiple localities (comma-separated or "and")
    has_multiple_locs = (',' in query or ' and ' in query_lower or ' or ' in query_lower)
    
    # Check for "X vs Y" pattern
    has_vs_pattern = bool(re.search(r'\w+\s+(vs|versus)\s+\w+', query_lower))
    
    return has_comparison_word or has_vs_pattern or (has_multiple_locs and has_comparison_word)


# ===============================================
# INTENT DETECTION LOGIC
# ===============================================

import re

def detect_intent(query):
    # This function assumes 'correct_spelling' and 'INTENT_PRIORITY' are defined globally or imported.
    # Placeholder for 'correct_spelling' (replace with your actual function)
    # def correct_spelling(q): return q 
    
    if not query or not query.strip():
        return {'intent': 'unknown', 'confidence': 0.0}

    # Clean & correct transcription errors
    query_lower = correct_spelling(query.lower())


    # ------------------
    # KEYWORD GROUPS
    # ------------------
    price_keywords = ["price", "cost", "rate", "worth", "expensive", "how much", "pricing", "charges"]
    rent_keywords = ["rent", "rental", "lease", "monthly rent"]
    locality_keywords = ["location", "connectivity", "landmark", "metro", "area", "how is", "is it good", "safe", "safety", "tell me about", "describe"]
    availability_keywords = ["available", "availability", "vacant", "stock"]
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening"]
    comparison_keywords = ['compare', 'vs', 'versus', 'better', 'between','which is best', 'rank', 'which area', 'comparison','investment comparison', 'should i choose','roi', 'return on investment', 'appreciation','growth potential', 'investment potential','best investment', 'best for investment','invest in', 'should i invest']
    # comparison_keywords ---> need to be in one line whole cos if space karke we put them down it create ... --> making list a tuple as empty space gets created
    PSF_KEYWORDS = ['per square', 'psf', 'sq ft price', 'square feet price', 'rate per', 'per sq ft', 'per sqft', 'square foot rate', 'square feet rate', 'price per square', 'rate per square', 'per square foot']
    size_keywords = ["size", "area", "square feet", "sqft", "sq ft", "how big", "square foot"]

    # -----------------------------
    # CONSOLIDATED INTENT MATCHING
    # -----------------------------
    matched = {}  # Use a dictionary to store unique intents and their highest confidence
    
    has_rent = any(k in query_lower for k in rent_keywords)
    has_price = any(k in query_lower for k in price_keywords)
    has_bhk = re.search(r'\d+\s*bhk', query_lower)

    # 1. Price/Rent (Highest Priority)
    # This consolidated block ensures 'price' or 'rent' is matched only once, with the highest confidence.
    # ✅ IMPROVED: Detect mixed intent for "price and rent" queries
    if has_rent and has_price:
        matched['price'] = 0.90  # Both mentioned
        matched['rent'] = 0.90
        # This will trigger 'mixed' intent below
    elif has_rent and not has_price:
        matched['rent'] = 0.95  # Clear rent focus
    elif has_price:
        matched['price'] = 0.92  # Clear price focus
    elif has_bhk:
        matched['price'] = 0.89  # Implied price query

    # 2. Locality info
    if any(k in query_lower for k in locality_keywords):
        matched['locality_info'] = 0.88

    # ✅ PSF detection - HIGHEST PRIORITY for price queries
    if any(k in query_lower for k in PSF_KEYWORDS):
        matched['psf_price'] = 0.98  # Higher than regular price
        # Remove regular price if PSF is detected
        if 'price' in matched:
            matched.pop('price')

    # one more add for area of houses basically
    if any(k in query_lower for k in size_keywords):
        matched['property_size'] = 0.93

    # 3. Availability
    if any(k in query_lower for k in availability_keywords):
        matched['availability'] = 0.85

    # 4. Comparison
    if any(k in query_lower for k in comparison_keywords):
        matched['comparison'] = 0.90
        
    # 5. Greeting (Pure Greeting Check)
    if any(g in query_lower for g in greeting_keywords):
        # Only treat as a pure greeting if it doesn't contain common property keywords
        if not any(t in query_lower for t in ["bhk", "price", "rent", "location", "property"]):
            matched['greeting'] = 1.0


    # 6.diambiguity resolution between property size and locality info ---> so like area related queries be of house size area but also locality as area so mix up na ho for that this if case checks on disambiguity
    if 'property_size' in matched and 'locality_info' in matched:
        # Check context to determine which "area" they mean
        
        # Size query indicators
        size_context = ['sqft', 'sq ft', 'square feet', 'how big', 'size of', 'area of']
        # Locality query indicators  
        locality_context = ['tell me about', 'how is', 'describe', 'about the area', 'good area']
        
        has_size_context = any(ctx in query_lower for ctx in size_context)
        has_locality_context = any(ctx in query_lower for ctx in locality_context)
        
        if has_size_context and not has_locality_context:
            # Clear size query - remove locality_info
            matched.pop('locality_info', None)
            matched['property_size'] = 0.95  # Boost confidence

        elif has_locality_context and not has_size_context:
            # Clear locality query - remove property_size
            matched.pop('property_size', None)
            matched['locality_info'] = 0.95  # Boost confidence
        else:
        # ✅ Ambiguous - default to property_size if BHK mentioned, else locality_info
            if has_bhk:
                matched.pop('locality_info', None)
                matched['property_size'] = 0.94
            else:
                matched.pop('property_size', None)
                matched['locality_info'] = 0.94

    # ---------------------------------------------------
    # DECISION LOGIC (single or multi-intent resolution)
    # ---------------------------------------------------
    final_matched = list(matched.items())

    if not final_matched:
        return {'intent': 'unknown', 'confidence': 0.0}

    # Sort by priority 
    final_matched.sort(key=lambda x: (INTENT_PRIORITY[x[0]], -x[1]))

    # New Detect combinations
    if(len(final_matched) >= 2):
        intent_names = [x[0] for x in final_matched]
    
    # Combo 1: Price + Rent (both asked together)
        if 'price' in intent_names and 'rent' in intent_names:
            
            return {
                'intent': 'mixed',
                 'primary': 'price',
                 'secondary': 'rent',
                 'confidence': 0.90,
                 'combo_type': 'price_rent'
                }
        
        # Combo 2: Price + Size
        if 'price' in intent_names and 'property_size' in intent_names:
            
            return {
            'intent': 'mixed',
            'primary': 'price',
            'secondary': 'property_size',
            'confidence': 0.88,
            'combo_type': 'price_size'
        }
    
        # Combo 3: PSF + Size
        if 'psf_price' in intent_names and 'property_size' in intent_names:
            
            return {
            'intent': 'mixed',
            'primary': 'psf_price',
            'secondary': 'property_size',
            'confidence': 0.87,
            'combo_type': 'psf_size'
        }
        
        # Combo 4: Rent + Size
        if 'rent' in intent_names and 'property_size' in intent_names:
            return {
        'intent': 'mixed',
        'primary': 'rent',
        'secondary': 'property_size',
        'confidence': 0.86,
        'combo_type': 'rent_size'
        }

    # Single intent or generic mixed
    if len(final_matched) == 1:
        intent, conf = final_matched[0]
        return {'intent': intent, 'confidence': conf}

    # Generic mixed (fallback)
    return {
        'intent': 'mixed',
        'primary': final_matched[0][0],
        'secondary': final_matched[1][0] if len(final_matched) > 1 else None,
        'confidence': final_matched[0][1]
    }
        

# ============================================================================
# RESPONSE HANDLERS
# ============================================================================

def format_price_response(data):
    """Format price/rent data - handles pure, mixed, and both intents"""
    
    intent_hint = data.get('intent_hint')
    
    # ✅ Handle vector search results first
    if data.get('method') == 'vector_search':
        best_meta = data.get('best_metadata', {})
        
        # If vector search found a rent-specific chunk
        if best_meta.get('type') == 'property_rent':
            return data['best_match']
        
        # Otherwise return the matched text
        return data['best_match']
    
    # ✅ Handle CSV results with flexible formatting
    if data.get('method') == 'csv_direct':
        locality = data['locality']
        prop_type = data['property_type']
        
        # CASE 1: Pure rent query → rent-focused response
        if intent_hint == 'rent':
            rent = data['avg_rent']
            response = f"The monthly rent for a {prop_type} in {locality} is ₹{rent:,}."
            
            # Optional: Add price as secondary info
            avg_price = data.get('avg_price')
            if avg_price:
                if avg_price >= 100:
                    price_str = f"₹{avg_price/100:.2f} Crores"
                else:
                    price_str = f"₹{avg_price} lakhs"
                response += f" If you're considering buying, the average price is {price_str}."
            
            return response
        
        # CASE 2: Mixed intent (both rent AND price) → show both equally
        elif intent_hint == 'mixed':
            avg_price = data.get('avg_price')
            rent = data['avg_rent']
            
            # Format price
            if avg_price >= 100:
                price_str = f"₹{avg_price/100:.2f} Crores"
            else:
                price_str = f"₹{avg_price} lakhs"
            
            response = f"For a {prop_type} in {locality}:\n"
            response += f"• Purchase price: {price_str}\n"
            response += f"• Monthly rent: ₹{rent:,}"
            
            # Add range
            min_p = data.get('min_price')
            max_p = data.get('max_price')
            if min_p and max_p:
                if max_p >= 100:
                    range_str = f"₹{min_p} lakhs to ₹{max_p/100:.2f} Crores"
                else:
                    range_str = f"₹{min_p} lakhs to ₹{max_p} lakhs"
                response += f"\n• Price range: {range_str}"
            
            return response
        
        # CASE 3: Pure price query → price-focused response
        else:  # intent_hint == 'price' or None
            avg_price = data.get('avg_price')
            
            if not avg_price:
                return "I found the property but pricing details are incomplete."
            
            # Format price
            if avg_price >= 100:
                price_str = f"₹{avg_price/100:.2f} Crores"
            else:
                price_str = f"₹{avg_price} lakhs"
            
            response = f"A {prop_type} in {locality} typically costs around {price_str}."
            
            # Add range
            min_p = data.get('min_price')
            max_p = data.get('max_price')
            
            if min_p and max_p:
                if max_p >= 100:
                    range_str = f"₹{min_p} lakhs to ₹{max_p/100:.2f} Crores"
                else:
                    range_str = f"₹{min_p} lakhs to ₹{max_p} lakhs"
                
                response += f" The price range is {range_str}."
            
            # Add trend
            if data.get('price_trend'):
                response += f" Price trend: {data['price_trend']}."
            
            # Optional: Mention rent as secondary info
            rent = data.get('avg_rent')
            if rent:
                response += f" Rental option available at ₹{rent:,}/month."
            
            return response
    
    return "Price information not available."

def format_comparison_response(result):
    """Format multi-locality comparison for voice output"""
    
    if not result.get('found'):
        return result.get('reason', "I couldn't compare those localities.")
    
    winner = result['winner']
    all_locs = result['all_localities']
    
    # Build natural voice response
    response = f"Based on my analysis, {winner['name']} is the best investment option "
    response += f"with a score of {winner['final_score']}. "
    
    # Explain why winner won
    response += f"\n\nIt has:\n"
    response += f"• Investment score: {winner['investment_score']}/10\n"
    response += f"• ROI potential: {winner['roi']}\n"
    response += f"• Rental yield: {winner['rental_yield']}%\n"
    response += f"• Appreciation rate: {winner['appreciation']}% per year\n"
    response += f"• Connectivity: {winner['connectivity']}/5"
    
    # Mention runner-ups
    if len(all_locs) > 1:
        runner_up = all_locs[1]
        response += f"\n\n{runner_up['name']} comes second with a score of {runner_up['final_score']}."
    
    # List all rankings if more than 2
    if len(all_locs) > 2:
        response += f"\n\nComplete ranking:\n"
        for i, loc in enumerate(all_locs, 1):
            response += f"{i}. {loc['name']} (Score: {loc['final_score']})\n"
    
    response += "\n\nWould you like detailed information about any specific area?"
    
    return response



def handle_greeting():
    return "Hello! I can help you with property prices, rental information, and locality details in Ahmedabad. How may I assist you?"

# Price query handler ==============================
def handle_price_query(query, entities):
    """Handle price queries"""
    
    print(f"🔍 Entities: {entities}")
    
    # ✅ DETECT intent here too
    intent_result = detect_intent(query)
    intent = intent_result.get('intent')
    
    if intent == 'mixed':
        intent = intent_result.get('primary') if not intent_result.get('is_both') else 'mixed'

    # -----------------------------------------------
    # ✅ 1st -->  Handle PSF / sq ft price queries HERE
    # -----------------------------------------------
    if detect_psf_query(query):
        print("📌 Detected PSF query")

        # --- Multi-locality PSF ---
        if entities.get('localities') and len(entities['localities']) >= 2:
            response_parts = []

            for loc in entities['localities']:
                result = get_price_per_sqft(loc, entities.get('property_type'))
                
                if result and result.get("found"):
                    psf_response = format_psf_response(result)
                    response_parts.append(
                        psf_response.get('response_text', '') 
                        if isinstance(psf_response, dict) 
                        else psf_response
                    )

            return "\n\n".join(response_parts) if response_parts else \
                "I couldn't find PSF data for those localities."

        # --- Single locality PSF ---
        result = get_price_per_sqft(
            entities.get('locality'),
            entities.get('property_type')
        )

        if result and result.get("found"):
            psf_response = format_psf_response(result)
            if isinstance(psf_response, dict):
                return psf_response.get(
                    'response_text',
                    f"PSF for {entities.get('locality')}: ₹{result.get('avg_psf', 'N/A')}"
                )
            return psf_response

        return "I couldn’t find PSF data for that locality."
    # -----------------------------------------------
    # Normal price Query flow showing below
    #-----------------------------------------------

    print("💰 Normal price/rent query detected (not PSF)")
    
    # Retrieve from knowledge base with intent
    result = retrieve_property_info(query, entities, intent=intent)

    if result and result.get("found"):
        # Hint for formatting (lets formatter decide LPA vs rent)
        result["intent_hint"] = intent
        
        response = format_price_response(result)

        # Append footer based on intent
        if intent == 'rent':
            response += "\n\nInterested in renting? Contact our office for viewings."
        elif intent == 'mixed':
            response += "\n\nFor rentals or purchase inquiries, contact our office."
        else:
            response += "\n\nFor more details or to schedule a visit, reach out to our team."

        return response
    
    return "I couldn't find pricing information for that property. Please contact our office for assistance & detailed information."    

# Locality info handler ==============================

def handle_locality_query(query, entities):
    
    locality = entities.get("locality")
    attribute = entities.get("locality_attribute", "all")
    
    if not locality:
        return "Could you please specify which locality you'd like to know about?"
    
    # Get CSV data
    info = get_locality_info(locality, attribute=attribute)
    
    if not info["found"]:
        return "I don't have detailed information about that locality. Could you try another area?"
    
    # ✅ Handle attribute-specific responses
    if attribute == "connectivity":
        csv_data = f"{info['locality']} has a connectivity rating of {info['connectivity']}/5."
        
        # ✅ ADD: Short Groq context (max 35 words)
        try:
            groq_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "system",
                    "content": "You provide 1-sentence summaries (max 35 words). No extra explanation."
                }, {
                    "role": "user",
                    "content": f"In exactly 25-35 words, why is {locality} Ahmedabad well-connected? Consider metro, highways, BRTS."
                }],
                max_tokens=60,  # ~35 words
                temperature=0.3
            )
            
            groq_context = groq_response.choices[0].message.content.strip()
            
            # Combine CSV + Groq
            return f"{csv_data} {groq_context}"
            
        except Exception as e:
            print(f"⚠️  Groq error: {e}")
            return csv_data  # Fallback to CSV only
    
    elif attribute == "metro":
        return f"The nearest metro station to {info['locality']} is {info['metro']}."
    
    elif attribute == "landmark":
        landmark = info['landmark']
        landmark_formatted = ''.join([' ' + c if c.isupper() else c for c in landmark]).strip()
        return f"The key landmark in {info['locality']} is {landmark_formatted}."
    
    # ✅ Default: CSV overview + short Groq insight
    csv_response = (
        f"{info['locality']} is a {info['primary_use']} locality in {info['geo_quadrant']}. "
        f"Connectivity: {info['connectivity']}/5. "
        f"Key landmark: {info['landmark']}. "
        f"Nearest metro: {info['metro']}."
    )
    
    # Add short Groq context
    try:
        groq_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "system",
                "content": "Provide 1 sentence (max 30 words) about area characteristics."
            }, {
                "role": "user",
                "content": f"In 25-30 words, what makes {locality} Ahmedabad popular for {info['invest_focus']}?"
            }],
            max_tokens=50,
            temperature=0.3
        )
        
        groq_insight = groq_response.choices[0].message.content.strip()
        return f"{csv_response} {groq_insight}"
        
    except:
        return csv_response
    
    


def handle_comparison_query(entities):
    """Handle multi-locality comparison queries"""
    
    localities = entities.get('localities', [])
    
    if len(localities) < 2:
        return "I need at least 2 localities to compare. Please mention areas like 'Satellite vs Bopal' or 'compare Thaltej, Vastrapur, and Bodakdev'."
    
    if len(localities) > 10:
        return f"That's too many areas to compare at once ({len(localities)}). Please limit to 5-7 localities for a meaningful comparison."
    
    print(f"🔍 Comparing localities: {localities}")
    
    # Call comparison function from knowledge_base.py
    result = compare_localities_multi(localities)
    
    if result.get('found'):
        return format_comparison_response(result)
    else:
        return result.get('reason', "I couldn't find comparison data for those localities.")
    
# Property size/area handler ==============================
def handle_size_query(entities):
    """Handle property size/area queries"""
    
    locality = entities.get('locality')
    prop_type = entities.get('property_type')
    
    if not locality:
        return "Please specify which locality you're asking about."
    
    # ✅ Query dataset for size
    from knowledge_base import df_property
    
    query_filter = df_property['Locality_Name'] == locality
    
    if prop_type:
        query_filter &= df_property['Property_Type'].str.contains(prop_type, case=False, na=False)
        results = df_property[query_filter]
        
        if not results.empty:
            avg_size = results['Avg_Size_SqFt'].mean()
            return f"The average size of a {prop_type} in {locality} is approximately {avg_size:,.0f} square feet."
    else:
        # ✅ Show all BHK types if not specified
        results = df_property[query_filter]
        
        if not results.empty:
            response = f"Average property sizes in {locality}:\n\n"
            for _, row in results.iterrows():
                prop = row['Property_Type']
                size = row['Avg_Size_SqFt']
                response += f"• {prop}: {size:,.0f} sq ft\n"
            
            return response.strip()
    
    return f"I don't have size data for {prop_type if prop_type else 'properties'} in {locality}."


# FUNCTION for basically wokring on returning top investments area when user just says city but no areas ------------------------------------
def handle_top_investment_areas():
    """Show top 3-5 investment areas from locality CSV"""
    try:
        from knowledge_base import df_locality
        
        # Sort by investment_score (assuming this column exists)
        top_areas = df_locality.nlargest(5, 'investment_score')[
            ['Locality_Name', 'investment_score', 'roi', 'appreciation', 'rental_yield']
        ]
        
        response = "Top investment areas in Ahmedabad:\n\n"
        
        for idx, row in top_areas.iterrows():
            response += f"{row['Locality_Name']}: "
            response += f"Investment score {row['investment_score']}/10, "
            response += f"ROI {row['roi']}, "
            response += f"Appreciation {row['appreciation']}%/year\n"
        
        response += "\nWould you like to compare any of these areas?"
        return response
        
    except Exception as e:
        print(f"Error fetching investment data: {e}")
        return "I can help you compare investment options. Please mention 2-3 specific areas like 'Compare Satellite vs Bopal for investment'."   
    

# Fallback handler ==================================


def handle_fallback(query, entities): 
    """Fallback handler with conversation history support"""

    print("⚠️ Intent unclear → attempting KB first...")

    # Always try knowledge base first
    kb = retrieve_property_info(query, entities)

    if kb and kb.get("found"):
        print("✅ Found via KB fallback")
        return format_price_response(kb)

    # Detect property query to block Groq
    property_keywords = [
        "price", "cost", "rent", "bhk", "flat",
        "apartment", "property", "locality", "area"
    ]
    is_property_query = any(k in query.lower() for k in property_keywords)

    if is_property_query:
        print("❌ Property query → Groq blocked")
        return (
            "I don't have that property in my database. "
            "Try asking about Satellite, Bodakdev, Bopal, SG Highway, "
            "Vastrapur or similar Ahmedabad localities."
        )

    # Allowed for general questions only
    print("→ Using Groq for generic query...")
    return handle_groq_general(query, groq_client)


# adding method just for needs_content 
def needs_context(query):
    """
    Check if query references previous conversation
    
    Returns True if query contains words like "that", "it", "there" etc.
    that indicate the user is referring to something from earlier.
    
    Args:
        query: User's current question
        
    Returns:
        bool: True if history is needed, False otherwise
    """
    context_keywords = [
        # Pronouns
        "that", "it", "there", "this", "these", "those",
        
        # Time references
        "previous", "earlier", "before", "last",
        
        # Direct references
        "you said", "you mentioned", "you told",
        
        # Follow-up indicators
        "tell me more", "what about", "how about", "and that",
        
        # Implicit references
        "the area", "the property", "the location",
        
        # Continuation words
        "same", "also", "too", "as well"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in context_keywords)


def handle_groq_general(query, grok_client= None, conversation_history=None):
    """Handle general questions with CSV context + short Groq response
    Args:
        query: User's question
        grok_client: Groq client instance
        conversation_history: List of past exchanges (optional)
    """

    if conversation_history is None:
        conversation_history = []  # ✅ Default to empty
    
    # Use the global client if none provided
    if grok_client is None:
        grok_client = groq_client
    
    # Check if client is available
    if grok_client is None:
        return "⚠️ API key required for this query. Please provide a Groq API key."
    
    # ✅ Try to extract locality for context
    from knowledge_base import fuzzy_match_locality
    
    query_lower = query.lower()
    words = query_lower.split()
    
    matched_locality = None
    for word in words:
        if len(word) > 4:
            matched = fuzzy_match_locality(word, threshold=70)
            if matched:
                matched_locality = matched
                break
    
    # ✅ If locality found, provide CSV context
    csv_context = ""
    if matched_locality:
        from knowledge_base import get_locality_info
        info = get_locality_info(matched_locality)
        
        if info.get('found'):
            csv_context = f"\n\nContext: {matched_locality} is a {info['primary_use']} area with {info['connectivity']}/5 connectivity. Key landmark: {info['landmark']}."
    
    try:
        # ✅ Build messages with conversation history
        messages = [
            {
                "role": "system",
                "content": """You are an Ahmedabad real estate assistant.
                
                RULES:
                - NEVER answer property prices or rent.
                - Keep answers SHORT: maximum 35-40 words.
                - If locality data is provided, acknowledge it.
                - Use conversation context naturally without mentioning memory.
                - Be concise and direct.
                """
            }
        ]
        
         # ✅ OPTIMIZATION 2 & 3: Conditional history with only last turn
        if conversation_history and needs_context(query):
            print("💬 Using conversation history (last turn)")
            for turn in conversation_history[-1:]:  # Only last 1 turn (saves ~50-80 tokens)
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["bot"]})
        else:
            print("⚡ Standalone query - skipping history")  # Saves ~60-120 tokens
        
        # ✅ Add current query
        messages.append({"role": "user", "content": query})
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=70,
            temperature=0.3
        )
        
        groq_response = response.choices[0].message.content.strip()
        
        # ✅ Combine with CSV context if available
        if csv_context:
            return groq_response + csv_context
        
        return groq_response

    except Exception as e:
        print("Groq error:", e)
        return "I'm here to help with Ahmedabad real estate. Please ask a question again."


# ============================================================================
# MAIN ROUTER
# ============================================================================

from knowledge_base import get_price_per_sqft, format_psf_response

def process_user_query(user_text, grok_api_key=None, is_local_client=False, conversation_history=None): #conversation_history=None --> optional defaults to none
    """
    Main query processor - FIXED VERSION
    - Handles price, rent, PSF, property size, comparisons, and general queries
    - Supports multi-intent combos (price+rent, price+size, psf+size, rent+size)
    - Uses Groq only if needed
    - also handles casual greetings and thanks + Conversation history placeholder --> when fallback on groq comes then
    """
    global groq_client

    if conversation_history is None:
        conversation_history = []  # Default to empty for single-turn calls

    # ----------------------------
    # 1️⃣ Groq API key handling
    # ----------------------------
    if grok_api_key and len(grok_api_key.strip()) > 10:
        groq_client = Groq(api_key=grok_api_key)
        print("🔑 Using user-provided Groq key")
    elif is_local_client and os.getenv("GROQ_API_KEY"):
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("💻 Local request → using backend .env key")
    else:
        return "⚠️ Please enter a valid Groq API key."

    # ----------------------------
    # 2️⃣ Logging
    # ----------------------------
    print("\n" + "="*70)
    print("📥 User:", user_text)
    print("="*70)

    text_lower = user_text.lower().strip()

    # ----------------------------
    # 3️⃣ Quick casual responses
    # ----------------------------
    if len(text_lower.split()) <= 3:
        if any(g in text_lower for g in ['hi', 'hello', 'hey']):
            return "Hello! I can help with property prices, rent, and area info. What would you like to know?"
        if 'how are you' in text_lower or 'how r you' in text_lower:
            return "I'm doing great! Ready to help with your real estate queries. What can I assist you with?"
        if any(t in text_lower for t in ['thanks', 'thank you', 'thankyou']):
            return "You're welcome! Anything else I can help with?"

    # ----------------------------
    # 4️⃣ Entity extraction
    # ----------------------------
    entities = extract_entities(user_text)
    entities['locality_attribute'] = extract_locality_attributes(user_text)
    print(f"🔍 Entities detected: {entities}")

    # ----------------------------
    # 5️⃣ Intent detection
    # ----------------------------
    intent_result = detect_intent(user_text)
    intent = intent_result.get("intent")
    combo_type = intent_result.get("combo_type")
    confidence = intent_result.get("confidence", 0.0)

    print(f"🎯 Intent: {intent} (conf={confidence})")
    if combo_type:
        print(f"🔗 Combo detected: {combo_type}")

    # ----------------------------
    # 6️⃣ Greetings (handle early)
    # ----------------------------
    if intent == "greeting":
        return handle_greeting()

    # ----------------------------
    # 7️⃣ Locality attribute queries
    # ----------------------------
    if entities.get('locality_attribute') != 'all':
        if intent == 'comparison' and not entities.get('locality'):
            print("📊 Investment comparison query without specific localities")
        else:
            print("📍 Detected locality attribute query")
            return handle_locality_query(user_text, entities)

    # ----------------------------
    # 8️⃣ Open-ended investment comparison
    # ----------------------------
    investment_keywords = ['best investment', 'roi', 'appreciation', 'growth']
    if (intent == 'comparison'
        and not entities.get('locality')
        and not entities.get('localities')
        and any(kw in user_text.lower() for kw in investment_keywords)):
        print("💰 Open investment query - fetching top areas")
        return handle_top_investment_areas()

    # ----------------------------
    # 9️⃣ Comparison queries
    # ----------------------------
    if intent == 'comparison':
        if entities.get('method') == 'multi_locality' and len(entities.get('localities', [])) >= 2:
            print("📊 Processing comparison query...")
            return handle_comparison_query(entities)
        else:
            return "I couldn't identify the localities you want to compare. Try: 'Compare Satellite vs Bopal' or 'Which is better: Thaltej, Vastrapur, or Bodakdev?'"

    # ----------------------------
    # 🔥 FIX: EXPLICIT PRICE/RENT/PSF HANDLERS
    # ----------------------------
    
    # 🔟 PSF Queries (highest priority for price queries)
    if detect_psf_query(user_text):
        print("📌 Detected PSF query")
        return handle_price_query(user_text, entities)
    
    # 1️⃣1️⃣ Regular PRICE queries
    if intent == 'price':
        print("💰 Detected PRICE query")
        return handle_price_query(user_text, entities)
    
    # 1️⃣2️⃣ Regular RENT queries
    if intent == 'rent':
        print("🏠 Detected RENT query")
        return handle_price_query(user_text, entities)

    # ----------------------------
    # 1️⃣3️⃣ Handle mixed intents / combos
    # ----------------------------
    if intent == 'mixed' and combo_type:
        print(f"🔀 Handling combo: {combo_type}")
        
        if combo_type == 'price_rent':
            result = retrieve_property_info(user_text, entities, intent='mixed')
            if result and result.get("found"):
                return format_price_response(result)
        
        elif combo_type == 'price_size':
            price_result = retrieve_property_info(user_text, entities, intent='price')
            size_result = handle_size_query(entities)
            if price_result and price_result.get("found"):
                response = format_price_response(price_result)
                response += f"\n\n{size_result}"
                return response
        
        elif combo_type == 'psf_size':
            psf_result = get_price_per_sqft(entities.get('locality'), entities.get('property_type'))
            if psf_result and psf_result.get("found"):
                response = f"In {psf_result['locality']}, {psf_result['property_type']} costs ₹{psf_result['avg_psf']:,.0f} per sq ft"
                response += f" with an average size of {psf_result['avg_size']:,.0f} sq ft."
                return response
        
        elif combo_type == 'rent_size':
            rent_result = retrieve_property_info(user_text, entities, intent='rent')
            size_result = handle_size_query(entities)
            response = ""
            if rent_result and rent_result.get("found"):
                response += format_price_response(rent_result)
            if size_result:
                response += f"\n\n{size_result}"
            return response

    # ----------------------------
    # 1️⃣4️⃣ Locality info
    # ----------------------------
    if intent == "locality_info":
        print("📍 Locality info query")
        return handle_locality_query(user_text, entities)

    # ----------------------------
    # 1️⃣5️⃣ Property size
    # ----------------------------
    if intent == "property_size":
        print("📏 Property size query")
        return handle_size_query(entities)

    # ----------------------------
    # 1️⃣6️⃣ Fallback: Check if it's a property query that fell through
    # ----------------------------
    property_keywords = ["price", "cost", "rate", "rent",
                         "bhk", "flat", "apartment", "property",
                         "locality", "area"]
    is_property_query = any(k in user_text.lower() for k in property_keywords)
    
    if is_property_query:
        print("⚠️ Property query detected but no specific handler matched")
        print("→ Attempting fallback to handle_price_query...")
        
        # Try to handle as price query
        result = handle_price_query(user_text, entities)
        if result and not result.startswith("I couldn't find"):
            return result
        
        # If that fails, try fallback
        return handle_fallback(user_text, entities)

    # ----------------------------
    # 1️⃣7️⃣ Non-property / general queries → Groq
    # ----------------------------
    print("→ Using Groq for general question...")
    if groq_client is None:
        return "⚠️ Please provide a valid Groq API key to answer this question."
    
    # ✅ Pass conversation_history to handle_groq_general --> working on conversation history in this method handle_groq...
    return handle_groq_general(user_text, groq_client, conversation_history)


# ============================================================================
# testing logic of this file 
#===========================================================================
# Add to bottom of intent_router.py for testing

# Add to bottom of intent_router.py
if __name__ == "__main__":
    test_queries = [
        "What is the price of 2 BHK in Satellite?",        # Should work now
        "Rent for 3 bhk in Bopal?",                        # Should work now
        "Tell me price per sqft in Vastrapur",             # PSF - should work
        "Price and rent for apartment in Thaltej",         # Mixed - should work
        "Compare Satellite vs Bodakdev for investment",    # Comparison - should work
        "What is the area of 3bhk in Shilaj?"             # Size - should work
    ]
    
    print("\n" + "="*70)
    print("TESTING FIXED QUERY PROCESSOR")
    print("="*70)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        response = process_user_query(query, os.getenv("GROQ_API_KEY"), True)
        print(f"🤖 Response: {response}\n")
        print("-"*70)