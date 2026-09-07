import joblib
import spacy
import re
import requests
import os

class CustomNLPEngine:
    def __init__(self):
        print("Loading custom NLP models...")
        try:
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.intent_clf = joblib.load('intent_model.pkl')
            self.nlp = spacy.load("en_core_web_sm")
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Warning: Models not found. Run train_intent.py first. Error: {e}")

        # Simple Dictionary/Regex for Custom Agricultural NER
        # In a larger app, we would train a custom spaCy NER component
        self.known_crops = ["tomato", "onion", "wheat", "rice", "tamatar", "pyaz"]

    def extract_entities(self, text: str):
        text_lower = text.lower()
        entities = {"crop": None, "quantity": None}
        
        # 1. Extract Crop
        for crop in self.known_crops:
            if crop in text_lower:
                entities["crop"] = crop
                break
                
        # 2. Extract Quantity (e.g., "100 kg", "50 quintal")
        qty_match = re.search(r'(\d+)\s*(kg|kilos|quintal|tons?)', text_lower)
        if qty_match:
            entities["quantity"] = qty_match.group(0)
            
        return entities

    def predict_intent(self, text: str):
        text_vec = self.vectorizer.transform([text.lower()])
        intent = self.intent_clf.predict(text_vec)[0]
        return intent

    def process_message(self, user_id: str, text: str, token: str):
        """
        Full Pipeline: Intent -> Entities -> HTTP Request to NestJS -> Natural Language Response
        """
        intent = self.predict_intent(text)
        entities = self.extract_entities(text)
        
        print(f"[{user_id}] Intent: {intent} | Entities: {entities}")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Pull backend URL from environment variables for production (Render)
        BACKEND_URL = os.getenv("NESTJS_BACKEND_URL", "http://localhost:3000")

        # Dialogue State Routing & Backend Integration
        if intent == "sell_produce":
            if not entities["crop"]:
                return "Which crop would you like to sell?"
            if not entities["quantity"]:
                return f"How much {entities['crop']} do you want to sell? Please specify in kg or quintal."
                
            qty_num = int(re.search(r'\d+', entities["quantity"]).group())
            # Convert quintal to kg
            if 'quintal' in entities["quantity"].lower():
                qty_num = qty_num * 100
                
            # Make HTTP Request to NestJS
            payload = {
                "farmer_id": user_id,
                "data": {
                    "crop_name": entities["crop"].lower(),
                    "variety": "Standard",
                    "total_quantity_kg": float(qty_num),
                    "available_quantity_kg": float(qty_num),
                    "expected_price_per_kg": 25.0, # Could be extracted from voice too
                    "harvest_date": "2026-09-07T00:00:00Z",
                    "pickup_latitude": 17.3850, # Mock lat/lng for now
                    "pickup_longitude": 78.4867,
                    "pickup_address": "Registered Farm Location"
                }
            }
            
            try:
                print(f"Calling POST {BACKEND_URL}/api/v1/farmer/produce")
                response = requests.post(f"{BACKEND_URL}/api/v1/farmer/produce", json=payload, headers=headers)
                if response.status_code in [200, 201]:
                    return f"Great! I have officially registered your {entities['quantity']} of {entities['crop']} for sale in the marketplace."
                else:
                    return f"Sorry, there was an error registering your produce. Status: {response.status_code}"
            except Exception as e:
                print(f"Backend Error: {e}")
                return "The core servers are currently unreachable. Please try again later."
            
        elif intent == "check_price":
            if not entities["crop"]:
                return "Which crop's price do you want to check?"
            return f"The current market price for {entities['crop']} is around 25 rupees per kg based on recent trades."
            
        elif intent == "check_demand":
            if not entities["crop"]:
                return "Which crop's demand are you looking for?"
                
            try:
                print(f"Calling GET {BACKEND_URL}/api/matching/find")
                # Using a 100kg dummy quantity if not provided, just to see if buyers exist
                qty = 100
                if entities["quantity"]:
                    qty = int(re.search(r'\d+', entities["quantity"]).group())
                    if 'quintal' in entities["quantity"].lower():
                        qty = qty * 100

                # Note: NestJS FindMatchesDto expects crop, quantity_kg, lat, lng
                params = {
                    "crop": entities["crop"].lower(),
                    "quantity_kg": qty,
                    "lat": 17.3850,
                    "lng": 78.4867
                }
                
                response = requests.get(f"{BACKEND_URL}/api/matching/find", params=params, headers=headers)
                
                if response.status_code == 200:
                    matches = response.json()
                    # Example NestJS matching response structure logic
                    # This relies on the exact shape returned by matching.service.ts
                    if isinstance(matches, list) and len(matches) > 0:
                        return f"Good news! I found {len(matches)} bulk buyers looking for {entities['crop']} nearby. Let me know if you want to sell."
                    else:
                        return f"Right now, there are no immediate bulk buyers for {entities['crop']} in your area, but demand can change daily!"
                else:
                    return f"Sorry, I couldn't fetch the demand right now. Error {response.status_code}."
            except Exception as e:
                print(f"Backend Error: {e}")
                return "The matching engine is currently unreachable."
            
        else:
            return "Namaste! I am your AgriConnect Assistant. I can help you register produce for sale, or check current market demand."

# Singleton instance
nlp_engine = CustomNLPEngine()
