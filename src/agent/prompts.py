SYSTEM_PROMPT = "You are a helpful real estate assistant. Provide advice based on the given property details and predicted price."

PROPERTY_ANALYSIS_PROMPT = """Please analyze this property:

Area: {area} sq ft
Bedrooms: {bedrooms}
Bathrooms: {bathrooms}
Stories: {stories}
Parking: {parking}

Predicted Price: {predicted_price}

Your Advice:"""

RAG_PROMPT = """Use the context below to answer the question.

Context: 
{context}

Question: {question}

Answer:"""
