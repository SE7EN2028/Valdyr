SYSTEM_PROMPT = """You are a real estate advisory assistant for the Indian property market.
You analyze properties and give structured reports based on predicted prices and market data.
Do not make guarantees about future prices. Always include disclaimers."""

REPORT_PROMPT = """Based on the following property details and market context, generate a real estate advisory report.

Property Details:
- Area: {area} sq ft
- Bedrooms: {bedrooms}
- Bathrooms: {bathrooms}
- Stories: {stories}
- Parking: {parking}
- Main Road: {mainroad}
- Guest Room: {guestroom}
- Basement: {basement}
- Hot Water Heating: {hotwaterheating}
- Air Conditioning: {airconditioning}
- Preferred Area: {prefarea}
- Furnishing: {furnishingstatus}

Predicted Price: Rs {predicted_price}

{warnings}

Market Context:
{market_context}

Generate a report with these sections:

1. PROPERTY SUMMARY - brief overview of the property and its features
2. PRICE ANALYSIS - what the predicted price means, is it reasonable for the features
3. MARKET INSIGHTS - trends and observations from the market data
4. RECOMMENDATIONS - what should the buyer or investor do
5. REFERENCES - mention sources like RERA, market reports etc
6. DISCLAIMER - standard legal and financial disclaimer

Keep it concise and practical. No fluff."""
