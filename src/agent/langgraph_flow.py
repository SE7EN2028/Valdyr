import sys
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.llm.provider import get_llm
from src.rag.retriever import get_relevant_docs
from src.agent.prompts import SYSTEM_PROMPT, REPORT_PROMPT

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "app"
model = joblib.load(BASE_DIR / "house_price_model.pkl")
model_columns = joblib.load(BASE_DIR / "model_columns.pkl")

DEFAULTS = {
    "area": 2000, "bedrooms": 3, "bathrooms": 2,
    "stories": 2, "parking": 1, "mainroad": 1,
    "guestroom": 0, "basement": 0, "hotwaterheating": 0,
    "airconditioning": 1, "prefarea": 1,
    "furnishingstatus": "semi-furnished"
}

class AgentState(TypedDict):
    input_data: dict
    warnings: str
    predicted_price: float
    market_context: str
    report: str
    provider: str


def validate_input(state: AgentState):
    data = state["input_data"]
    warnings = []

    for key, default in DEFAULTS.items():
        if key not in data or data[key] is None or data[key] == "":
            data[key] = default
            warnings.append(f"{key} was missing, used default: {default}")

    if data["area"] < 100 or data["area"] > 50000:
        warnings.append(f"area ({data['area']}) looks unusual")

    if data["bedrooms"] < 1 or data["bedrooms"] > 10:
        warnings.append(f"bedrooms ({data['bedrooms']}) looks unusual")

    warning_text = ""
    if warnings:
        warning_text = "Data Warnings:\n" + "\n".join(["- " + w for w in warnings])

    return {"input_data": data, "warnings": warning_text}


def predict_price(state: AgentState):
    data = state["input_data"]

    input_dict = {
        "area": data["area"],
        "bedrooms": data["bedrooms"],
        "bathrooms": data["bathrooms"],
        "stories": data["stories"],
        "parking": data["parking"],
        "mainroad": data["mainroad"],
        "guestroom": data["guestroom"],
        "basement": data["basement"],
        "hotwaterheating": data["hotwaterheating"],
        "airconditioning": data["airconditioning"],
        "prefarea": data["prefarea"],
        "furnishingstatus_semi-furnished": 1 if data["furnishingstatus"] == "semi-furnished" else 0,
        "furnishingstatus_unfurnished": 1 if data["furnishingstatus"] == "unfurnished" else 0,
    }

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=model_columns, fill_value=0)
    price = model.predict(df)[0]

    return {"predicted_price": float(price)}


def retrieve_context(state: AgentState):
    data = state["input_data"]
    query = f"property with {data['bedrooms']} bedrooms {data['area']} sqft investment advice market trends"
    context = get_relevant_docs(query, k=4)
    return {"market_context": context}


def generate_report(state: AgentState):
    data = state["input_data"]
    provider = state.get("provider", "groq")
    llm = get_llm(provider=provider)

    prompt = REPORT_PROMPT.format(
        area=data["area"],
        bedrooms=data["bedrooms"],
        bathrooms=data["bathrooms"],
        stories=data["stories"],
        parking=data["parking"],
        mainroad="Yes" if data["mainroad"] else "No",
        guestroom="Yes" if data["guestroom"] else "No",
        basement="Yes" if data["basement"] else "No",
        hotwaterheating="Yes" if data["hotwaterheating"] else "No",
        airconditioning="Yes" if data["airconditioning"] else "No",
        prefarea="Yes" if data["prefarea"] else "No",
        furnishingstatus=data["furnishingstatus"],
        predicted_price=f"{int(state['predicted_price']):,}",
        warnings=state["warnings"],
        market_context=state["market_context"]
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    response = llm.invoke(messages)
    return {"report": response.content}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("validate", validate_input)
    graph.add_node("predict", predict_price)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("report", generate_report)

    graph.set_entry_point("validate")
    graph.add_edge("validate", "predict")
    graph.add_edge("predict", "retrieve")
    graph.add_edge("retrieve", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_advisory(property_data: dict, provider="groq") -> dict:
    app = build_graph()
    result = app.invoke({"input_data": property_data, "provider": provider})
    return {
        "predicted_price": result["predicted_price"],
        "warnings": result["warnings"],
        "report": result["report"]
    }
