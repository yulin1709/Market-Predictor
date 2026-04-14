import sys
sys.path.insert(0, 'market_predictor')
from inference.pipeline import predict, _load_models
_load_models()

headlines = [
    "Thailand reaches deal with Iran to transit tankers through Hormuz",
    "Pacific LNG Freight: 160,000 cu m TFDE day rate heard at 75000/day",
    "Saudi Arabia cuts crude output by 500000 bpd in surprise move",
    "US shale production hits record high in Permian Basin",
    "Nord Stream pipeline explosion disrupts European gas supply",
]

scored = [predict(h) for h in headlines]
avg_h = sum(a["high_prob"] for a in scored) / len(scored)
avg_m = sum(a["medium_prob"] for a in scored) / len(scored)
avg_l = sum(a["low_prob"] for a in scored) / len(scored)
avg_s = sum(a["entities"].get("sentiment", 0) for a in scored) / len(scored)

print("Base avg_high:", round(avg_h * 100, 1))

from app.streamlit_app import _per_commodity_signals
result = _per_commodity_signals(scored, avg_h, avg_m, avg_l, avg_s)
for k, v in result.items():
    name = k.strip().replace("\n", " ")
    print(f"{name}: {v['confidence']:.1f}%  dir={v['direction']}  n_relevant={v['n_relevant']}")
