import sys, requests
sys.path.insert(0, 'market_predictor')
from dotenv import load_dotenv; load_dotenv('market_predictor/.env')
from inference.pipeline import predict, _load_models
from data.auth import get_headers

_load_models()
r = requests.get('https://api.ci.spglobal.com/news-insights/v1/search',
    headers=get_headers(), params={'q':'crude oil','pageSize':10,'page':1}, timeout=20)
articles = r.json().get('results', [])

labels = []; sentiments = []
for a in articles[:10]:
    hl = (a.get('headline') or a.get('title',''))[:70]
    res = predict(hl, skip_explanation=True)
    lbl = res['label']
    sent = res['entities']['sentiment']
    sup = res['entities']['supply_impact']
    labels.append(lbl)
    sentiments.append(sent)
    print(f"{lbl:6} sent={sent:+.2f} sup={sup} | {hl}")

print()
print("HIGH:", labels.count('HIGH'), "MEDIUM:", labels.count('MEDIUM'), "LOW:", labels.count('LOW'))
print("Avg sentiment:", round(sum(sentiments)/len(sentiments), 3))
