import json,time,urllib.request
from pathlib import Path
out=Path('reports/rngd_porting/web_acceptance.json')
for _ in range(300):
    try:
        urllib.request.urlopen('http://127.0.0.1:7860/config',timeout=3)
        break
    except Exception: time.sleep(3)
else: raise RuntimeError('Web startup did not complete; inspect web_server.log')
from gradio_client import Client
client=Client('http://127.0.0.1:7860')
rows=[]
for prompt in ['A red fox beside a mountain lake at sunrise','A watercolor painting of a small blue house among autumn trees']:
    start=time.perf_counter()
    response=client.predict('sdxl',prompt,1,20261102,api_name='/generate_image_ui')
    rows.append({'prompt':prompt,'elapsed_s':time.perf_counter()-start,'image':str(response[0]),'status':response[1],'log':response[2]})
    out.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    assert response[0] is not None,response
print(json.dumps(rows,ensure_ascii=False))
