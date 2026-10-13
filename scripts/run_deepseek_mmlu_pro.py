import asyncio, os, sys
sys.path.insert(0, '/opt/llmbench')

from aiohttp_socks import ProxyConnector
import aiohttp

# Patch aiohttp.TCPConnector to use SOCKS proxy
_orig_connector = aiohttp.TCPConnector
class PatchedConnector(_orig_connector):
    def __init__(self, *args, **kwargs):
        # Check if this is an OpenRouter request
        super().__init__(*args, **kwargs)

# Simpler: monkey-patch the session creation
_orig_clientSession_init = aiohttp.ClientSession.__init__
def patched_init(self, *args, **kwargs):
    connector = kwargs.get('connector')
    if connector is None:
        # Use SOCKS proxy connector for all outbound
        try:
            connector = ProxyConnector.from_url('socks5://127.0.0.1:10810')
            kwargs['connector'] = connector
        except:
            pass
    _orig_clientSession_init(self, *args, **kwargs)

aiohttp.ClientSession.__init__ = patched_init

# Now run lm_eval programmatically
os.environ['HF_TOKEN'] = ''

from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

result = simple_evaluate(
    model='local-chat-completions',
    model_args='model=deepseek/deepseek-chat,base_url=https://openrouter.ai/api/v1/chat/completions,api_key=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,num_concurrent=4,max_gen_toks=1024',
    tasks=['mmlu_pro'],
    apply_chat_template=True,
    batch_size=1,
)

import json
out = '/opt/llmbench/out/lmeval_deepseek-v4-flash_queue/mmlu_pro'
os.makedirs(out, exist_ok=True)

# Save results
if result and 'results' in result:
    with open(f'{out}/results_deepseek_mmlu_pro.json', 'w') as f:
        json.dump(result['results'], f, indent=2)
    
    for k, v in result['results'].items():
        if 'mmlu_pro' in k.lower():
            acc = v.get('exact_match,custom-extract-prompt', v.get('acc,none', '?'))
            print(f'{k}: {acc}', flush=True)
else:
    print('NO RESULTS', flush=True)
    print(json.dumps(result, indent=2, default=str)[:500] if result else 'None', flush=True)
