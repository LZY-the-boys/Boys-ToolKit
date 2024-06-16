import logging_config
import os
import openai
import logging
import time
import random
import tqdm
import ray
from utils import args
import re
import math

MAXLEN=200
SLEEP_FAST = 0.125
SLEEP_SLOW = 20

LOG = logging.getLogger('ray')

class OpenaiAPIMaxRetryError(Exception):
    # 关于sk-api的所有生成进程需要停止
    pass

class OpenaiAPIDataItemError(Exception):
    # 关于当前prompt的生成进程需要停止
    pass

class OpenaiAPIRestartError(Exception):
    # 重新开始这个数据的流程
    pass

# online openai api
if os.getenv('API') == 'online':
    api_keys = {
        # ---------- fast ---------------
        'sk-xxx': (1, 'https://api.openai.com/v1'),
    }
# local server
else:
    api_keys = { 
        f'sk-{i}': (0.1, os.getenv('API')) 
        for i in range(400)
    }


def split_data(data):
    global api_keys
    assert len(api_keys)
    if os.getenv('API') not in ['FAST','SLOW']:
        # 最大并发
        api_keys = { 
            f'sk-{i}': (0.1, os.getenv('API')) 
            for i in range(len(data))
        }

    total_size = len(data)
    sizes = [1 / a[0] for a in list(api_keys.values())]
    total_parts = sum(sizes)
    part_sizes = [(size / total_parts) * total_size for size in sizes]

    start, end = 0, 0
    split_data = []
    for i, part_size in enumerate(part_sizes):
        end = min(max(math.ceil(start + part_size), start + 1), len(data))
        split_data.append(data[start:end])
        start = end

    if end != len(data):
        split_data[-1] += data[end:len(data)]

    split_len = {a[:6]: len(d) for a, d in zip(api_keys, split_data)}

    print(f'total {total_size}, split data in {split_len}')
    return split_data

@ray.remote
def ray_infer(data, func, api_key, args):

    import utils; utils.args.update(**args)

    if len(data) == 0:
        return []
    if not os.getenv('DEBUG'):
        # 会影响pdb调试
        from ray.experimental.tqdm_ray import tqdm
    else:
        from tqdm import tqdm

    # it's useless because we only change the LOG pointer, not the LOG itself
    # global LOG
    # configure_logging()
    # LOG = logging.getLogger('ray')

    start = time.time()

    openai.api_key = api_key
    openai.api_base = api_keys[api_key][1]
    out = []
    try:
        for d in tqdm(data, desc=api_key[:5]):
            out.append(func(input=d))
    except (OpenaiAPIMaxRetryError, KeyboardInterrupt) as e:
        LOG.error(str(e), exc_info=True)
    except:
        import sys, pdb, bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type, value)
        LOG.error(value, exc_info=True)
        if os.getenv('DEBUG'):
            pdb.post_mortem(tb)
    finally:
        unhandle_num = len(data) - len(out)
        if unhandle_num > 0:
            LOG.error(
                f'there are {unhandle_num} data item unhandled for {api_key[:5]}', exc_info=True
            )
            out += data[len(out):]

    end = time.time()

    LOG.info({'api_key': api_key[:6], 'time_per_item': (end - start) / len(out)})
    return out

def infer(data, func, api_key):
    start = time.time()

    out = []
    try:
        for d in tqdm.tqdm(data, desc=api_key[:6]):
            out.append(func(input=d))
    except (OpenaiAPIMaxRetryError) as e: # KeyboardInterrupt
        unhandle_num = len(data) - len(out)
        print(f'there are {unhandle_num} data item unhandled for {api_key[:5]}')
        out.append(data[len(out):])

    end = time.time()

    print({'api_key': api_key[:6], 'time_per_item': (end - start) / len(out)})
    return out

## For OpenAI API call
class OpenAI():
    def __init__(self, model, system_prompt=''):
        super(OpenAI, self).__init__()
        if system_prompt is None:
            self.messages = []
        else:
            self.messages = [{'role': 'system', 'content': system_prompt}]
        self.seed = args.seed
        self.model = model  #"gpt-3.5-turbo"
        self.previous_prompts = [] # avoid each input the same prompt 

    def clear(self):
        self.messages = []
    def set_seed(self,seed):
        self.seed = seed
    def generate(self, input):

        if isinstance(input,list) or len(input) < 2:
            LOG.error('input format is wrong')
            if os.getenv('DEBUG'):
                import pdb; pdb.set_trace()
            raise OpenaiAPIDataItemError

        if self.previous_prompts.count(input) >= 3:
            print(self.previous_prompts)
            LOG.error("prompt repetitions detect!!. Stopping.")
            if os.getenv('DEBUG'):
                import pdb; pdb.set_trace()
            raise OpenaiAPIMaxRetryError

        self.messages.append({
            "role": "user",
            "content": input,
        })
        if 'Next, diligently contrast the deviations between these two topics,' not in input and 'Then Carefully compare the differences between the two topics, ignoring the above similarities. Sum' not in input:
            self.previous_prompts.append(input)
        if len(self.previous_prompts) > 10:
            self.previous_prompts.pop(0)

        if len(self.messages) > 10:
            LOG.error('you may have infinite loop')
            if os.getenv('DEBUG'):
                import pdb; pdb.set_trace()
            raise OpenaiAPIDataItemError

        if args.dry_run:
            response = {'role': 'assistant', 'content': 'dry-run'}
            self.messages.append(response)

            LOG.debug({
                'api_key': openai.api_key,
                'message': self.messages
            })
            return response['content']

        while True:
            for retry in range(300):
                try:
                    if "-instruct" in self.model:
                        completion = openai.Completion.create(
                            model=self.model,
                            prompt=self.messages[-1]['content'],  # only completion
                            max_tokens=MAXLEN,
                        )
                    elif "gpt-3.5" in self.model or "gpt-4" in self.model:
                        completion = openai.ChatCompletion.create(
                            model=self.model, messages=self.messages, temperature=0,seed=self.seed,
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=self.model, messages=self.messages, temperature=0,seed=self.seed,
                            max_tokens=MAXLEN,stop=['<|endoftext|>', 'Human:']
                        )

                    if os.getenv('API') == 'FAST':
                        time.sleep(SLEEP_FAST)
                    elif os.getenv('API') == 'SLOW':
                        time.sleep(SLEEP_SLOW)
                    else:
                        pass
                    break

                except Exception as e:

                    error_message = str(e)

                    LOG.error({
                        'api_key': openai.api_key,
                        'error': error_message,
                    }, exc_info=True)

                    if 'You exceeded your current quota' in error_message or '用户额度不足' in error_message or '该令牌额度已用尽' in error_message or 'Incorrect API key provided:' in error_message or 'is invalid. Please check the API reference' in error_message or 'requests per day (RPD):' in error_message:
                        # 这个api的后续都需要停止
                        raise OpenaiAPIMaxRetryError

                    if 'Please reduce the length of the messages' in error_message:
                        # 只需要停这一个zero-shot/few-shot的后续
                        if os.getenv('DEBUG'):
                            import pdb; pdb.set_trace()
                        raise OpenaiAPIDataItemError
                    
                    if 'Your OpenAI account has been deactivated' in error_message:
                        # proxy server error
                        time.sleep(SLEEP_FAST)
                    elif 'That model is currently overloaded with other requests.' in error_message or '当前分组上游负载已饱和' in error_message:
                        time.sleep(random.randint(1, 10))

                    elif 'Rate limit reached' in error_message:
                        result = re.findall("(\d+\.\d+)(?=s.)", error_message)
                        if len(result):
                            wait_time = float(result[-1])
                        else:
                            wait_time = random.randint(1, 10)

                        time.sleep(wait_time)

                    elif 'HTTP code 504 from API' in error_message:
                        time.sleep(SLEEP_FAST)
                    else:
                        if os.getenv('DEBUG'):
                            import pdb; pdb.set_trace()
                        time.sleep(random.randint(1, 20))
                        import pdb;pdb.set_trace()

            # TODO: retry
            if retry < 300:
                break
            else:
                LOG.error(f"cannot use {openai.api_key}, early exist", exc_info=True)
                # return None
                raise OpenaiAPIMaxRetryError

        if "-instruct" in self.model:
            response = completion["choices"][0]["text"]
            self.messages.append({'role': 'assistant', 'content': response})
            LOG.debug(self.messages)
            return response

        response = completion["choices"][0]["message"].to_dict()

        self.messages.append(response)

        LOG.debug(self.messages)
        return response['content']
    
def run(dataset, func):

    def invalid(r):
        return not hasattr(r, 'valid') or not r.valid

    data_chunks = split_data(dataset)

    if args.dry_run:
        results = [infer(dataset, func, 'sk-none')]
    elif args.debug:
        # 默认/tmp/ray
        ray.init(
            num_cpus=1,
            log_to_driver=True,
            logging_level=logging.DEBUG,
        )
        results = [
            ray_infer.remote(dataset, func, next(iter(api_keys.keys())), args)
        ]
        results = ray.get(results)
        ray.shutdown() 
    else:
        ray.init(num_cpus=len(api_keys),log_to_driver=True)
        results = [
            ray_infer.remote(chunk, func, api_key, args) for chunk, api_key in zip(data_chunks, api_keys)
        ]
        results = ray.get(results)
        ray.shutdown()

    # maybe some apikey has zero data items
    results = [r for rr in results for r in rr if len(rr) ]

    # flatten
    results, unfinished_results = [r for r in results if not invalid(r) ], [r for r in results if invalid(r) ]
        
    return results, unfinished_results
