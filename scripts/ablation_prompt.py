#!/usr/bin/env python3
"""Prompt 消融实验 — 对比三种 prompt 条件的描述质量"""
import json, time, random, sys, re
from pathlib import Path
from openai import OpenAI
from collections import Counter

FORBIDDEN = ['light','lighting','shadow','bright','dark','depth','atmosphere',
             'airy','mood','perspective','contrast','illuminat']

def has_object_class(caption):
    """检查是否包含物体类别词"""
    class_words = ['door','floor','ceiling','window','wall','table','chair','sofa',
                   'couch','cabinet','bookshelf','shelf','desk','bed','rug','carpet',
                   'lamp','pillow','curtain','pot','plant','ottoman','counter','stove',
                   'fridge','wardrobe','nightstand']
    return any(w in caption.lower() for w in class_words)

def run_condition(client, name, prompt_template, items, model='qwen2.5:3b'):
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=400, temperature=0.3,
            messages=[{'role':'user','content':prompt_template + '\n\n' + items}])
    except Exception as e:
        return {'error': str(e), 'time': time.time()-t0}
    dt = time.time() - t0
    raw = r.choices[0].message.content.strip()
    try:
        if '[' in raw and ']' in raw:
            parsed = json.loads(raw[raw.index('['):raw.rindex(']')+1])
        else:
            parsed = []
    except:
        parsed = []

    forbidden_count = sum(1 for p in parsed
                         for w in FORBIDDEN
                         if w in p.get('caption','').lower())
    class_count = sum(1 for p in parsed if has_object_class(p.get('caption','')))
    return {
        'name': name,
        'items': len(parsed),
        'time': dt,
        'tokens': r.usage.completion_tokens if r.usage else 0,
        'forbidden_words': forbidden_count,
        'has_class': class_count,
        'has_class_field': 'class' in (parsed[0] if parsed else {}),
        'result': parsed,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='office0')
    parser.add_argument('--api-base', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model', default='qwen2.5:3b')
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key='x')
    caps = json.load(open(
        f'/datadisk3/renlide_home/dataAnnotator/Datasets/Replica/{args.scene}/sg_cache/cfslam_captions_llava.json'))
    random.seed(42)
    samples = random.sample(caps, min(args.count, len(caps)))
    items = '\n'.join(f"{c.get('obj_idx',0)}: {c['caption']}" for c in samples)

    # Raw LLaVA baseline — just take the captions as-is
    raw_baseline = [{'id': c.get('obj_idx',0), 'caption': c['caption']} for c in samples]
    raw_class = sum(1 for c in raw_baseline if has_object_class(c['caption']))
    raw_forbidden = sum(1 for c in raw_baseline
                       for w in FORBIDDEN if w in c['caption'].lower())

    # Condition A: Old simple prompt
    old_prompt = 'Clean up each description into one short coherent sentence. Output ONLY a JSON array: [{"id": id, "caption": "text"}]'
    res_a = run_condition(client, 'A: Old simple prompt', old_prompt, items, args.model)

    # Condition B: New structured prompt (our production prompt)
    new_prompt = """Identify what each object IS (category + material + position). Output ONLY JSON: [{"id":number,"class":"category","caption":"description"}].
Name the category (door/floor/window/table/sofa/chair/cabinet/bed/lamp/rug/bookshelf/desk/counter). Describe material/color/shape. ONE sentence.
NEVER use: light, lighting, shadow, bright, dark, depth, atmosphere, airy, mood, perspective, contrast, illuminate.
If too vague to identify: class="unknown", caption="An unidentifiable object." """
    res_b = run_condition(client, 'B: New structured prompt', new_prompt, items, args.model)

    # Condition C: New prompt WITHOUT forbidden words list (ablation)
    ablated_prompt = """Identify what each object IS (category + material + position). Output ONLY JSON: [{"id":number,"class":"category","caption":"description"}].
Name the category. Describe material/color/shape. ONE sentence.
If too vague to identify: class="unknown", caption="An unidentifiable object." """
    res_c = run_condition(client, 'C: Ablated (no forbidden list)', ablated_prompt, items, args.model)

    # Report
    print('='*60)
    print('  PROMPT ABLATION STUDY')
    print('='*60)
    print(f'  Scene: {args.scene}, 样本: {len(samples)} captions')
    print(f'  Model: {args.model}')
    print()
    print(f'  Baseline (raw LLaVA): {raw_class}/{len(samples)} with object class, {raw_forbidden} forbidden words')
    print()

    for res in [res_a, res_b, res_c]:
        n = res['name']
        if 'error' in res:
            print(f'  {n}: ERROR {res["error"][:80]}')
            continue
        rate = res['has_class']/max(res['items'],1)*100
        fw = res['forbidden_words']
        print(f'  {n}:')
        print(f'    Items: {res["items"]}, Speed: {res["time"]:.1f}s, Tokens: {res["tokens"]}')
        print(f'    Has class: {res["has_class"]}/{res["items"]} ({rate:.0f}%)')
        print(f'    Forbidden words: {fw}')
        print(f'    Has class field: {res["has_class_field"]}')
        if res['result']:
            r0 = res['result'][0]
            print(f'    Sample: [{r0.get("id","?")}] {r0.get("caption","")[:100]}')
        print()

    # Save results
    with open(f'/tmp/ablation_{args.scene}.json', 'w') as f:
        json.dump({
            'baseline': {'class_rate': raw_class/len(samples), 'forbidden': raw_forbidden},
            'old_prompt': {k:res_a[k] for k in ['items','time','tokens','forbidden_words','has_class']},
            'new_prompt': {k:res_b[k] for k in ['items','time','tokens','forbidden_words','has_class']},
            'ablated': {k:res_c[k] for k in ['items','time','tokens','forbidden_words','has_class']},
        }, f, indent=2)
    print(f'Saved: /tmp/ablation_{args.scene}.json')

if __name__ == '__main__':
    import argparse
    main()
