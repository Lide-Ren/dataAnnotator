#!/usr/bin/env python3
"""实验2：模型选型验证 — 用不同模型跑完整 Step5, 对比实际管线 QA 质量"""
import json, time, subprocess, os, sys, argparse

MODELS = {
    'qwen2.5:3b': 'Qwen2.5-3B',
    'qwen2.5:7b': 'Qwen2.5-7B',
    'qwen2.5:14b': 'Qwen2.5-14B',
    'glm4:latest': 'GLM-4-9B',
}

def run_step5(scene, model, api_base='http://127.0.0.1:8080/v1'):
    """Run build_scenegraph_batch.py and return output stats"""
    script = '/datadisk3/renlide_home/dataAnnotator/scripts/build_scenegraph_batch.py'
    env = os.environ.copy()
    env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    # Start llama.cpp server for this model
    port = 8090
    # (Server lifecycle handled by benchmark_local_models.py pattern)

    t0 = time.time()
    cmd = ['python', script, '--scene', scene, '--api_base', api_base, '--model', model]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    dt = time.time() - t0

    # Parse output
    refined = json.load(open(f'/datadisk3/renlide_home/dataAnnotator/Datasets/Replica/{scene}/sg_cache/cfslam_refined_captions.json'))
    relations = json.load(open(f'/datadisk3/renlide_home/dataAnnotator/Datasets/Replica/{scene}/sg_cache/cfslam_object_relations.json'))

    return {
        'model': model,
        'time': dt,
        'refined_count': len(refined),
        'relation_count': len(relations),
        'avg_caption_len': sum(len(r.get('caption','')) for r in refined) / max(len(refined), 1),
        'class_diversity': len(set(r.get('class','') for r in refined)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='office0')
    parser.add_argument('--models', nargs='+', default=['qwen2.5:3b', 'qwen2.5:7b'])
    args = parser.parse_args()

    print('='*60)
    print('  MODEL QUALITY COMPARISON (Real Pipeline)')
    print(f'  Scene: {args.scene}')
    print('='*60)

    results = {}
    for model_key in args.models:
        name = MODELS.get(model_key, model_key)
        print(f'\n  Testing {name}...')
        res = run_step5(args.scene, model_key)
        results[name] = res
        print(f'    Time: {res["time"]:.0f}s')
        print(f'    Refined: {res["refined_count"]}, Relations: {res["relation_count"]}')
        print(f'    Class diversity: {res["class_diversity"]}')
        print(f'    Avg caption: {res["avg_caption_len"]:.0f} chars')

    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    for name, res in results.items():
        print(f'  {name}: {res["refined_count"]} ref, {res["relation_count"]} rel, '
              f'{res["class_diversity"]} classes, {res["time"]:.0f}s')

if __name__ == '__main__':
    main()
