"""Cache an immutable Qwen snapshot; downloads weights without loading a model."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from huggingface_hub import model_info, snapshot_download
    revision = model_info(args.model, revision=args.revision).sha
    path = snapshot_download(args.model, revision=revision, max_workers=3,
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'model': args.model, 'revision': revision,
                                 'cached_snapshot': str(path), 'model_loaded': False}, indent=2) + '\n')
    print(output.read_text())


if __name__ == '__main__':
    main()
