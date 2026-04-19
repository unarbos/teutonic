#!/usr/bin/env python3
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, use_safetensors=True)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.adapter_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
