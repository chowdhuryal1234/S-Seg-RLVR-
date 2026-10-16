"""Six preselected, unmarked local-patch classifications with frozen Qwen32B.

One fresh-context greedy answer per patch. IDs and ground truth are bookkeeping
only. Native20 crops are passed directly to the same processor (actual224).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from nucleus_rl.coordinate_calibration import MODEL, REVISION, save, sha256


PROMPT = [
    {"role": "system", "content": "You classify microscopy image patches."},
    {"role": "user", "content": "Is the tissue at the center of this image patch inside a cell nucleus, or is it background tissue? "
     "A nucleus may extend beyond the edges of the patch; a complete nucleus outline is not required. "
     "Reply with exactly one lowercase word: nucleus or background. Do not add explanations, punctuation, coordinates, or other text."},
]


def parse_label(text):
    label = text.strip() if isinstance(text, str) else None
    return label if label in ("nucleus", "background") else None


def summary(records):
    valid = [r for r in records if r["format_valid"]]
    positives = sum(r["gt_label"] == "nucleus" for r in records)
    tp = sum(r["gt_label"] == "nucleus" and r["predicted_label"] == "nucleus" for r in valid)
    tn = sum(r["gt_label"] == "background" and r["predicted_label"] == "background" for r in valid)
    fp = sum(r["gt_label"] == "background" and r["predicted_label"] == "nucleus" for r in valid)
    fn = sum(r["gt_label"] == "nucleus" and r["predicted_label"] == "background" for r in valid)
    return {"total": len(records), "valid_count": len(valid), "invalid_count": len(records)-len(valid),
            "correct_count": tp+tn, "overall_correct_fraction": (tp+tn)/len(records),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "invalid_positive_count": sum(not r["format_valid"] and r["gt_label"] == "nucleus" for r in records),
            "invalid_background_count": sum(not r["format_valid"] and r["gt_label"] == "background" for r in records),
            "positive_total": positives, "background_total": len(records)-positives,
            "scope": "Six selected local patches from one image; invalid output is separate from biological misclassification. Not dataset accuracy."}


def prepare(output, reference_run):
    import numpy as np
    from PIL import Image
    if output.exists():
        raise FileExistsError("Use a new directory and preserve prior attempts")
    ref = json.loads((reference_run / "result.json").read_text())
    manifest = json.loads((reference_run / "candidate_manifest.json").read_text())
    if (ref["status"] != "completed_recognition_coordinate_probe" or ref["model"]["model"] != MODEL
            or ref["model"]["resolved_revision"] != REVISION or ref["model"]["adapter"] is not None):
        raise ValueError("Expected the completed frozen32B marked-image diagnostic")
    if (sha256(reference_run / "original.png") != manifest["source_image_sha256"]
            or sha256(reference_run / "gt_mask.png") != manifest["source_mask_sha256"]):
        raise ValueError("Original unmarked image/GT changed")
    original = Image.open(reference_run / "original.png").convert("RGB")
    mask = np.asarray(Image.open(reference_run / "gt_mask.png"))
    if original.size != (128,128) or mask.shape != (128,128):
        raise ValueError("Expected the fixed native128 source")
    output.mkdir(parents=True)
    (output / "crops").mkdir()
    (output / "masks").mkdir()
    records = []
    for i,candidate in enumerate(manifest["candidates"],1):
        x0,y0,x1,y1 = candidate["box_original"]
        if x1-x0 != 20 or y1-y0 != 20:
            raise ValueError("All six previously fixed boxes must be20x20")
        crop = original.crop((x0,y0,x1,y1))
        labels = mask[y0:y1,x0:x1]
        center_id = int(labels[10,10])
        if bool(center_id) != candidate["is_nucleus"] or (center_id or None) != candidate["gt_instance_id"]:
            raise ValueError("Center-label key differs from original GT")
        if not center_id and np.any(labels):
            raise ValueError("Background controls must have no annotated nuclear foreground")
        image_file,mask_file = f"crops/{i:02d}.png",f"masks/{i:02d}.png"
        crop.save(output / image_file)
        Image.fromarray(((labels>0)*255).astype(np.uint8)).save(output / mask_file)
        if not np.array_equal(np.asarray(Image.open(output / image_file)),np.asarray(original)[y0:y1,x0:x1]):
            raise ValueError("Crop must preserve source pixels exactly")
        records.append({"id":candidate["id"],"image_file":image_file,"mask_file":mask_file,
                        "crop_box":candidate["box_original"],"gt_label":"nucleus" if center_id else "background",
                        "gt_instance_id":center_id or None,"gt_foreground_fraction":float(np.mean(labels>0)),
                        "retained_nucleus_fraction":float(np.sum(labels==center_id)/np.sum(mask==center_id)) if center_id else None,
                        "native_size":[20,20],"image_sha256":sha256(output/image_file),"mask_sha256":sha256(output/mask_file)})
    if [r["id"] for r in records] != list("ABCDEF"):
        raise ValueError("Use all six fixed regions without reselection")
    for filename,source in (("original.png","original.png"),("gt_mask.png","gt_mask.png"),
                            ("candidate_manifest.json","candidate_manifest.json")):
        (output/filename).write_bytes((reference_run/source).read_bytes())
    save(output/"plan.json",{
        "created_utc":datetime.now(timezone.utc).isoformat(),"scope":"Six magnified local-patch classifications from one selected training-source image",
        "model":MODEL,"revision":REVISION,"generation_controls":ref["generation_controls"],"prompt":PROMPT,
        "reference_result_sha256":sha256(reference_run/"result.json"),"reference_manifest_sha256":sha256(reference_run/"candidate_manifest.json"),
        "source_image_sha256":sha256(output/"original.png"),"source_mask_sha256":sha256(output/"gt_mask.png"),
        "records":records,"attempts_per_crop":1,"fresh_context_per_crop":True,"order":list("ABCDEF"),
        "native_input_size":[20,20],"expected_processed_size":[224,224],"manual_resize":False,
        "processor_size_preflight":"Exact installed processor accepted native20 and produced224 without manual resizing before model loading",
        "model_sees":"One unmarked crop plus identical prompt only. No filenames/IDs, letters, coordinates, GT mask, class key, true class counts or earlier answers.",
        "interpretation_limits":"Positive crops contain partial nuclei. Cropping changes context, attention, prompt and effective object magnification (6.4x versus the prior full128 view); it cannot isolate the cause of prior failure or estimate benchmark performance.",
        "runner_sha256":sha256(__file__),"sam_used":False,"optimizer_steps":0})
    print(f"Prepared six immutable crops at {output}",flush=True)


def run(output, reference_run):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import set_seed
    from nucleus_rl.evaluate import processed_image_size
    from nucleus_rl.prompt_ablation import tensor_digest
    from nucleus_rl.train import load_policy
    started=time.monotonic()
    if (output/"started.json").exists():
        raise FileExistsError("Attempt already started; do not retry selectively")
    plan=json.loads((output/"plan.json").read_text())
    ref=json.loads((reference_run/"result.json").read_text())
    ref_model=json.loads((reference_run/"model.json").read_text())
    if sha256(__file__) != plan["runner_sha256"] or PROMPT != plan["prompt"]:
        raise ValueError("Locked implementation/prompt changed")
    if sha256(reference_run/"result.json") != plan["reference_result_sha256"] or plan["generation_controls"] != ref["generation_controls"]:
        raise ValueError("Reference or controls changed")
    for record in plan["records"]:
        for f,k in (("image_file","image_sha256"),("mask_file","mask_sha256")):
            if sha256(output/record[f]) != record[k]:
                raise ValueError("Locked crop or GT changed")
    save(output/"started.json",{"utc":datetime.now(timezone.utc).isoformat(),"runner_sha256":sha256(__file__)})
    controls=plan["generation_controls"]
    args=argparse.Namespace(model=MODEL,revision=REVISION,adapter=None,seed=controls["seed"],min_pixels=controls["min_pixels"],
                            max_pixels=controls["max_pixels"],coordinate_frame="not_requested",prompt_style="isolated_binary_classification",direct_gpu_load=True)
    torch.set_num_threads(8)
    model,processor,metadata=load_policy(args)
    model.eval()
    metadata["sam_coordinate_system"]="not_used"
    defaults=model.generation_config.to_dict()
    if any(defaults.get(k)!=ref_model["generation_config"].get(k) for k in set(defaults)|set(ref_model["generation_config"])
           if k not in ("transformers_version","_from_model_config")):
        raise ValueError("Generation defaults changed")
    if any(p.requires_grad for p in model.parameters()) or processor.image_processor.to_dict()!=json.loads((reference_run/"processor_config.json").read_text()):
        raise ValueError("Trainable parameters or changed image processor")
    save(output/"model.json",{**metadata,"generation_config":defaults})
    save(output/"processor_config.json",processor.image_processor.to_dict())
    records=[]
    common_token_hash=None
    for record in plan["records"]:
        image=Image.open(output/record["image_file"]).convert("RGB")
        messages=[dict(m) for m in PROMPT]
        messages[-1]["content"]=[{"type":"image"},{"type":"text","text":PROMPT[-1]["content"]}]
        chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[chat],images=[image],padding=True,return_tensors="pt")
        size=processed_image_size(processor,inputs)
        if image.size!=(20,20) or size!=(224,224):
            raise ValueError(f"Unexpected native/processed sizes: {image.size}/{size}")
        pixel_hash,token_hash=tensor_digest(inputs["pixel_values"]),tensor_digest(inputs["input_ids"])
        if common_token_hash is not None and common_token_hash!=token_hash:
            raise ValueError("All six inputs must use identical task tokens and image-grid token count")
        common_token_hash=token_hash
        np.savez_compressed(output/f"{record['id']}_processor_input.npz",pixel_values=inputs["pixel_values"].numpy(),
                            image_grid_thw=inputs["image_grid_thw"].numpy(),input_ids=inputs["input_ids"].numpy())
        inputs=inputs.to(model.device)
        set_seed(controls["seed"])
        t0=time.monotonic()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=controls["max_new_tokens"],do_sample=controls["do_sample"],use_cache=controls["use_cache"])
        duration=time.monotonic()-t0
        answer_ids=generated[:,inputs["input_ids"].shape[1]:]
        text=processor.batch_decode(answer_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        (output/f"{record['id']}_completion.txt").write_text(text+"\n")
        predicted=parse_label(text)
        result={**record,"raw_text":text,"predicted_label":predicted,"format_valid":predicted is not None,
                "correct":predicted==record["gt_label"],"input_size":list(image.size),"processed_size":list(size),
                "pixel_values_sha256":pixel_hash,"input_ids_sha256":token_hash,
                "input_tokens":int(inputs["input_ids"].shape[1]),"completion_tokens":int(answer_ids.shape[1]),
                "hit_token_cap":int(answer_ids.shape[1])==controls["max_new_tokens"],"generation_seconds":duration,"fresh_context":True}
        save(output/f"{record['id']}_result.json",result)
        records.append(result)
        print(f"{record['id']}: {text!r}",flush=True)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected gradients")
    result={"status":"completed_isolated_crop_probe","created_utc":datetime.now(timezone.utc).isoformat(),
            "model":metadata,"generation_controls":controls,"prompt":PROMPT,"records":records,"summary":summary(records),
            "reference_result_sha256":plan["reference_result_sha256"],"source_image_sha256":plan["source_image_sha256"],
            "source_mask_sha256":plan["source_mask_sha256"],"reference_processor_and_generation_defaults_identical":True,
            "all_task_token_hashes_identical":True,"elapsed_seconds":time.monotonic()-started,
            "peak_gpu_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,"sam_used":False,"optimizer_steps":0,"policy_trainable_parameters":0,
            "scope":plan["scope"],"interpretation_limits":plan["interpretation_limits"]}
    save(output/"result.json",result)
    print(json.dumps(result["summary"]),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("prepare","run"))
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--reference-run",required=True,type=Path)
    args=parser.parse_args()
    (prepare if args.mode=="prepare" else run)(args.output,args.reference_run)
