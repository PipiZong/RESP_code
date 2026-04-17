import os
from tqdm import tqdm
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import json
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
    load_in_4bit = True
)

FastVisionModel.for_inference(model)

def parse_json_response(response):
    text = response.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    text = text[start:]
    if "}" in text:
        text = text[:text.rfind("}") + 1]
    else:
        # Try to close a truncated JSON object
        text = text + "}"

    return text


def call_qwen(tokenizer, message, log_file, test_frame, ref_frame):

    input_text = tokenizer.apply_chat_template(message, add_generation_prompt = True)

    test_img = Image.open(test_frame).convert("RGB")
    if ref_frame is not None:
        ref_img = Image.open(ref_frame).convert("RGB")

        inputs = tokenizer(
            images = [ref_img, test_img],
            text = input_text,
            add_special_tokens = False,
            truncation = False,  # avoid dropping image tokens
            return_tensors = "pt",
        ).to("cuda")
    else:
        inputs = tokenizer(
            images = test_img,
            text = input_text,
            add_special_tokens = False,
            truncation = False,  # avoid dropping image tokens
            return_tensors = "pt",
        ).to("cuda")

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    generated_ids = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 10240,
                    use_cache = True, temperature = 0.1, do_sample = False)

    output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens = True)
    response = parse_json_response(response)
    predict_label = 1 if json.loads(response)['glitch_detected'] else 0

    # write response to log_file
    with open(log_file, 'a') as f:
        f.write(
            f'Video: {test_frame}, Ref: {ref_frame}, Response: {response}, Prediction: {predict_label}\n')
    return predict_label



file = open('glitch_with_ref_godot', 'r')
context_question_1 = file.read()
file.close()

file = open('glitch_without_ref_godot', 'r')
context_question_2 = file.read()
file.close()

file = open('glitch_with_ref_buggy_godot', 'r')
context_question_3 = file.read()
file.close()

log_file = 'qwen3vl_8b_previous_godot.txt'
res_dict = {}
folders_path = "godot_video_folders.txt"
with open(folders_path, "r", encoding="utf-8") as f:
    video_folders = [line.strip() for line in f if line.strip()]
idx = 0
for folder in tqdm(video_folders):
    idx += 1
    if folder.startswith('.'):
        continue
    print('Processing folder: ', folder, idx)
    file_list = sorted(os.listdir(folder))
    for i in range(len(file_list)):
        file = file_list[i]
        test_frame = os.path.join(folder, file)
        if i==0:
            message = [
                {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": context_question_2}
                ]}
        ]

            res = call_qwen(tokenizer, message, log_file, test_frame, None)
        else:
            ref_file = file_list[i-1]
            ref_frame = os.path.join(folder, ref_file)
            if res == 0:
                message = [
                    {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": context_question_1}
                ]}
                ]
                res = call_qwen(tokenizer, message, log_file, test_frame, ref_frame)
            else:
                message = [
                    {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": context_question_3}
                ]}
                ]
                res = call_qwen(tokenizer, message, log_file, test_frame, ref_frame)
        













