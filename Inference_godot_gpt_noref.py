import os
import base64
from openai import OpenAI
import json
from PIL import Image

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")
model = OpenAI(api_key=api_key)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_gpt(model, message):

    response = model.responses.create(
        model="gpt-5",
        input=message
    )

    # Print the model's response
    response = response.output_text
    return response

instruction = (
        "** Task Description: **\n"
        "You are a helpful assistant analyzing video game images and screenshots for glitches. "
        "You will be given a screenshot from a video game, and your job is to analyze the screenshot and determine "
        "whether it shows any of the glitch categories listed below. Return exactly this JSON:\n"
        "{\n"
        "    \"reasoning\": \"Brief explanation of how the screenshot was analyzed to identify (or rule out) a glitch.\",\n"
        "    \"glitch_detected\": true or false\n"
        "}\n"
        "** Known Glitch Categories (use strictly observable evidence): **\n"
        "clipping: A character or object visibly intersects or passes through another solid surface, or geometry overlaps in a way that could not occur with correct depth/occlusion.\n"
        "floating: A character or object is visibly not in contact with the ground or surface it should be resting on, defying expected physics or gravity.\n"
        "missing object: Parts of the character or object are unexpectedly missing, such as missing limbs, missing head/torso.\n"
        "corrupted texture: A surface shows clearly broken or incorrect texturing, such as scrambled/rainbow patterns, severe stretching.\n"
        "lighting issue: The scene lighting is clearly wrong, such as abnormal darkness/brightness, not explainable by natural scene change.\n"
        "** Important notes on rarity (avoid bias): **\n"
        "Clipping, floating, missing object, corrupted texture and lighting issue generally occur infrequently in shipped builds. Do not prefer “no glitch” just because they are rare. Base your decision only on visual evidence in the screenshot.\n"
    )

def test_single_image(test_path):
    base64_image = encode_image(test_path)
    image_source = {
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{base64_image}",
    }
    message = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                image_source,
            ],
        }
    ]
    return call_gpt(model, message)


def test_from_jsonl(test_jsonl_path, output_file="predictions.jsonl"):
    """Test all examples from a JSONL file"""
    results = []

    with open(test_jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            print('Data: ', line)

            obj = json.loads(line)
            test_path = obj["test_path"]
            true_label = obj["class"]

            print(f"Testing image {idx + 1}...")

            # Get prediction
            prediction = test_single_image(test_path)
            print('True label: ', true_label)
            print('Generated label: ', prediction)
            # Store result
            result = {
                "test_path": obj["test_path"],
                "true_label": true_label,
                "prediction": prediction,
            }

            results.append(result)

    # Save results
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    test_jsonl_path = 'godot_all_1000samples.jsonl'
    test_from_jsonl(test_jsonl_path, output_file="predictions_noref.jsonl")
