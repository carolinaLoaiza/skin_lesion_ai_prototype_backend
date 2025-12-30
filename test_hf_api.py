"""
Test script to call Hugging Face Gradio API via HTTP (without gradio_client)
"""

import requests
import json
from pathlib import Path

# Hugging Face Space URL
HF_SPACE_URL = "https://carolinaloaiza-skin-lesion-inference.hf.space"

def test_hf_api():
    """Test calling the HF Gradio API via HTTP"""

    # Test image path
    image_path = Path("tests/manual/data/images/B_0074139_55_F_LL_3p9.jpg")

    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"📤 Sending request to: {HF_SPACE_URL}")
    print(f"📸 Image: {image_path}")

    try:
        # Step 1: Upload the image file first
        print(f"\n⏳ Step 1: Uploading image...")

        with open(image_path, "rb") as f:
            files = {"files": ("image.jpg", f, "image/jpeg")}
            upload_response = requests.post(
                f"{HF_SPACE_URL}/gradio_api/upload",
                files=files,
                timeout=30
            )

        print(f"Upload Status Code: {upload_response.status_code}")

        if upload_response.status_code != 200:
            print(f"❌ Error uploading image: {upload_response.text}")
            return

        upload_data = upload_response.json()
        print(f"✅ Image uploaded: {upload_data}")

        # Get the uploaded file path
        if isinstance(upload_data, list) and len(upload_data) > 0:
            uploaded_file_path = upload_data[0]
        else:
            print(f"❌ Unexpected upload response format: {upload_data}")
            return

        # Step 2: Prepare payload with uploaded file path
        # Gradio expects a dict with 'path' key for images
        image_obj = {
            "path": uploaded_file_path,
            "url": f"{HF_SPACE_URL}/gradio_api/file={uploaded_file_path}",
            "meta": {"_type": "gradio.FileData"}
        }

        payload = {
            "data": [
                image_obj,           # Image object with path
                55,                   # age
                "female",            # sex
                "Left Leg",          # location
                3.9,                 # diameter
                True                 # include_shap
            ]
        }

        # Step 3: Call the predict endpoint to queue the job
        print(f"\n⏳ Step 2: Queueing prediction...")
        response = requests.post(
            f"{HF_SPACE_URL}/gradio_api/call/predict",
            json=payload,
            timeout=10
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Error queueing job: {response.text}")
            return

        # Get event ID from response
        event_data = response.json()
        event_id = event_data.get("event_id")

        if not event_id:
            print(f"❌ No event_id in response: {event_data}")
            return

        print(f"✅ Job queued with event_id: {event_id}")

        # Step 4: Poll for results using Server-Sent Events
        print(f"\n⏳ Step 3: Waiting for results...")

        sse_url = f"{HF_SPACE_URL}/gradio_api/call/predict/{event_id}"
        print(f"Polling: {sse_url}")

        # Use streaming to handle SSE
        response = requests.get(sse_url, stream=True, timeout=120)

        if response.status_code != 200:
            print(f"❌ Error getting results: {response.text}")
            return

        # Parse SSE stream
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                print(f"[DEBUG] Raw line: {line_str}")  # DEBUG

                # SSE format: "data: {...}"
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove "data: " prefix
                    print(f"[DEBUG] Data string: {data_str}")  # DEBUG

                    try:
                        data = json.loads(data_str)
                        print(f"[DEBUG] Parsed data: {data}")  # DEBUG

                        # Skip if data is None or empty
                        if not data:
                            continue

                        # Handle different response types
                        if isinstance(data, list):
                            # This is the final result from "event: complete"
                            print(f"\n✅ Prediction completed!")
                            print(f"\n📊 Response:")

                            if len(data) > 0:
                                prediction_json = data[0]

                                # Parse if it's a JSON string
                                if isinstance(prediction_json, str):
                                    parsed = json.loads(prediction_json)
                                    print(json.dumps(parsed, indent=2))
                                else:
                                    print(json.dumps(prediction_json, indent=2))
                            else:
                                print(json.dumps(data, indent=2))
                            break

                        elif isinstance(data, dict):
                            # Handle status events
                            if data.get("msg") == "process_starts":
                                print("  Processing...")

                            elif data.get("msg") == "estimation":
                                rank = data.get("rank")
                                queue_size = data.get("queue_size")
                                print(f"  Queue position: {rank}/{queue_size}")

                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        pass

    except requests.exceptions.Timeout:
        print("\n❌ Request timeout")
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection error - is the HF Space running?")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hf_api()
