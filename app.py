from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import replicate
import cloudinary.uploader
import tempfile
import os

# Set up environment variables

cloudinary.config(
    cloud_name='dh3irgunk',
    api_key='243325685945982',
    api_secret='CLnmnzuyKQ0b_zbllkg-ZJ_lb-A'
)

app = Flask(__name__)
model = replicate

# Render live transcription page
@app.route("/")
def live_transcription():
    return render_template("index.html")

# Render audio upload page
@app.route("/upload")
def upload():
    return render_template("upload.html")

# Function to transcribe audio using Whisper
@app.route("/process-audio", methods=["POST"])
def process_audio_data():
    audio_file = request.files["audio"]

    if audio_file:
        print("Processing audio...")
        # Save the audio file with a secure filename
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(tempfile.gettempdir(), filename)
        audio_file.save(audio_path)

        try:
            # Upload the audio file to Cloudinary
            upload_result = cloudinary.uploader.upload(audio_path, resource_type="video")
            temp_audio_url = upload_result["secure_url"]

            output = model.run(
                "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
                input={
                    "task": "transcribe",
                    "audio": temp_audio_url,
                    "language": "english",
                    "timestamp": "chunk",
                    "batch_size": 64,
                    "diarise_audio": False,
                },
            )

            print(output)
            if(output!="Thank You"):
                results = output["text"]

            return jsonify({"transcript": results})
        except Exception as e:
            print(f"Error running Replicate model: {e}")
            return jsonify({"error": "Error processing audio"})
    else:
        return jsonify({"error": "No audio file uploaded"})

# Function to handle audio file uploads and transcriptions
@app.route("/upload-audio", methods=["POST"])
def upload_audio():
    audio_file = request.files["audio"]
    prompt = request.form.get('prompt', 'You are a summarizing agent')

    if audio_file:
        print("Processing uploaded audio...")
        # Save the audio file with a secure filename
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(tempfile.gettempdir(), filename)
        audio_file.save(audio_path)

        try:
            # Upload the audio file to Cloudinary
            upload_result = cloudinary.uploader.upload(audio_path, resource_type="video")
            temp_audio_url = upload_result["secure_url"]

            output = model.run(
                "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
                input={
                    "task": "transcribe",
                    "audio": temp_audio_url,
                    "language": "english",
                    "timestamp": "chunk",
                    "batch_size": 64,
                    "diarise_audio": False,
                },
            )

            print(output)
            results = output["text"]

            return jsonify({"transcript": results})
        except Exception as e:
            print(f"Error running Replicate model: {e}")
            return jsonify({"error": "Error processing audio"})
    else:
        return jsonify({"error": "No audio file uploaded"})

# Function to generate suggestions using Mixtral
@app.route("/get-suggestion", methods=["POST"])
def get_suggestion():
    print("Getting suggestion...")
    data = request.get_json()  # Parse JSON data from the request
    transcript = data.get("transcript", "")  # Extract transcript
    prompt_text = data.get("prompt", "")  # Extract prompt text

    prompt = f"""
    {transcript}
    ------
    {prompt_text}
    """

    suggestion = ""
    # The meta/llama-2-7b-chat model can stream output as it's running.
    for event in replicate.stream(
        "meta/llama-2-7b-chat",
        input={
            "top_k": 0,
            "top_p": 1,
            "prompt": prompt,
            "temperature": 0.75,
            "system_prompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            "length_penalty": 1,
            "max_new_tokens": 800,
            "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
            "presence_penalty": 0
        },
    ):
        suggestion += str(event)  # Accumulate the output

    return jsonify({"suggestion": suggestion})  # Send as JSON response

# if __name__ == "__main__":
#     app.run(debug=False,host="0.0.0.0")
