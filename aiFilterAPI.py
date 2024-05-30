from flask import Flask, request, jsonify
from filter import generateCustomImage

app = Flask(__name__)

@app.route('/')
def home():
    image = request.args.get("image")
    prompt = request.args.get("prompt")
    if image and prompt :
        result = generateCustomImage(image, prompt)
        return result, 200
    return "Hello World", 200

if __name__ == "__main__" :
    app.run(debug=True)