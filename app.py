from flask import request, jsonify,Flask, render_template
from flask_cors import CORS, cross_origin
import uuid
from CNN_Classifier.utils.common import decodeBase64ToImage
from CNN_Classifier.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    file_name = f"Image_{uuid.uuid4().hex}.jpg"
    decodeBase64ToImage(image, file_name)
    result = clApp.classifier.predict(file_name)
    return jsonify({
        "predicted_class": result["predicted_class"],
        "confidence_pct": round(float(result["confidence"]) * 100, 2),
    })



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080) #for AWS
