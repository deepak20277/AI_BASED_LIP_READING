from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained model safely
model = load_model('model/lip_model_ctc.h5', compile=False)
print("✅ Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
