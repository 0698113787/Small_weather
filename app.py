from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import random

# Initialize Flask app
app = Flask(__name__)

# Generate synthetic weather data for training
def generate_weather_data(n_samples=1000):
    """Generate synthetic weather data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    humidity = np.random.normal(60, 20, n_samples)  # Average humidity 60%, std 20
    pressure = np.random.normal(1013, 30, n_samples)  # Average pressure 1013 hPa, std 30
    
    # Generate temperature with some realistic relationships
    # Higher humidity tends to correlate with higher temperature
    # Higher pressure tends to correlate with clearer, potentially warmer weather
    temperature = (
        15 +  # Base temperature
        0.2 * humidity +  # Humidity effect
        0.05 * (pressure - 1013) +  # Pressure effect
        np.random.normal(0, 5, n_samples)  # Random noise
    )
    
    # Ensure reasonable ranges
    humidity = np.clip(humidity, 0, 100)
    pressure = np.clip(pressure, 950, 1050)
    temperature = np.clip(temperature, -10, 40)
    
    return pd.DataFrame({
        'humidity': humidity,
        'pressure': pressure,
        'temperature': temperature
    })

# Create and train the model
print("Generating synthetic weather data...")
weather_data = generate_weather_data(1000)

# Prepare features and target
X = weather_data[['humidity', 'pressure']]
y = weather_data['temperature']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather Temperature Predictor</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            padding: 30px;
        }
        h1 {
            text-align: center;
            color: #2d3436;
            margin-bottom: 30px;
            font-size: 2.2em;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #636e72;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
            transition: border-color 0.3s;
        }
        input[type="number"]:focus {
            outline: none;
            border-color: #74b9ff;
        }
        .predict-btn {
            width: 100%;
            background: linear-gradient(135deg, #00b894, #00a085);
            color: white;
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .predict-btn:hover {
            transform: translateY(-2px);
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
        }
        .temperature {
            font-size: 2.5em;
            font-weight: bold;
            color: #2d3436;
            margin: 10px 0;
        }
        .model-info {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 14px;
            color: #636e72;
        }
        .sample-data {
            margin-top: 20px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌤️ Weather Temperature Predictor</h1>
        
        <form id="weatherForm">
            <div class="form-group">
                <label for="humidity">Humidity (%):</label>
                <input type="number" id="humidity" name="humidity" min="0" max="100" 
                       value="65" step="0.1" required>
            </div>
            
            <div class="form-group">
                <label for="pressure">Atmospheric Pressure (hPa):</label>
                <input type="number" id="pressure" name="pressure" min="950" max="1050" 
                       value="1013" step="0.1" required>
            </div>
            
            <button type="submit" class="predict-btn">🔮 Predict Temperature</button>
        </form>
        
        <div id="result" class="result" style="display: none;">
            <h3>Predicted Temperature:</h3>
            <div id="temperature" class="temperature"></div>
            <p>Based on the given humidity and pressure conditions</p>
        </div>

        <div class="sample-data">
            <h4>💡 Try these sample values:</h4>
            <p><strong>Summer day:</strong> Humidity: 75%, Pressure: 1015 hPa</p>
            <p><strong>Winter day:</strong> Humidity: 45%, Pressure: 1020 hPa</p>
            <p><strong>Rainy day:</strong> Humidity: 90%, Pressure: 1005 hPa</p>
        </div>

        <div class="model-info">
            <h4>📊 Model Information:</h4>
            <p><strong>Algorithm:</strong> Linear Regression</p>
            <p><strong>R² Score:</strong> {{ r2_score }}</p>
            <p><strong>Training Data:</strong> 1000 synthetic weather samples</p>
            <p><strong>Features:</strong> Humidity and Atmospheric Pressure</p>
        </div>
    </div>

    <script>
        document.getElementById('weatherForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const humidity = document.getElementById('humidity').value;
            const pressure = document.getElementById('pressure').value;
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    humidity: parseFloat(humidity),
                    pressure: parseFloat(pressure)
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('temperature').textContent = data.temperature + '°C';
                document.getElementById('result').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error making prediction. Please try again.');
            });
        });
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, r2_score=f"{r2:.3f}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        
        # Make prediction
        prediction = model.predict([[humidity, pressure]])[0]
        
        return jsonify({
            'temperature': f"{prediction:.1f}",
            'humidity': humidity,
            'pressure': pressure
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    """API endpoint to get model information"""
    return jsonify({
        'model_type': 'Linear Regression',
        'features': ['humidity', 'pressure'],
        'r2_score': float(r2),
        'mse': float(mse),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    })

if __name__ == '__main__':
    print("\n🌤️ Weather Prediction App is starting...")
    print("📊 Model trained and ready!")
    print("🚀 Open http://127.0.0.1:5000 in your browser")
    print("\n" + "="*50)
    app.run(debug=True, host='127.0.0.1', port=5000)