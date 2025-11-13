from flask import Flask, render_template
from model import Model

app = Flask(__name__)

# Initialize model on startup
model = Model(model_id="gpt2-small", device="cuda")

@app.route('/')
def index():
    # Hardcoded prompt as specified
    prompt = "Hello, how are"
    
    # Get distributions for all positions
    data = model.get_all_positions_distn(prompt, topk=10)
    
    # Render template with all position data
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

