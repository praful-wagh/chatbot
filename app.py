from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import log

app = Flask(__name__)

predict = PredictPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET','POST'])
def chat():
    if request.method=='GET':
        predict.load_model()
        return render_template('home.html')
    else:
        data = request.get_json()
        user_message = data['message']
        log('User: ' + user_message)
        bot_response = predict.ask(user_message)
        response = {'botResponse': bot_response}
        log('Bot: ' + bot_response)
        return jsonify(response)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)