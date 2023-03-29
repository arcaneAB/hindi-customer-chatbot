from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process():
    user_input = request.form['message']
    # You can add your logic to process the user input and generate a response
    # Here, we'll just return a simple response
    response = "Thanks for your message: " + user_input
    return {'response': response}

if __name__ == '__main__':
    app.run(debug=True)
