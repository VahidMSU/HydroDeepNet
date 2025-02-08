from flask import Flask, send_from_directory, jsonify

app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/logout', methods=['GET'])
def logout():
    # Implement your logout logic here
    response = jsonify({"message": "User logged out successfully"})
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
