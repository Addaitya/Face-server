import os
from dotenv import load_dotenv
from flask import (
    Flask , request, jsonify
)
import numpy as np
import cv2
from utils.encoder import Encoder
from utils.db import PersonCollection

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY') 
URI= os.getenv('DB_URI')
ECD = Encoder()
PERSON = PersonCollection(URI)
COSIN = os.getenv('COSIN_INDEX')
SEARCH_FIELD = os.getenv('SEARCH_FIELD')


app = Flask(__name__)
# app = Blueprint('main', __name__)

@app.before_request
def auth():
    '''
    Check access token is valid or not 
    '''
    try:
        
        auth_header = request.headers.get('Authorization')
    except RuntimeError as e:
        return jsonify({"error": f"Request access issue: {str(e)}"}), 500

    if not auth_header:
        return jsonify({"error": "Authorization header missing"}), 401

    # Check if it's a Bearer token
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Invalid token type"}), 401

    # Extract the token
    token = auth_header.split(" ")[1]

    if token != SECRET_KEY:  
        return jsonify({"error": "Invalid token"}), 401
    


@app.post('/register_face')
def register_face():
    '''
    Here request provide name, student_id and face image as input.
    It stores the student info and face embeddings in the database. 
    '''

    try:
        print(f"request form {request.form}")
        print(f"request file {request.files}")
        student_name = request.form['name'] if 'name' in request.form else None
        student_id = request.form['student_id'] if 'student_id' in request.form else None
        face_image = request.files['face_image'] if 'face_image' in request.files else None

        
        if not student_name or not student_id or not face_image:
            return jsonify({"error": "missing field"}), 400
        
        if PERSON.check_person(student_id):
            return jsonify({"error": "This student_id already exist"}), 400
        
        bytes_data = face_image.read()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        face_boxes = ECD.get_faceboxes(img)
        if len(face_boxes) != 1:
            return jsonify({"error": f"Either no face is detected or more than one face is detected. FaceCount: {len(face_boxes)}"}), 400
        
        embeddings = ECD.encode(img)

        person_data = {
            "name": student_name,
            "student_id": student_id,
            "embedding": embeddings[0]
        }
        PERSON.add_person(person_data)

        return jsonify({
            'name': student_name, 
            "student_id": student_id
        })


    except Exception as e:
        print(f"Error in register_face: \n{e}")
        return jsonify({"error": "Internal Server error"}), 500
    

@app.post('/check_face')
def check_face():
    try:
        face_image = request.files['face_image'] if 'face_image' in request.files else None
        if not face_image:
            return jsonify({"error": "missing image"}), 400
        
        bytes_data = face_image.read()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_boxes = ECD.get_faceboxes(img)

        if len(face_boxes) == 0:
            return jsonify({"error": "No face detected"}), 400
        
        embeddings = ECD.encode(img)
        person_info = []
        for face_box, embedding in zip(face_boxes.tolist() ,embeddings):
            result = PERSON.search(
                embedding,
                index_name=COSIN,
                field=SEARCH_FIELD
            )
            if result:
                person_info.append({
                    "person": result[0], 
                    "face_box" : {
                        "x": face_box[0],
                        "y": face_box[1],
                        "w": face_box[2],
                        "h": face_box[3]
                    }
                })
            else:
                person_info.append({
                    "person": None, 
                    "face_box" : {
                        "x": face_box[0],
                        "y": face_box[1],
                        "w": face_box[2],
                        "h": face_box[3]
                    }
                })
        
        return jsonify(person_info)
    
    except Exception as e:
        print(f"Error in check_face: \n{e}")
        return jsonify({"error": "Internal Server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)