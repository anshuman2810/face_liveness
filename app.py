import cv2
import numpy as np
import time
import base64
import threading
from collections import deque
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_socketio import SocketIO, emit
import pandas as pd
import os
import hashlib
import smtplib
import uuid
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.consumer.storage.session import SessionStorage
from requests_oauthlib import OAuth2Session
from authlib.integrations.flask_client import OAuth

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

class LivenessDetector:
    def __init__(self):
        # Load face and eye cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Variables for liveness detection
        self.face_history = deque(maxlen=30)  # Store face positions for motion detection
        self.eye_states = deque(maxlen=60)    # Store eye states for blink detection
        
        # Challenge variables
        self.liveness_score = 0
        self.challenge_active = False
        self.challenge_type = None
        self.challenge_start_time = 0
        self.challenge_completed = False
        self.frame_count = 0
        
    def texture_analysis(self, face_roi):
        """Analyze texture patterns in the face to detect printouts"""
        if face_roi.size == 0:
            return False
            
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate gradient magnitude using Sobel
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calculate texture variance
        texture_score = np.var(magnitude)
        
        # Threshold determined empirically
        return texture_score > 1000

    def detect_motion(self, face_history):
        """Detect facial movement across frames"""
        if len(face_history) < 5:
            return False, 0
        
        # Calculate centroid of face rectangle for each frame
        centroids = [(x + w/2, y + h/2) for (x, y, w, h) in face_history]
        
        # Calculate movement as the sum of distances between consecutive centroids
        movement = 0
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i-1][0]
            dy = centroids[i][1] - centroids[i-1][1]
            dist = np.sqrt(dx*dx + dy*dy)
            movement += dist
        
        # Return True if movement exceeds threshold
        return movement > 10, movement

    def detect_blinks(self, eye_states):
        """Count blinks from eye state history"""
        if len(eye_states) < 10:
            return 0
        
        # Count transitions from eyes open to eyes closed
        blinks = 0
        for i in range(1, len(eye_states)):
            if eye_states[i-1] and not eye_states[i]:
                blinks += 1
        
        return blinks

    def reflectance_analysis(self, face_roi):
        """Analyze light reflectance patterns on the face"""
        if face_roi.size == 0:
            return False
            
        # Convert to HSV for better lighting analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Extract the Value channel (brightness)
        _, _, v = cv2.split(hsv)
        
        # Calculate variance in brightness
        var_brightness = np.var(v)
        
        # Actual face should have varied lighting across the face
        return var_brightness > 150

    def process_frame(self, frame):
        """Process a single frame for liveness detection"""
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        liveness_confirmed = False
        status_message = ""
        
        if len(faces) > 0:
            # Sort faces by area (width * height) in descending order
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            # Process the largest face
            x, y, w, h = faces[0]
            
            # Add face to history for motion tracking
            self.face_history.append((x, y, w, h))
            
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            eye_region = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                eye_region,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Track if eyes are open
            eyes_detected = len(eyes) > 0
            self.eye_states.append(eyes_detected)
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(display_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            
            # Perform liveness checks
            motion_detected, motion_amount = self.detect_motion(self.face_history)
            texture_result = self.texture_analysis(face_roi)
            reflectance_result = self.reflectance_analysis(face_roi)
            blink_count = self.detect_blinks(list(self.eye_states))
            
            # Calculate overall liveness score
            self.liveness_score = (
                (2 if motion_detected else 0) + 
                (blink_count * 2) + 
                (3 if texture_result else 0) + 
                (2 if reflectance_result else 0)
            )
            
            # Display details on the frame
            cv2.putText(display_frame, f"Motion: {'Yes' if motion_detected else 'No'} ({motion_amount:.1f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Blinks: {blink_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Texture: {'Real' if texture_result else 'Fake'}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Reflectance: {'Real' if reflectance_result else 'Fake'}", 
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Liveness Score: {self.liveness_score}", 
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Periodic challenge
            self.frame_count += 1
            if not self.challenge_active and self.frame_count % 100 == 0 and not self.challenge_completed:
                self.challenge_active = True
                self.challenge_type = np.random.choice(["blink", "move"])
                self.challenge_start_time = time.time()
            
            # Handle challenges
            if self.challenge_active:
                if self.challenge_type == "blink":
                    cv2.putText(display_frame, "Challenge: Please blink twice", 
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if blink_count >= 2:
                        self.challenge_completed = True
                        self.challenge_active = False
                else:
                    cv2.putText(display_frame, "Challenge: Please move your head slightly", 
                                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if motion_amount > 50:
                        self.challenge_completed = True
                        self.challenge_active = False
                
                # Check challenge timeout
                if time.time() - self.challenge_start_time > 7.0:
                    self.challenge_active = False
                    cv2.putText(display_frame, "Challenge failed. Try again.", 
                                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Final liveness determination
            if self.challenge_completed and self.liveness_score >= 13:
                liveness_confirmed = True
                status_message = "LIVENESS CONFIRMED: Real Person"
                cv2.putText(display_frame, status_message, 
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif self.liveness_score < 12:
                status_message = "LIVENESS FAILED: Likely Spoof"
                cv2.putText(display_frame, status_message, 
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return display_frame, {
            'message': status_message, 
            'is_live': liveness_confirmed
        }

class UserDatabase:
    def __init__(self, db_path='users.xlsx'):
        self.db_path = db_path
        
        # Create the Excel file if it doesn't exist
        if not os.path.exists(db_path):
            df = pd.DataFrame(columns=[
                'username', 
                'email', 
                'password', 
                'reset_token', 
                'reset_token_expiry'
            ])
            df.to_excel(db_path, index=False)
    
    def user_exists(self, username=None, email=None):
        df = pd.read_excel(self.db_path)
        
        if username is not None:
            return not df[df['username'] == username].empty
        
        if email is not None:
            return not df[df['email'] == email].empty
        
        return False
    
    def get_user_by_email(self, email):
        df = pd.read_excel(self.db_path)
        user = df[df['email'] == email]
        return user.iloc[0] if not user.empty else None
    
    def validate_login(self, username, password):
        df = pd.read_excel(self.db_path)
        
        # Hash the password for comparison
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        user = df[(df['username'] == username) & (df['password'] == hashed_password)]
        return not user.empty
    
    def add_user(self, username, email, password):
        df = pd.read_excel(self.db_path)
        
        # Hash the password before storing
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        new_user = pd.DataFrame({
            'username': [username],
            'email': [email],
            'password': [hashed_password],
            'reset_token': [None],
            'reset_token_expiry': [None]
        })
        
        # Append new user and save
        df = pd.concat([df, new_user], ignore_index=True)
        df.to_excel(self.db_path, index=False)
    
    def generate_reset_token(self, email):
        df = pd.read_excel(self.db_path)
        
        # Generate unique reset token
        reset_token = str(uuid.uuid4())
        reset_token_expiry = datetime.now() + timedelta(hours=1)
        
        # Update user's reset token and expiry
        df.loc[df['email'] == email, 'reset_token'] = reset_token
        df.loc[df['email'] == email, 'reset_token_expiry'] = reset_token_expiry
        
        # Save updated dataframe
        df.to_excel(self.db_path, index=False)
        
        return reset_token
    
    def validate_reset_token(self, token):
        df = pd.read_excel(self.db_path)
        
        # Find user with matching token
        user = df[df['reset_token'] == token]
        
        if user.empty:
            return None
        
        # Check token expiry
        expiry = pd.to_datetime(user['reset_token_expiry'].iloc[0])
        
        if expiry < datetime.now():
            return None
        
        return user.iloc[0]
    
    def reset_password(self, token, new_password):
        df = pd.read_excel(self.db_path)
        
        # Find user with matching token
        user = df[df['reset_token'] == token]
        
        if user.empty:
            return False
        
        # Hash new password
        hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
        
        # Update password and clear reset token
        df.loc[df['reset_token'] == token, 'password'] = hashed_password
        df.loc[df['reset_token'] == token, 'reset_token'] = None
        df.loc[df['reset_token'] == token, 'reset_token_expiry'] = None
        
        # Save updated dataframe
        df.to_excel(self.db_path, index=False)
        
        return True

class EmailService:
    def __init__(self, sender_email, sender_password, smtp_server='smtp.gmail.com', smtp_port=587):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
    
    def send_reset_email(self, recipient_email, reset_token):
        # Create message
        message = MIMEMultipart()
        message['From'] = self.sender_email
        message['To'] = recipient_email
        message['Subject'] = 'Password Reset Request'
        
        # Create reset link
        reset_link = f"http://localhost:5000/reset_password/{reset_token}"
        
        # Email body
        body = f"""
        Hello,

        You have requested to reset your password. 
        Please click the link below to reset your password:

        {reset_link}

        This link will expire in 1 hour.

        If you did not request a password reset, please ignore this email.

        Best regards,
        Liveness Detection App
        """
        
        message.attach(MIMEText(body, 'plain'))
        
        try:
            # Establish secure connection
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            
            # Send email
            server.send_message(message)
            server.quit()
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False



# Configure Email Service (Replace with your email credentials)
email_service = EmailService(
    sender_email='abdprajapati090@gmail.com',  # Replace with your email
    sender_password='cyoe lktn uyxw tgof'   # Use App Password for Gmail
)
    

# Flask and SocketIO setup
app = Flask(__name__)
app.secret_key = 'J@y@nshum@nprojec#'  # Important for sessions
socketio = SocketIO(app)

app.config["GOOGLE_OAUTH_CLIENT_ID"] = "921549520582-43vg6ccs3m9aeuhmug2k5l6jtbtvch6e.apps.googleusercontent.com"  # Replace with your client ID
app.config["GOOGLE_OAUTH_CLIENT_SECRET"] = "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"  # Replace with your client secret

# Create Google blueprint
google_bp = make_google_blueprint(
    scope=["profile", "email"],
    storage=SessionStorage(),
    redirect_to="index"
)
app.register_blueprint(google_bp, url_prefix="/login")


# Global variables
detector = LivenessDetector()
user_db = UserDatabase()
capture = None
is_streaming = False

def video_stream():
    global capture, is_streaming
    capture = cv2.VideoCapture(0)
    
    if not capture.isOpened():
        print("Error: Could not open camera.")
        return
    
    while is_streaming:
        ret, frame = capture.read()
        if not ret:
            break
        
        processed_frame, status = detector.process_frame(frame)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Emit frame and status
        socketio.emit('video_feed', frame_b64)
        socketio.emit('liveness_status', status)
        
        socketio.sleep(0.033)  # ~30 FPS
    
    if capture:
        capture.release()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if user_db.validate_login(username, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/login/google')
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    
    # Get user info from Google
    resp = google.get("/oauth2/v2/userinfo")
    assert resp.ok, resp.text
    google_info = resp.json()
    
    # Extract email and name
    email = google_info.get('email')
    name = google_info.get('name')
    
    # Check if user exists in database
    if not user_db.user_exists(email=email):
        # Create a new user with a temporary password
        temp_password = str(uuid.uuid4())
        user_db.add_user(name, email, temp_password)
    
    # Log in the user
    session['username'] = name
    session['email'] = email
    flash('Google login successful!', 'success')
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation checks
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        if user_db.user_exists(username=username):
            flash('Username already exists', 'error')
            return render_template('signup.html')
        
        if user_db.user_exists(email=email):
            flash('Email already registered', 'error')
            return render_template('signup.html')
        
        # Add user to database
        user_db.add_user(username, email, password)
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        
        # Check if email exists
        user = user_db.get_user_by_email(email)
        
        if user is not None:
            # Generate reset token
            reset_token = user_db.generate_reset_token(email)
            
            # Send reset email
            if email_service.send_reset_email(email, reset_token):
                flash('Password reset link sent to your email.', 'success')
            else:
                flash('Failed to send reset email. Please try again.', 'error')
        else:
            flash('No account found with this email.', 'error')
    
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # Validate token
    user = user_db.validate_reset_token(token)
    
    if user is None:
        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        # Validate passwords
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('reset_password.html', token=token)
        
        # Reset password
        if user_db.reset_password(token, new_password):
            flash('Password reset successfully. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Failed to reset password.', 'error')
    
    return render_template('reset_password.html', token=token)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@socketio.on('connect')
def handle_connect():
    global is_streaming
    if 'username' in session and not is_streaming:
        is_streaming = True
        socketio.start_background_task(video_stream)

@socketio.on('disconnect')
def handle_disconnect():
    global is_streaming, capture
    is_streaming = False
    if capture:
        capture.release()

if __name__ == '__main__':
    socketio.run(app, debug=True)