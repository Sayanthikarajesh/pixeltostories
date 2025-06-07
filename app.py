import os
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload and output directories if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400

    # Clean up previous outputs
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            filenames.append(filename)

    if not filenames:
        return jsonify({'error': 'No valid files uploaded'}), 400

    try:
        subprocess.run(['python', 'storynew.py'], check=True)
        
        # Get the paths of generated files
        output_files = {
            'captions': os.path.join(OUTPUT_FOLDER, 'captions_and_entities.txt'),
            'story': os.path.join(OUTPUT_FOLDER, 'generated_story.txt'),
            'graph': os.path.join(OUTPUT_FOLDER, 'knowledge_graph.jpg'),
            'audio': os.path.join(OUTPUT_FOLDER, 'audio.wav')  # Added audio file path
        }
        
        # Check if files exist
        for key, path in output_files.items():
            if not os.path.exists(path):
                # Just log a warning for audio file, don't fail if it's missing
                if key == 'audio':
                    print(f"Warning: Audio file was not generated at {path}")
                    output_files['audio'] = None
                else:
                    return jsonify({'error': f'Processing completed but {key} file was not generated'}), 500
        
        return jsonify({
            'message': 'Processing complete! Your story and visualizations are ready.',
            'output_files': output_files,
            'uploaded_files': filenames
        })
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/assets/<path:filename>')
def serve_logo(filename):
    return send_from_directory('assets', filename)

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)