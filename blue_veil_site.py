from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from blue_veil import fast_pred, first_pred, create_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def parse_classification_report(report):
    print(list(filter(None, report.split(' '))))
    '''for i in list(filter(None, report.split(' '))):
        print(i)'''
    return []#parsed_report

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return 'no_file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = fast_pred(file_path)
            accuracy_report = first_pred(file_path)  # Get accuracy from model creation
            
            # Parse classification report
            parsed_report = parse_classification_report(accuracy_report)
            return render_template('result.html', result=result, parsed_report=parsed_report)
    return render_template('upload.html')

@app.route('/result')
def result():
    return render_template('result.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
