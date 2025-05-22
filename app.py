from flask import Flask, render_template, request, redirect, url_for, session, flash, g
import os
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
from datetime import datetime
import sqlite3
import hashlib
from flask import jsonify

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

# Configuration
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
DATABASE = 'database.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# Global model variable
model = None
model_path = os.path.join(MODEL_FOLDER, 'best.pt')  # Default model
app.jinja_env.globals.update(datetime=datetime)


def load_model(path):
    global model
    try:
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        model.eval()
        print(f"Model loaded from {path}")
    except Exception as e:
        print(f"Model loading error: {e}")
        model = None

# Load the default model at startup
# if os.path.exists(model_path):
#     load_model(model_path)

def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                value = datetime.strptime(value, '%Y-%m-%d')
            except ValueError:
                return value
    return value.strftime(format)

app.jinja_env.filters['datetimeformat'] = datetimeformat
def init_db():
    with app.app_context():
        db = sqlite3.connect(DATABASE)
        cursor = db.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                surname TEXT NOT NULL,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                user_role TEXT NOT NULL,
                email TEXT,
                skupina TEXT,
                specialization TEXT,
                department TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                surname TEXT NOT NULL,
                birth_date TEXT NOT NULL,
                medical_history TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
                analysis_result TEXT,
                analyzed_by INTEGER,
                FOREIGN KEY(patient_id) REFERENCES patients(id),
                FOREIGN KEY(analyzed_by) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                assigned_to INTEGER,
                created_by INTEGER,
                due_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(assigned_to) REFERENCES users(id),
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,
                model_used TEXT,
                start_date TEXT, 
                end_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                created_by INTEGER,
                FOREIGN KEY(created_by) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignment_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id INTEGER,
                filename TEXT,
                FOREIGN KEY(assignment_id) REFERENCES assignments(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignment_students (
                assignment_id INTEGER,
                student_id INTEGER,
                PRIMARY KEY (assignment_id, student_id),
                FOREIGN KEY(assignment_id) REFERENCES assignments(id),
                FOREIGN KEY(student_id) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                answers TEXT,
                score INTEGER,
                submitted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(assignment_id) REFERENCES assignments(id),
                FOREIGN KEY(student_id) REFERENCES users(id),
                UNIQUE(assignment_id, student_id)
            )
        ''')

        test_doctors = [
            ('Ján', 'Doktor', 'jan.doktor', hash_password('password123'), 'doctor', 'jan.doktor@mediscan.sk',
             'Radiology', 'Diagnostic Imaging'),
            ('Eva', 'Lekárka', 'eva.lekarka', hash_password('password456'), 'doctor', 'eva.lekarka@mediscan.sk',
             'Neurology', 'Neurology Department')
        ]

        test_students = [
            (
            'Peter', 'Študent', 'peter.student', hash_password('student123'), 'student', 'peter.student@university.sk'),
            ('Mária', 'Študentka', 'maria.studentka', hash_password('student456'), 'student',
             'maria.studentka@university.sk')
        ]

        test_patients = [
            ('Ján', 'Kováč', '1985-05-15', 'History of hypertension', 1),
            ('Anna', 'Vargová', '1978-11-22', 'Diabetes type 2', 1),
            ('Peter', 'Novák', '1992-03-08', 'No significant history', 2)
        ]

        for name, surname, username, password, role, email, *rest in test_doctors:
            try:
                if role == 'doctor':
                    specialization, department = rest
                    cursor.execute(
                        "INSERT OR IGNORE INTO users (name, surname, username, password, user_role, email, specialization, department) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (name, surname, username, password, role, email, specialization, department)
                    )
            except sqlite3.IntegrityError:
                pass

        for name, surname, username, password, role, email in test_students:
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO users (name, surname, username, password, user_role, email) VALUES (?, ?, ?, ?, ?, ?)",
                    (name, surname, username, password, role, email)
                )
            except sqlite3.IntegrityError:
                pass

        for name, surname, birth_date, medical_history, created_by in test_patients:
            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO patients (name, surname, birth_date, medical_history, created_by) VALUES (?, ?, ?, ?, ?)",
                    (name, surname, birth_date, medical_history, created_by)
                )
            except sqlite3.IntegrityError:
                pass

        db.commit()
        db.close()


def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def process_image(image_path, model_name=None):
    try:
        if model_name:
            model_path = os.path.join('models', model_name)
            if not os.path.exists(model_path):
                return {"success": False, "error": f"Model file {model_name} not found"}

            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                model.conf = 0.25  # сниженный порог уверенности
                model.iou = 0.45   # стандартный IoU
            except Exception as e:
                return {"success": False, "error": f"Error loading model: {str(e)}"}
        else:
            if not hasattr(process_image, 'default_model'):
                process_image.default_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                process_image.default_model.conf = 0.25
                process_image.default_model.iou = 0.45
            model = process_image.default_model

        if not os.path.exists(image_path):
            return {"success": False, "error": "Image file not found"}

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        results = model(img)
        detections = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

        return {
            "success": True,
            "detections": detections,
            "model_used": model_name or "default"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_used": model_name or "default"
        }


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path, model_name=None):
    try:
        if model_name:
            model_path = os.path.join('models', model_name)
            if not os.path.exists(model_path):
                return {"success": False, "error": f"Model file {model_name} not found"}

            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            except Exception as e:
                return {"success": False, "error": f"Error loading model: {str(e)}"}
        else:
            if not hasattr(process_image, 'default_model'):
                process_image.default_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model = process_image.default_model

        if not os.path.exists(image_path):
            return {"success": False, "error": "Image file not found"}

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        results = model(img)
        detections = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

        original_name = os.path.basename(image_path)
        output_filename = f"analyzed_{model_name or 'default'}_{original_name}"
        output_path = os.path.join('static', output_filename)

        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(img))
        plt.axis('off')

        for detection in detections:
            xmin, ymin, xmax, ymax = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, color='red', linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(xmin, ymin, f"{detection['name']} {detection['confidence']:.2f}",
                     color='white', backgroundcolor='red', fontsize=8)

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        return {
            "success": True,
            "detections": detections,
            "output_image": output_filename,
            "original_image": original_name,
            "model_used": model_name or "default"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model_used": model_name or "default"
        }

def calculate_age(birth_date_str):
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except:
        return "N/A"

def parse_end_date(s: str) -> datetime | None:
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


@app.route('/')
def home():
    if 'username' in session:
        if session.get('user_role') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patients_list'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            db = get_db()
            cursor = db.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()

            if user and user['password'] == hash_password(password):
                session['username'] = username
                session['user_role'] = user['user_role']
                session['user_name'] = f"{user['name']} {user['surname']}"

                if user['user_role'] == 'doctor':
                    return redirect(url_for('doctor_dashboard'))
                elif user['user_role'] == 'student':
                    return redirect(url_for('student_dashboard'))
                else:
                    return redirect(url_for('login'))

            flash('Invalid credentials', 'error')
        except sqlite3.Error as e:
            flash('Database error. Please try again.', 'error')
            app.logger.error(f"Database error: {e}")

    return render_template('login.html')


@app.route('/patients')
def patients_list():
    if 'username' not in session:
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM patients")
    patients = cursor.fetchall()

    return render_template('patients.html', patients=patients)


@app.route('/patient/<int:patient_id>')
def patient_detail(patient_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cursor.fetchone()

    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('patients_list'))

    cursor.execute("SELECT * FROM medical_images WHERE patient_id = ?", (patient_id,))
    images = cursor.fetchall()

    models = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template('patient_detail.html', patient=patient, images=images, available_models=models)


@app.route('/patient/<int:patient_id>/upload', methods=['GET', 'POST'])
def upload_image(patient_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    if session.get('user_role') != 'doctor':
        flash('Unauthorized action', 'error')
        return redirect(url_for('patient_detail', patient_id=patient_id))

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cursor.fetchone()

    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('patients_list'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Get current user ID
            cursor.execute("SELECT id FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()

            if user:
                cursor.execute(
                    "INSERT INTO medical_images (patient_id, filename, upload_date) VALUES (?, ?, ?)",
                    (patient_id, filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
                db.commit()
                flash('Image uploaded successfully', 'success')
                return redirect(url_for('patient_detail', patient_id=patient_id))

    return render_template('upload_image.html', patient=patient)


@app.route('/analyze/<int:patient_id>/<int:image_id>', methods=['GET', 'POST'])
def analyze_image(patient_id, image_id):
    selected_model = request.args.get('model') or session.get('last_used_model')

    if 'username' not in session:
        return redirect(url_for('login'))

    if session.get('user_role') != 'doctor':
        flash('Unauthorized action', 'error')
        return redirect(url_for('patient_detail', patient_id=patient_id))

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    patient = cursor.fetchone()

    cursor.execute("SELECT * FROM medical_images WHERE id = ? AND patient_id = ?", (image_id, patient_id))
    image = cursor.fetchone()

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'confirm' and 'analysis_result' in session:
            analysis_result = session['analysis_result']
            cursor.execute(
                "UPDATE medical_images SET analysis_result = ?, analyzed_by = ? WHERE id = ?",
                (json.dumps(analysis_result), session['username'], image_id)
            )
            db.commit()
            flash('Results confirmed', 'success')
            return redirect(url_for('patient_detail', patient_id=patient_id))
        elif action == 'retry':
            if selected_model:
                session['last_used_model'] = selected_model
            return redirect(url_for('analyze_image',
                                    patient_id=patient_id,
                                    image_id=image_id,
                                    model=selected_model))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image['filename'])
    result = process_image(image_path, model_name=selected_model)

    if not result.get('success'):
        flash(f"Analysis error: {result.get('error', 'Unknown error')}", 'error')
        return redirect(url_for('patient_detail', patient_id=patient_id))

    if selected_model:
        session['last_used_model'] = selected_model

    session['analysis_result'] = {
        'detections': result['detections'],
        'output_image': result['output_image'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'doctor': session['username'],
        'model_used': selected_model
    }

    return render_template('analysis.html',
                           patient=dict(patient),
                           image=dict(image),
                           image_path=url_for('static', filename=result['output_image']),
                           detections=result['detections'],
                           selected_model=selected_model)


@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    if session.get('user_role') != 'doctor':
        flash('You do not have permission to access this page', 'error')
        return redirect(url_for('patients_list'))

    print(f"Current path in doctor_dashboard: {request.path}")

    return render_template('doctor_dashboard.html')

@app.route('/student_dashboard')
def student_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    if session.get('user_role') != 'student':
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    return render_template('student_dashboard.html')


@app.route('/start-test/<int:assignment_id>')
def start_test(assignment_id):
    if 'username' not in session or session.get('user_role') != 'student':
        flash('Unauthorized', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()

    # Проверяем, назначен ли тест студенту
    cursor.execute('''
        SELECT a.* 
        FROM assignments a
        JOIN assignment_students ast ON a.id = ast.assignment_id
        JOIN users u ON ast.student_id = u.id
        WHERE a.id = ? AND u.username = ?
    ''', (assignment_id, session['username']))
    assignment = cursor.fetchone()

    if not assignment:
        flash('Test not found or not assigned to you', 'error')
        return redirect(url_for('assignments_list'))

    # Проверяем, не прошел ли уже студент этот тест
    cursor.execute('''
        SELECT 1 FROM test_results 
        WHERE assignment_id = ? AND student_id = (
            SELECT id FROM users WHERE username = ?
        )
    ''', (assignment_id, session['username']))
    if cursor.fetchone():
        flash('You have already completed this test', 'error')
        return redirect(url_for('assignment_detail', assignment_id=assignment_id))

    # Получаем изображения для тесgt
    cursor.execute('SELECT id, filename FROM assignment_images WHERE assignment_id = ?', (assignment_id,))
    images = [dict(img) for img in cursor.fetchall()]

    return render_template('start_test.html',
                           assignment=dict(assignment),
                           images=images
                           )


@app.route('/submit-test/<int:assignment_id>', methods=['POST'])
def submit_test(assignment_id):
    if 'username' not in session or session.get('user_role') != 'student':
        return jsonify({"success": False, "message": "Unauthorized"}), 403

    db = get_db()
    cursor = db.cursor()

    # 1. Проверка, что тест назначен студенту
    cursor.execute('''
        SELECT 1
          FROM assignment_students ast
          JOIN users u ON ast.student_id = u.id
         WHERE ast.assignment_id = ? AND u.username = ?
    ''', (assignment_id, session['username']))
    if not cursor.fetchone():
        return jsonify({"success": False, "message": "Test not assigned to you"}), 404

    # 2. Проверка, что студент ещё не сдавал
    cursor.execute('''
        SELECT 1 FROM test_results
         WHERE assignment_id = ?
           AND student_id = (SELECT id FROM users WHERE username = ?)
    ''', (assignment_id, session['username']))
    if cursor.fetchone():
        return jsonify({"success": False, "message": "You have already completed this test"}), 400

    # 3. Данные из фронта
    data = request.get_json()
    rectangles_data = data.get("rectangles", {})

    # 4. Список изображений задания
    cursor.execute('SELECT id, filename FROM assignment_images WHERE assignment_id = ?', (assignment_id,))
    images = cursor.fetchall()

    # Подготовка словарей
    student_rectangles = {}
    model_rectangles   = {}
    intersections      = {}
    scores             = []

    # Получаем id студента для названий файлов
    cursor.execute("SELECT id FROM users WHERE username = ?", (session['username'],))
    student_id = cursor.fetchone()['id']

    # 5. Обрабатываем каждое изображение
    for image in images:
        img_id = image['id']
        fname  = image['filename']
        key    = f'rectangles_{img_id}'

        # a) прямоугольники студента
        student_rects = rectangles_data.get(key, [])
        student_rectangles[str(img_id)] = student_rects

        # b) прямоугольники модели
        model_rects = get_model_detections(fname, assignment_id)
        model_rectangles[str(img_id)] = model_rects

        # c) пересечения
        img_inter = calculate_intersections(student_rects, model_rects)
        intersections[str(img_id)] = img_inter

        # d) оценка по изображению
        img_score = calculate_image_score(student_rects, model_rects, img_inter)
        scores.append(img_score)

    # 6. Итоговая оценка
    overall_score = int(sum(scores) / len(scores)) if scores else 0

    # 7. Генерация и сохранение combined-изображений
    combined_imgs = {}
    for image in images:
        img_id = image['id']
        orig   = image['filename']
        orig_path = os.path.join(app.config['UPLOAD_FOLDER'], orig)

        # Открываем исходник
        im = Image.open(orig_path).convert('RGB')
        draw = ImageDraw.Draw(im)

        # Вспомогательная функция
        def drect(r, color):
            x,y,w,h = r['x'], r['y'], r['width'], r['height']
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)

        # Рисуем: модель (синие), пересечения (зеленые), студент (красные)
        for r in model_rectangles[str(img_id)]:   drect(r, 'blue')
        for r in intersections[str(img_id)]:       drect(r, 'limegreen')
        for r in student_rectangles[str(img_id)]:  drect(r, 'red')

        # Сохраняем итоговую картинку
        combined_name = f"combined_{assignment_id}_{student_id}_{img_id}.png"
        combined_path = os.path.join(app.config['UPLOAD_FOLDER'], combined_name)
        im.save(combined_path)
        combined_imgs[str(img_id)] = combined_name

    # 8. Запись в БД со столбцом combined_images (JSON)
    cursor.execute('''
        INSERT INTO test_results
          (assignment_id, student_id, answers, score, combined_images)
        VALUES (?, (SELECT id FROM users WHERE username = ?), ?, ?, ?)
    ''', (
        assignment_id,
        session['username'],
        json.dumps({
            'student_rectangles': student_rectangles,
            'model_rectangles':   model_rectangles,
            'intersections':      intersections
        }),
        overall_score,
        json.dumps(combined_imgs)
    ))
    result_id = cursor.lastrowid
    db.commit()

    # 9. Ответ фронту
    return jsonify({
        "success": True,
        "message": "Test submitted successfully",
        "score": overall_score,
        "result_id": result_id
    })

@app.route('/test-result/<int:result_id>')
def test_result_view(result_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()

    # Получаем результат теста
    cursor.execute('SELECT * FROM test_results WHERE id = ?', (result_id,))
    result = cursor.fetchone()

    if not result:
        flash('Result not found', 'error')
        return redirect(url_for('student_dashboard'))

    status = result['status']

    # Проверяем, имеет ли пользователь право его видеть
    if session.get('user_role') == 'student':
        cursor.execute('SELECT id FROM users WHERE username = ?', (session['username'],))
        user_id = cursor.fetchone()['id']
        if result['student_id'] != user_id:
            flash('Access Denied', 'error')
            return redirect(url_for('student_dashboard'))

    answers = json.loads(result['answers'])

    # Получаем список изображений вместе с их ID
    cursor.execute('SELECT id, filename FROM assignment_images WHERE assignment_id = ?', (result['assignment_id'],))
    image_rows = cursor.fetchall()

    images_data = []
    for image in image_rows:
        img_id = image['id']
        filename = image['filename']
        key = f'rectangles_{img_id}'

        images_data.append({
            'filename': filename,
            'student_rects': answers['student_rectangles'].get(str(img_id), []),
            'model_rects': answers['model_rectangles'].get(str(img_id), []),
            'intersections': answers['intersections'].get(str(img_id), []),
            'score': calculate_image_score(
                answers['student_rectangles'].get(str(img_id), []),
                answers['model_rectangles'].get(str(img_id), []),
                answers['intersections'].get(str(img_id), [])
            )
        })
        print(answers)

    return render_template('test_result.html', images=images_data, score=result['score'], status=status)

@app.route('/change_password', methods=['POST'])
def change_password():
    # Только залогиненные пользователи могут менять пароль
    if 'username' not in session:
        return redirect(url_for('login'))

    old_pw     = request.form.get('old_password', '')
    new_pw     = request.form.get('new_password', '')
    confirm_pw = request.form.get('confirm_password', '')

    # Проверяем, что все поля заполнены
    if not old_pw or not new_pw or not confirm_pw:
        flash('Please fill in all fields.', 'error')
        return redirect(url_for('settings'))

    # Новые пароли совпадают?
    if new_pw != confirm_pw:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('settings'))

    db = get_db()
    cur = db.cursor()
    # Получаем текущего пользователя из БД
    cur.execute("SELECT password FROM users WHERE username = ?", (session['username'],))
    row = cur.fetchone()
    if not row:
        flash('User not found in database.', 'error')
        return redirect(url_for('settings'))

    # Проверяем старый пароль
    hashed_old = hash_password(old_pw)
    if row['password'] != hashed_old:
        flash('The old password was entered incorrectly..', 'error')
        return redirect(url_for('settings'))

    # Хешируем и сохраняем новый
    new_hashed = hash_password(new_pw)
    cur.execute(
        "UPDATE users SET password = ? WHERE username = ?",
        (new_hashed, session['username'])
    )
    db.commit()

    flash('Password successfully changed!', 'success')
    return redirect(url_for('settings'))

def get_model_detections(filename, assignment_id):
    """Получаем результаты работы модели для изображения"""
    db = get_db()
    cursor = db.cursor()

    # Получаем модель, используемую в этом задании
    cursor.execute('SELECT model_used FROM assignments WHERE id = ?', (assignment_id,))
    assignment = cursor.fetchone()
    model_name = assignment['model_used']

    # Анализируем изображение с помощью модели
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return []

    result = process_image(image_path, model_name=model_name)

    if not result.get('success'):
        print(f"[ERROR] Model processing failed: {result.get('error')}")
        return []

    detections = result.get('detections', [])
    print(f"[INFO] {filename} detections: {len(detections)}")

    # Преобразуем обнаружения в прямоугольники
    rectangles = []
    for detection in detections:
        print(detection)
        # Уменьшаем порог с 0.5 до 0.2
        if detection.get('confidence', 0) > 0.2 and detection['name'] != 'USG':
            rectangles.append({
                'x': detection['xmin'],
                'y': detection['ymin'],
                'width': detection['xmax'] - detection['xmin'],
                'height': detection['ymax'] - detection['ymin']
            })

    return rectangles

def calculate_intersections(student_rects, model_rects):
    """Вычисляем области пересечения между прямоугольниками студента и модели"""
    intersections = []

    for s_rect in student_rects:
        for m_rect in model_rects:
            # Вычисляем пересечение двух прямоугольников
            x_left = max(s_rect['x'], m_rect['x'])
            y_top = max(s_rect['y'], m_rect['y'])
            x_right = min(s_rect['x'] + s_rect['width'], m_rect['x'] + m_rect['width'])
            y_bottom = min(s_rect['y'] + s_rect['height'], m_rect['y'] + m_rect['height'])

            if x_right > x_left and y_bottom > y_top:
                intersections.append({
                    'x': x_left,
                    'y': y_top,
                    'width': x_right - x_left,
                    'height': y_bottom - y_top
                })

    return intersections


def calculate_image_score(student_rects, model_rects, intersections):
    if not model_rects:
        return 100 if not student_rects else 0

    # Площади
    total_model_area = sum(r['width'] * r['height'] for r in model_rects)
    total_student_area = sum(r['width'] * r['height'] for r in student_rects)
    intersection_area = sum(r['width'] * r['height'] for r in intersections)

    # union = A + B - intersection
    union_area = total_model_area + total_student_area - intersection_area
    if union_area <= 0:
        return 0

    iou = intersection_area / union_area
    return int(iou * 100)

@app.route('/test-history')
def test_history():
    if 'username' not in session or session.get('user_role') != 'student':
        return redirect(url_for('login'))

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cur = get_db().cursor()
    cur.execute("""  
          SELECT a.id, a.description, a.model_used, a.start_date, a.end_date,
                 tr.id AS result_id, tr.score
            FROM assignments a
            JOIN assignment_students ast ON a.id = ast.assignment_id
            LEFT JOIN test_results tr
              ON a.id = tr.assignment_id
             AND tr.student_id = (
                 SELECT id FROM users WHERE username = ?
             )
           WHERE ast.student_id = (
                 SELECT id FROM users WHERE username = ?
             )
        """, (session['username'], session['username']))
    rows = cur.fetchall()

    history = []
    for r in rows:
        rec = dict(r)
        end_str = rec['end_date']
        completed = rec['result_id'] is not None
        expired = (end_str < now_str) and not completed

        # <-- добавляем только, если тест уже завершён или просрочен
        if completed or expired:
            history.append({**rec, 'status': completed and 'completed' or 'expired'})

    return render_template('test_history.html', history=history)

@app.route('/students')
def students_list():
    if 'username' not in session:
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id, name, surname, email, skupina FROM users WHERE user_role = 'student'")
    students = cursor.fetchall()

    models = [f for f in os.listdir('models') if f.endswith('.pt')]

    return render_template('students.html', students=students, models=models)

@app.route('/add_student', methods=['POST'])
def add_student():
    if 'username' not in session or session.get('user_role') != 'doctor':
        return jsonify({"success": False, "message": "Access denied"}), 403

    data = request.form
    name = data.get('name')
    surname = data.get('surname')
    email = data.get('email')
    password = data.get('password')
    skupina = data.get('skupina')
    username = email.split('@')[0]

    db = get_db()
    cursor = db.cursor()

    try:
        cursor.execute('''
            INSERT INTO users (name, surname, username, password, user_role, email, skupina)
            VALUES (?, ?, ?, ?, 'student', ?, ?)
        ''', (name, surname, username, hash_password(password), email, skupina))
        db.commit()

        new_id = cursor.lastrowid
        return jsonify({
            "success": True,
            "student": {
                "id": new_id,
                "name": name,
                "surname": surname,
                "skupina": skupina
            }
        })
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email already exists"}), 409


@app.route('/student/<int:student_id>')
def student_detail(student_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ? AND user_role = 'student'", (student_id,))
    student = cursor.fetchone()

    if not student:
        flash('Student not found', 'error')
        return redirect(url_for('students_list'))

    return render_template('student_detail.html', student=student)


@app.route('/create_task', methods=['GET', 'POST'])
def create_task():
    if 'username' not in session or session.get('user_role') != 'doctor':
        flash('Access denied', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT id, name, surname FROM users WHERE user_role = 'student'")
    students = cursor.fetchall()

    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        assigned_to = request.form['assigned_to']
        due_date = request.form['due_date']

        cursor.execute("SELECT id FROM users WHERE username = ?", (session['username'],))
        creator = cursor.fetchone()

        cursor.execute('''
            INSERT INTO tasks (title, description, assigned_to, created_by, due_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, description, assigned_to, creator['id'], due_date))
        db.commit()
        flash('Task created successfully!', 'success')
        return redirect(url_for('doctor_dashboard'))

    return render_template('create_task.html', students=students)


@app.route('/create_assignment', methods=['POST'])
def create_assignment():
    print('starting to create assgnment')

    if 'username' not in session or session.get('user_role') != 'doctor':
        flash('Access denied', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cursor = db.cursor()

    try:
        start_date = request.form['start_date'].replace('T', ' ')
        end_date = request.form['end_date'].replace('T', ' ')

        if ':' in start_date and start_date.count(':') == 1:
            start_date += ':00'
        if ':' in end_date and end_date.count(':') == 1:
            end_date += ':00'

        cursor.execute("SELECT id FROM users WHERE username = ?", (session['username'],))
        creator = cursor.fetchone()

        student_ids = request.form.getlist('students')  # должно быть списком
        print('all student ids : ', student_ids)
        print('request', request.form.getlist)

        cursor.execute('''
            INSERT INTO assignments (description, model_used, start_date, end_date, created_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            request.form['description'],
            request.form['model'],
            start_date,
            end_date,
            creator['id']
        ))
        assignment_id = cursor.lastrowid

        # Сохраняем прикрепленные изображения
        if 'images' in request.files:
            files = request.files.getlist('images')
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    cursor.execute('''
                        INSERT INTO assignment_images (assignment_id, filename)
                        VALUES (?, ?)
                    ''', (assignment_id, filename))

        # Связываем студентов с заданием
        for student_id in student_ids:
            print('student #', student_id)
            cursor.execute('''
                INSERT INTO assignment_students (assignment_id, student_id)
                VALUES (?, ?)
            ''', (assignment_id, student_id))

        db.commit()
        print('Úloha bola úspešne vytvorená!')
        flash('Úloha bola úspešne vytvorená!', 'success')
        return redirect(url_for('doctor_assignments_list'))

    except Exception as e:
        print('errorrrrrrrr')
        db.rollback()
        flash(f'Chyba pri vytváraní úlohy: {str(e)}', 'error')
        return redirect(url_for('doctor_assignments_list'))

@app.route('/assignments')
def assignments_list():
    if 'username' not in session:
        return redirect(url_for('login'))

    db     = get_db()
    cursor = db.cursor()

    cursor.execute('''
        SELECT a.*, 
               (SELECT name || ' ' || surname
                  FROM users
                 WHERE id = a.created_by) AS creator_name
          FROM assignments a
          JOIN assignment_students ast
            ON a.id = ast.assignment_id
         WHERE ast.student_id = (
               SELECT id FROM users
                WHERE username = ?
         )
      ORDER BY a.created_at DESC
    ''', (session['username'],))
    raw = cursor.fetchall()
    assignments = [dict(a) for a in raw]

    now     = datetime.now()
    active  = []
    expired = []

    for a in assignments:
        # 1) Парсим start_date
        sd = a.get('start_date')
        if sd:
            for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                try:
                    a['start_date'] = datetime.strptime(sd, fmt)
                    break
                except ValueError:
                    a['start_date'] = None

        # 2) Парсим end_date
        ed = a.get('end_date')
        a_end = None
        if ed:
            for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
                try:
                    a_end = datetime.strptime(ed, fmt)
                    break
                except ValueError:
                    a_end = None
        a['end_date'] = a_end

        # 3) Проверяем, прошёл ли студент
        cursor.execute('''
            SELECT id
              FROM test_results
             WHERE assignment_id = ?
               AND student_id = (
                   SELECT id FROM users
                    WHERE username = ?
               )
        ''', (a['id'], session['username']))
        tr = cursor.fetchone()
        a['completed'] = bool(tr)
        if tr:
            a['result_id'] = tr['id']

        # 4) Куда класть?
        if a['completed']:
            expired.append(a)
        elif a_end and a_end < now:
            expired.append(a)
        else:
            active.append(a)

    return render_template(
        'assignments_student.html',
        active=active,
        expired=expired
    )

@app.route('/student/assignment/<int:assignment_id>')
def student_assignment_detail(assignment_id):
    if 'username' not in session or session.get('user_role')!='student':
        flash('Доступ запрещён', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor()

    # Основная инфа по заданию
    cur.execute('''
        SELECT a.*, u.name AS creator_name, u.surname AS creator_surname
        FROM assignments a
        JOIN users u ON a.created_by=u.id
        WHERE a.id=?
    ''', (assignment_id,))
    assignment = cur.fetchone()
    if not assignment:
        flash('Задание не найдено', 'error')
        return redirect(url_for('assignments_list'))

    # Все картинки из задания
    cur.execute('SELECT * FROM assignment_images WHERE assignment_id=?', (assignment_id,))
    images = cur.fetchall()

    # Проверяем, прошёл ли студент уже тест
    cur.execute('''
        SELECT id
        FROM test_results
        WHERE assignment_id=?
          AND student_id=(SELECT id FROM users WHERE username=?)
    ''', (assignment_id, session['username']))
    tr = cur.fetchone()
    completed = bool(tr)
    result_id = tr['id'] if tr else None

    return render_template('student_assignment_detail.html',
                           assignment=dict(assignment),
                           images=[dict(i) for i in images],
                           completed=completed,
                           result_id=result_id)

@app.route('/doctor/assignments')
def doctor_assignments_list():
    if 'username' not in session or session.get('user_role') != 'doctor':
        flash('Access Denied', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor()
    cur.execute('''
        SELECT
          a.id,
          a.description,
          a.model_used,
          a.start_date,
          a.end_date,
          (SELECT COUNT(*) 
             FROM assignment_students ast 
            WHERE ast.assignment_id = a.id
          ) AS student_count,
          EXISTS(
            SELECT 1 
              FROM test_results tr 
             WHERE tr.assignment_id = a.id
          ) AS is_submitted
        FROM assignments a
       WHERE a.created_by = (
             SELECT id 
               FROM users 
              WHERE username = ?
       )
    ORDER BY a.created_at DESC
    ''', (session['username'],))
    rows = cur.fetchall()
    assignments = [dict(r) for r in rows]

    # — конвертим строки дат в datetime
    for a in assignments:
        sd = a.get('start_date')
        if sd:
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
                try:
                    a['start_date'] = datetime.strptime(sd, fmt)
                    break
                except ValueError:
                    a['start_date'] = None

        ed = a.get('end_date')
        if ed:
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
                try:
                    a['end_date'] = datetime.strptime(ed, fmt)
                    break
                except ValueError:
                    a['end_date'] = None

    return render_template('doctor_assignments.html', assignments=assignments)

# --- Для врача ---
@app.route('/doctor/assignment/<int:assignment_id>')
def doctor_assignment_detail(assignment_id):
    if 'username' not in session or session.get('user_role') != 'doctor':
        flash('Access Denied', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor()

    # 1) Основная инфа по заданию
    cur.execute('''
        SELECT a.*, u.name AS creator_name, u.surname AS creator_surname
          FROM assignments a
          JOIN users u ON a.created_by = u.id
         WHERE a.id = ?
    ''', (assignment_id,))
    assignment = cur.fetchone()
    if not assignment:
        flash('Test not found', 'error')
        return redirect(url_for('doctor_assignments_list'))

    # 2) Все картинки, прикреплённые к этому заданию
    cur.execute('''
        SELECT id, filename
          FROM assignment_images
         WHERE assignment_id = ?
    ''', (assignment_id,))
    images = [dict(row) for row in cur.fetchall()]

    # 3) Список назначенных студентов
    cur.execute('''
        SELECT u.id, u.name, u.surname, u.skupina
          FROM assignment_students ast
          JOIN users u ON ast.student_id = u.id
         WHERE ast.assignment_id = ?
    ''', (assignment_id,))
    students = [dict(row) for row in cur.fetchall()]

    # 4) Результаты: вместе с JSON-строкой combined_images
    cur.execute('''
        SELECT tr.*,
               u.name || ' ' || u.surname AS student_name
          FROM test_results tr
          JOIN users u ON tr.student_id = u.id
         WHERE tr.assignment_id = ?
    ''', (assignment_id,))
    results = []
    for row in cur.fetchall():
        rec = dict(row)
        # Парсим JSON с именами composite-картинок
        combined_json = rec.get('combined_images')
        try:
            rec['combined_images'] = json.loads(combined_json) if combined_json else {}
        except json.JSONDecodeError:
            rec['combined_images'] = {}
        results.append(rec)

    return render_template(
        'doctor_assignment_detail.html',
        assignment=dict(assignment),
        images=images,
        students=students,
        results=results
    )

@app.route('/confirm-test-result/<int:result_id>', methods=['POST'])
def confirm_test_result(result_id):
    if 'username' not in session or session.get('user_role')!='doctor':
        flash('Access Denied', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor()
    cur.execute("UPDATE test_results SET status = 'approved' WHERE id = ?", (result_id,))
    db.commit()

    flash('The result has been approved.', 'success')
    cur.execute("SELECT assignment_id FROM test_results WHERE id = ?", (result_id,))
    aid = cur.fetchone()['assignment_id']
    return redirect(url_for('doctor_assignments_list'))


@app.route('/reject-test-result/<int:result_id>', methods=['POST'])
def reject_test_result(result_id):
    if 'username' not in session or session.get('user_role')!='doctor':
        flash('Access Denied', 'error')
        return redirect(url_for('login'))

    db = get_db()
    cur = db.cursor()
    cur.execute("UPDATE test_results SET status = 'rejected' WHERE id = ?", (result_id,))
    db.commit()

    flash('The result is rejected.', 'warning')
    cur.execute("SELECT assignment_id FROM test_results WHERE id = ?", (result_id,))
    aid = cur.fetchone()['assignment_id']
    return redirect(url_for('doctor_assignments_list'))

@app.route('/upload_model', methods=['GET', 'POST'])
def upload_model():
    if 'username' not in session or session.get('user_role') != 'doctor':
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'model_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['model_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and file.filename.endswith('.pt'):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['MODEL_FOLDER'], filename)
            file.save(save_path)

            # Reload model immediately after upload
            load_model(save_path)
            flash(f'Model "{filename}" uploaded and loaded successfully.', 'success')
            return redirect(url_for('doctor_dashboard'))
        else:
            flash('Only .pt files are allowed', 'error')

    return render_template('upload_model.html')


@app.template_filter('to_datetime')
def to_datetime(value):
    if isinstance(value, str):
        try:
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        except:
            pass
    return value


@app.route('/settings')
def settings():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('settings.html')

app.jinja_env.globals.update(calculate_age=calculate_age)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    os.makedirs(os.path.dirname(DATABASE) or os.path.curdir, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    init_db()

