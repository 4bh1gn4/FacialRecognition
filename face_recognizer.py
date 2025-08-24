import cv2, os, time, json
import numpy as np
from datetime import datetime
from pathlib import Path

# ---------------------------------
DATA_DIR = Path("faces_db")           # where face images live
MODEL_PATH = Path("lbph_model.xml")   # the trained model file
LABELS_PATH = Path("labels.json")     # maps numeric labels -> names
SAMPLES_PER_PERSON = 30               # images captured during enrollment
FACE_SIZE = (200, 200)                # normalized face crop size
CONF_THRESHOLD = 80                   # lower is better; <= this = recognized
CAM_INDEX = 0                         # try 1 or 2 if you have multiple cameras
RES_W, RES_H = 1920, 1080             # request 1080p from cam
# ---------------------------------

DATA_DIR.mkdir(exist_ok=True)

# Load Haar face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    raise RuntimeError("Could not load Haar cascade. Check your OpenCV install.")

# Create recognizer (LBPH = simple & local)
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

def load_label_map():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

def save_label_map(label_map):
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)

def build_label_map_from_dirs():
    # stable order: sort by folder name
    people = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    return {idx: name for idx, name in enumerate(people)}

def load_dataset():
    images, labels = [], []
    label_map = build_label_map_from_dirs()
    inv = {v: k for k, v in label_map.items()}
    for person in label_map.values():
        for imgf in (DATA_DIR / person).glob("*.png"):
            img = cv2.imread(str(imgf), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            images.append(img)
            labels.append(inv[person])
    return images, np.array(labels, dtype=np.int32), label_map

def train_model():
    images, labels, label_map = load_dataset()
    if len(images) < 10 or len(set(labels)) == 0:
        return False, {}
    recognizer.train(images, labels)
    recognizer.write(str(MODEL_PATH))
    save_label_map(label_map)
    return True, label_map

def load_or_train():
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        recognizer.read(str(MODEL_PATH))
        return True, load_label_map()
    ok, label_map = train_model()
    return ok, label_map

def collect_samples(name, cam, samples=SAMPLES_PER_PERSON):
    person_dir = DATA_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    while saved < samples:
        ok, frame = cam.read()
        if not ok: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80,80))
        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, FACE_SIZE)
            cv2.imwrite(str(person_dir / f"{int(time.time()*1000)}.png"), face)
            saved += 1
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"Capturing {name}: {saved}/{samples}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Enroll", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("Enroll")

def main():
    conf_threshold = CONF_THRESHOLD
    cam = cv2.VideoCapture(CAM_INDEX)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)

    if not cam.isOpened():
        print("Camera not opening. Try CAM_INDEX=1 or 2, or close Zoom/Teams.")
        return

    ok, label_map = load_or_train()
    if not ok:
        print("No trained model yet. Press N to enroll your first person.")
        label_map = {}

    while True:
        ok, frame = cam.read()
        if not ok: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80,80))

        for (x,y,w,h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            name_text = "Unknown"
            try:
                if MODEL_PATH.exists() and label_map:
                    label, conf = recognizer.predict(face)
                    if conf <= conf_threshold and label in label_map:
                        name_text = f"{label_map[label]} ({conf:.0f})"
                    else:
                        name_text = f"Unknown ({conf:.0f})"
            except cv2.error:
                name_text = "Unknown"

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(frame, name_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, "N: Enroll  |  +/-: threshold  |  Q: Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"Threshold={conf_threshold}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)


        cv2.imshow("Face Recognition (LBPH)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            # enroll flow
            print("\nType the name to enroll (letters/numbers/spaces are fine):")
            cam.release()
            cv2.destroyAllWindows()
            cam = cv2.VideoCapture(CAM_INDEX)
            name = input("> ").strip()
            if name:
                collect_samples(name, cam, samples=SAMPLES_PER_PERSON)
                ok, label_map = train_model()
                if ok: print("Model trained.")
                else: print("Not enough data yet—add more samples/people.")
        elif key in (ord('+'), ord('=')):  # looser (more likely to label)
            conf_threshold = min(200, conf_threshold + 5)
        elif key == ord('-'):              # stricter (more Unknowns)
            conf_threshold = max(30, conf_threshold - 5)
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
