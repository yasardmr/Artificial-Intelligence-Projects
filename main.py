import torch
import cv2
import numpy as np
import os
import json
from collections import deque, defaultdict
import time
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import shutil
from sklearn.model_selection import train_test_split
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
class EnhancedModelUpdater:
    def __init__(self, model, initial_threshold=0.6, min_samples_for_update=100):
        self.model = model
        self.confidence_threshold = initial_threshold
        self.performance_history = []
        self.calibration_history = []
        self.categorized_detections = defaultdict(list)
        self.last_evaluation_time = datetime.datetime.now()
        self.evaluation_interval = datetime.timedelta(days=7)  # Haftalık değerlendirme
        self.min_samples_for_update = min_samples_for_update

    def categorize_detection(self, detection, confidence):
        if confidence > 0.7:
            category = 'high'
        elif confidence > 0.5:
            category = 'medium'
        else:
            category = 'low'
        self.categorized_detections[category].append(detection)

    def evaluate_performance(self, X_test, y_test):
        with torch.no_grad():
            predictions = self.model(X_test)
            predicted_labels = predictions.argmax(dim=1)
            predicted_probs = nn.functional.softmax(predictions, dim=1)

        accuracy = accuracy_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels, average='weighted')
        recall = recall_score(y_test, predicted_labels, average='weighted')
        f1 = f1_score(y_test, predicted_labels, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def evaluate_calibration(self, X_test, y_test):
        with torch.no_grad():
            predictions = self.model(X_test)
            predicted_probs = nn.functional.softmax(predictions, dim=1).max(dim=1)[0]

        prob_true, prob_pred = calibration_curve(y_test, predicted_probs.numpy(), n_bins=10)
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        return calibration_error

    def manual_review_required(self, confidence):
        return confidence < 0.5  # Düşük güvenilirlikli tespitler için manuel inceleme

    def active_learning_sample(self, n_samples=10):
        low_confidence_samples = self.categorized_detections['low']
        return np.random.choice(low_confidence_samples, min(n_samples, len(low_confidence_samples)), replace=False)

    def periodic_evaluation(self):
        current_time = datetime.datetime.now()
        if current_time - self.last_evaluation_time > self.evaluation_interval:
            print("Performing periodic evaluation...")
            current_performance = self.evaluate_performance(self.X_test, self.y_test)
            current_calibration = self.evaluate_calibration(self.X_test, self.y_test)

            if current_performance['f1'] > self.performance_history[-1]['f1']:
                self.confidence_threshold = min(self.confidence_threshold + 0.05, 0.9)
                print(f"Performance improved. Confidence threshold increased to {self.confidence_threshold}")
            else:
                self.confidence_threshold = max(self.confidence_threshold - 0.05, 0.5)
                print(f"Performance did not improve. Confidence threshold decreased to {self.confidence_threshold}")

            self.performance_history.append(current_performance)
            self.calibration_history.append(current_calibration)
            self.last_evaluation_time = current_time

    def update_model(self, X_train, y_train, X_test, y_test, epochs=5, lr=0.001):
        if len(X_train) < self.min_samples_for_update:
            print(f"Not enough samples for update. Current: {len(X_train)}, Required: {self.min_samples_for_update}")
            return self.model

        self.X_test, self.y_test = X_test, y_test  # Store for periodic evaluation
        self.periodic_evaluation()

        print("Updating model...")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Store the current model state
        previous_state = self.model.state_dict().copy()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        self.model.eval()
        updated_performance = self.evaluate_performance(X_test, y_test)
        updated_calibration = self.evaluate_calibration(X_test, y_test)

        print(f"Model updated. New performance: {updated_performance}")
        print(f"New calibration error: {updated_calibration}")

        # Check if the new model performs better
        if len(self.performance_history) > 0:
            if updated_performance['f1'] <= self.performance_history[-1]['f1']:
                print("New model doesn't perform better. Reverting to previous state.")
                self.model.load_state_dict(previous_state)
                return self.model

        self.performance_history.append(updated_performance)
        self.calibration_history.append(updated_calibration)

        return self.model

class AutoLabeledDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Burada, veri formatınıza göre uygun dönüşümleri yapmanız gerekebilir
        # Örnek: bbox, confidence, class_id
        bbox = torch.tensor(item[:4], dtype=torch.float32)
        confidence = torch.tensor(item[4], dtype=torch.float32)
        class_id = torch.tensor(item[5], dtype=torch.long)
        return bbox, confidence, class_id
class DualModelSystem:
    def __init__(self, object_model_path, space_model_path):
        self.object_model = torch.hub.load('E:/yolov9_3/pythonProject1/yolov9', 'custom', path=object_model_path,
                                           source='local')
        self.space_model = torch.hub.load('E:/yolov9_2/pythonProject1/yolov9', 'custom', path=space_model_path,
                                          source='local')
        self.object_model_updater = EnhancedModelUpdater(self.object_model)
        self.space_model_updater = EnhancedModelUpdater(self.space_model)
        self.model_weights = {'object': 0.5, 'space': 0.5}
        self.detection_history = {'object': deque(maxlen=30), 'space': deque(maxlen=30)}
        self.shelf_layout = None
        self.blurThreshold = 35
        self.auto_labeled_data = {'object': [], 'space': []}
        self.manual_review_queue = []
        self.active_learning_queue = []
        self.last_frame = None
        self.output_folder = 'detected_objects'
        os.makedirs(self.output_folder, exist_ok=True)
        self.last_update_time = time.time()
        self.performance_history = {'object': [], 'space': []}
        self.last_save_time = time.time()
        self.save_interval = 300  # 5 dakika (saniye cinsinden)
        self.output_folder = 'detected_objects'
        self.classes = {0: 'object', 1: 'space'}
        for class_name in self.classes.values():
            os.makedirs(os.path.join(self.output_folder, class_name), exist_ok=True)

    def prepare_class_specific_yolo_dataset(self, source_folder, destination_folder, class_id, train_ratio=0.7,
                                            val_ratio=0.2, create_test_set=False):
        class_name = self.classes[class_id]
        class_folder = os.path.join(source_folder, class_name)

        # Hedef klasörleri oluştur
        os.makedirs(os.path.join(destination_folder, class_name, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(destination_folder, class_name, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(destination_folder, class_name, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(destination_folder, class_name, 'labels', 'val'), exist_ok=True)
        if create_test_set:
            os.makedirs(os.path.join(destination_folder, class_name, 'images', 'test'), exist_ok=True)
            os.makedirs(os.path.join(destination_folder, class_name, 'labels', 'test'), exist_ok=True)

        # Kaynak klasördeki tüm jpg dosyalarını listele
        image_files = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]

        if not image_files:
            print(f"Uyarı: '{class_folder}' klasöründe hiç jpg dosyası bulunamadı.")
            return

        # Veri setini böl
        if create_test_set:
            train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
            val_files, test_files = train_test_split(temp_files,
                                                     train_size=val_ratio / (val_ratio + (1 - train_ratio - val_ratio)),
                                                     random_state=42)
        else:
            train_files, val_files = train_test_split(image_files, train_size=train_ratio / (train_ratio + val_ratio),
                                                      random_state=42)

        def copy_files(file_list, subset):
            for file in file_list:
                # Orijinal görüntüyü kopyala
                shutil.copy(
                    os.path.join(class_folder, file),
                    os.path.join(destination_folder, class_name, 'images', subset, file)
                )

                # Etiket dosyasını kopyala
                txt_file = file.replace('.jpg', '.txt')
                txt_path = os.path.join(class_folder, txt_file)
                if os.path.exists(txt_path):
                    shutil.copy(
                        txt_path,
                        os.path.join(destination_folder, class_name, 'labels', subset, txt_file)
                    )
                else:
                    print(f"Uyarı: '{txt_file}' etiket dosyası bulunamadı.")

        # Dosyaları kopyala
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        if create_test_set:
            copy_files(test_files, 'test')

        # YOLO için yapılandırma dosyası oluştur
        with open(os.path.join(destination_folder, class_name, 'data.yaml'), 'w') as f:
            f.write(f"train: {os.path.join(destination_folder, class_name, 'images', 'train')}\n")
            f.write(f"val: {os.path.join(destination_folder, class_name, 'images', 'val')}\n")
            if create_test_set:
                f.write(f"test: {os.path.join(destination_folder, class_name, 'images', 'test')}\n")
            f.write("nc: 1\n")  # Tek sınıf
            f.write(f"names: ['{class_name}']\n")

        print(
            f"{class_name.capitalize()} için YOLO veri seti hazırlandı: {os.path.join(destination_folder, class_name)}")
        print(f"Eğitim seti: {len(train_files)} görüntü")
        print(f"Doğrulama seti: {len(val_files)} görüntü")
        if create_test_set:
            print(f"Test seti: {len(test_files)} görüntü")

    def manual_class_specific_yolo_dataset_preparation(self):
        source_folder = self.output_folder
        destination_folder = 'yolo_datasets'

        print("\nSınıfa Özel YOLO Veri Seti Hazırlama")
        print("====================================")
        print(f"Kaynak klasör: {source_folder}")
        print(f"Hedef klasör: {destination_folder}")

        create_test_set = input("Test seti oluşturmak istiyor musunuz? (e/h): ").lower() == 'e'

        for class_id, class_name in self.classes.items():
            user_input = input(
                f"{class_name.capitalize()} sınıfı için YOLO veri seti hazırlamak istiyor musunuz? (e/h): ")
            if user_input.lower() == 'e':
                self.prepare_class_specific_yolo_dataset(source_folder, destination_folder, class_id,
                                                         create_test_set=create_test_set)
                print(f"{class_name.capitalize()} sınıfı için YOLO veri seti hazırlama işlemi tamamlandı.")
            else:
                print(f"{class_name.capitalize()} sınıfı için YOLO veri seti hazırlama işlemi iptal edildi.")

    def save_detection(self, frame, detection, class_id):
        class_name = self.classes[class_id]

        # Benzersiz bir dosya adı oluştur
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.jpg"

        # Tüm kareyi kaydet
        save_path = os.path.join(self.output_folder, class_name, filename)
        cv2.imwrite(save_path, frame)

        # Bounding box bilgilerini txt dosyasına kaydet
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(self.output_folder, class_name, txt_filename)

        # YOLO formatında bounding box bilgilerini yaz
        img_height, img_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, detection[:4])
        confidence = detection[4]

        x_center = (x1 + x2) / (2 * img_width)
        y_center = (y1 + y2) / (2 * img_height)
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}")


    def detect(self, img, model_type='object'):
        model = self.object_model if model_type == 'object' else self.space_model
        results = model(img)
        return results.xyxy[0].cpu().numpy()

    def apply_weights(self, detections, detection_type):
        if len(detections) > 0:
            detections[:, 4] *= self.model_weights[detection_type]
        return detections

    def shutdown(self):
        print("Program kapatılıyor, veriler kaydedildi.")

    def process_detection(self, detection, confidence, detection_type, current_frame):
        updater = self.object_model_updater if detection_type == 'object' else self.space_model_updater

        x1, y1, x2, y2 = map(int, detection[:4])
        object_crop = current_frame[y1:y2, x1:x2]

        # Bulanıklık kontrolü
        blurValue = int(cv2.Laplacian(object_crop, cv2.CV_64F).var())

        # Eğer bulanıksa, sadece blur değerini döndür
        if blurValue <= self.blurThreshold:
            return blurValue

        updater.categorize_detection(detection, confidence)

        if updater.manual_review_required(confidence):
            return blurValue, 'manual_review'
        elif confidence > updater.confidence_threshold:
            return blurValue, 'high_confidence'
        else:
            return blurValue, 'discard'

    def is_valid_position(self, detection):
        # Implement position validation logic
        return True

    def is_temporally_consistent(self, detection, detection_type):
        # Implement temporal consistency check
        return True

    def analyze_shelf_layout(self, frame):
        # Implement shelf layout analysis
        self.shelf_layout = None  # Replace with actual analysis,

    def process_frame(self, frame):
        object_boxes = self.detect(frame, 'object')
        object_boxes = self.apply_weights(object_boxes, 'object')

        space_boxes = self.detect(frame, 'space')
        space_boxes = self.apply_weights(space_boxes, 'space')

        processed_frame = frame.copy()
        object_count = 0
        space_count = 0

        for detection in object_boxes:
            result = self.process_detection(detection, detection[4], 'object', frame)
            if isinstance(result, tuple):
                blur_value, action = result
                if action in ['manual_review', 'high_confidence']:
                    self.save_detection(frame, detection, 0)  # 0 for object class
                    object_count += 1
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Urun {detection[4]:.2f} Blur: {blur_value}'
                    cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        for detection in space_boxes:
            result = self.process_detection(detection, detection[4], 'space', frame)
            if isinstance(result, tuple):
                blur_value, action = result
                if action in ['manual_review', 'high_confidence']:
                    self.save_detection(frame, detection, 1)  # 1 for space class
                    space_count += 1
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f'Bosluk {detection[4]:.2f} Blur: {blur_value}'
                    cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return processed_frame, object_count, space_count

        if processed_object_boxes or processed_space_boxes:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base_filename = f"{timestamp}"
            frame_filename = f"frame_{base_filename}.jpg"
            txt_filename = f"frame_{base_filename}.txt"

            # Resmi kaydet
            frame_path = os.path.join(self.output_folder, frame_filename)
            cv2.imwrite(frame_path, frame_to_save)

            # Tespit bilgilerini YOLO formatında TXT olarak kaydet
            txt_path = os.path.join(self.output_folder, txt_filename)
            img_height, img_width = frame.shape[:2]

            with open(txt_path, 'w') as f:
                # Nesneler için class_id 0
                for *box, blur in processed_object_boxes:
                    x_center = (box[0] + box[2]) / (2 * img_width)
                    y_center = (box[1] + box[3]) / (2 * img_height)
                    width = (box[2] - box[0]) / img_width
                    height = (box[3] - box[1]) / img_height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Boşluklar için class_id 1
                for *box, blur in processed_space_boxes:
                    x_center = (box[0] + box[2]) / (2 * img_width)
                    y_center = (box[1] + box[3]) / (2 * img_height)
                    width = (box[2] - box[0]) / img_width
                    height = (box[3] - box[1]) / img_height
                    f.write(f"1 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(
                f"Saved {len(processed_object_boxes)} objects and {len(processed_space_boxes)} spaces to {txt_filename}")

            # Dosya oluşturma zamanlarını eşitle
            os.utime(txt_path, (os.path.getatime(frame_path), os.path.getmtime(frame_path)))

        processed_frame = frame_to_save

        return processed_frame, len(processed_object_boxes), len(processed_space_boxes)

    def draw_boxes(self, img, boxes, color, label_prefix):
        img_copy = img.copy()
        for *xyxy, conf, cls, blur_value in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f'{label_prefix} {conf:.2f} Blur: {blur_value}'
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img_copy
    def prepare_data_for_update(self, detection_type):
            all_data = self.auto_labeled_data[detection_type] + [sample for sample, label in self.active_learning_queue
                                                                 if label == 1]

            if len(all_data) < self.object_model_updater.min_samples_for_update:
                print(f"Not enough data for {detection_type} model update.")
                return None, None, None, None

            # Veri dengeleme
            class_counts = defaultdict(int)
            for sample in all_data:
                class_counts[int(sample[5])] += 1

            max_samples = max(class_counts.values())
            balanced_data = []
            for cls in class_counts:
                cls_samples = [sample for sample in all_data if int(sample[5]) == cls]
                balanced_samples = resample(cls_samples, n_samples=max_samples, replace=True)
                balanced_data.extend(balanced_samples)

            np.random.shuffle(balanced_data)

            split_index = int(0.8 * len(balanced_data))
            train_data = balanced_data[:split_index]
            test_data = balanced_data[split_index:]

            X_train = torch.tensor([sample[:4] for sample in train_data]).float()
            y_train = torch.tensor([sample[5] for sample in train_data]).long()

            X_test = torch.tensor([sample[:4] for sample in test_data]).float()
            y_test = torch.tensor([sample[5] for sample in test_data]).long()

            return X_train, y_train, X_test, y_test

    def update_models(self):
            object_data = self.prepare_data_for_update('object')
            space_data = self.prepare_data_for_update('space')

            if object_data[0] is not None:
                self.object_model = self.object_model_updater.update_model(*object_data)
            if space_data[0] is not None:
                self.space_model = self.space_model_updater.update_model(*space_data)

            self.auto_labeled_data = {'object': [], 'space': []}
            self.active_learning_queue = []

    def manual_update_interface(self):
            print("\nManual Update Interface")
            print("1. Update Object Model")
            print("2. Update Space Model")
            print("3. Update Both Models")
            print("4. Cancel")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                self.manual_model_update('object')
            elif choice == '2':
                self.manual_model_update('space')
            elif choice == '3':
                self.manual_model_update('object')
                self.manual_model_update('space')
            elif choice == '4':
                print("Update cancelled.")
            else:
                print("Invalid choice. Update cancelled.")

    def manual_model_update(self, model_type):
            data = self.prepare_data_for_update(model_type)
            if data[0] is not None:
                if model_type == 'object':
                    self.object_model = self.object_model_updater.update_model(*data)
                else:
                    self.space_model = self.space_model_updater.update_model(*data)
                print(f"{model_type.capitalize()} model updated successfully.")
            else:
                print(f"Not enough data to update {model_type} model.")

    def manual_review_interface(self):
        low_confidence_dir = "low_confidence"
        image_files = [f for f in os.listdir(low_confidence_dir) if f.endswith(".jpg")]

        if not image_files:
            print("No low confidence detections to review.")
            return

        print("\nManual Review Mode")
        print("------------------")
        print("Instructions:")
        print("- Press 'y' to accept the detection")
        print("- Press 'n' to reject the detection")
        print("- Press 'q' to quit the review process")
        print("------------------")

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(low_confidence_dir, image_file)
            json_path = os.path.join(low_confidence_dir, f"{base_name}.json")

            # JSON dosyasından tespit bilgilerini oku
            with open(json_path, 'r') as f:
                detection_info = json.load(f)

            original_img = cv2.imread(image_path)
            review_img = original_img.copy()

            # Tespit kutusunu çiz
            x1, y1, x2, y2 = map(int, detection_info["bbox"])
            cv2.rectangle(review_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Bilgileri görüntüye ekle
            info_text = [
                f"Type: {detection_info['detection_type']}",
                f"Confidence: {detection_info['confidence']:.2f}",
                f"Class: {detection_info['class']}"
            ]
            for i, text in enumerate(info_text):
                cv2.putText(review_img, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Görüntüyü büyüt
            scale_percent = 200  # orijinal boyutun %200'ü
            width = int(review_img.shape[1] * scale_percent / 100)
            height = int(review_img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_img = cv2.resize(review_img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("Manual Review", resized_img)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):  # Accept detection
                    detection = np.array(detection_info["bbox"] +
                                         [detection_info["confidence"], detection_info["class"]])
                    self.auto_labeled_data[detection_info["detection_type"]].append(detection)
                    print(f"Detection accepted and added to {detection_info['detection_type']} dataset.")
                    break
                elif key == ord('n'):  # Reject detection
                    print("Detection rejected.")
                    break
                elif key == ord('q'):  # Quit review process
                    print("Manual review process terminated.")
                    cv2.destroyAllWindows()
                    return

            cv2.destroyWindow("Manual Review")
            os.remove(image_path)
            os.remove(json_path)

        print("All low confidence detections have been reviewed.")
    def active_learning_interface(self):
            active_learning_samples = self.object_model_updater.active_learning_sample(5)
            active_learning_samples.extend(self.space_model_updater.active_learning_sample(5))

            for sample in active_learning_samples:
                x1, y1, x2, y2, conf, cls = sample
                review_img = self.last_frame.copy()
                cv2.rectangle(review_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.putText(review_img, f"Confidence: {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.imshow("Active Learning", review_img)
                print("Is this detection correct? (y/n)")
                key = cv2.waitKey(0) & 0xFF

                if key == ord('y'):
                    self.active_learning_queue.append((sample, 1))  # 1 for correct
                elif key == ord('n'):
                    self.active_learning_queue.append((sample, 0))  # 0 for incorrect

                cv2.destroyWindow("Active Learning")

            print(f"Added {len(self.active_learning_queue)} samples for active learning.")

    def inspect_auto_labeled_data(self):
        print("\nAuto-Labeled Data Inspection")
        print("============================")

        for detection_type in self.auto_labeled_data:
            data = self.auto_labeled_data[detection_type]
            print(f"\nDetection Type: {detection_type}")
            print(f"Total samples: {len(data)}")

            if len(data) > 0:
                # Sınıf dağılımını hesapla
                classes = [int(d[5]) for d in data]
                unique_classes, class_counts = np.unique(classes, return_counts=True)

                print("\nClass Distribution:")
                for cls, count in zip(unique_classes, class_counts):
                    print(f"  Class {cls}: {count} samples")

                # Güven değeri istatistikleri
                confidences = [d[4] for d in data]
                print(f"\nConfidence Statistics:")
                print(f"  Min: {min(confidences):.3f}")
                print(f"  Max: {max(confidences):.3f}")
                print(f"  Mean: {np.mean(confidences):.3f}")
                print(f"  Median: {np.median(confidences):.3f}")

                # Güven değeri histogramını çiz
                plt.figure(figsize=(10, 5))
                plt.hist(confidences, bins=20, edgecolor='black')
                plt.title(f'Confidence Distribution for {detection_type}')
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.show()

                # Örnek veri göster
                print("\nSample Data:")
                for i in range(min(5, len(data))):
                    print(f"  Sample {i + 1}: {data[i]}")

            else:
                print("No data available for this detection type.")

        print("\nInspection complete.")

    def update_model(self, model_type, train_loader, test_loader, epochs=10, learning_rate=0.001):
        model = self.object_model if model_type == 'object' else self.space_model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Kayıp: {total_loss/len(train_loader):.4f}")

        return model

    def check_and_update_models(self, min_samples=1000, update_interval=24 * 60 * 60):
        current_time = time.time()
        if current_time - self.last_update_time > update_interval:
            for model_type in ['object', 'space']:
                if len(self.auto_labeled_data[model_type]) >= min_samples:
                    print(f"Updating {model_type} model...")
                    self.update_model(model_type)
            self.last_update_time = current_time

    def evaluate_model(self, model, test_loader):
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def manual_model_update(self, model_type):
        print(f"\n{model_type.capitalize()} modelini güncelleme işlemi başlatılıyor...")

        current_performance = self.evaluate_model(
            self.object_model if model_type == 'object' else self.space_model,
            DataLoader(AutoLabeledDataset(self.auto_labeled_data[model_type]), batch_size=32)
        )

        updated_performance = self.update_model(model_type)

        if updated_performance is None:
            print("Model güncellenemedi. Yeterli veri yok.")
            return

        print("\nPerformans Karşılaştırması:")
        print(f"Metrik      Önceki       Yeni")
        print("---------------------------------")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            print(f"{metric:<12} {current_performance[metric]:.4f}     {updated_performance[metric]:.4f}")

        if updated_performance['f1'] > current_performance['f1']:
            print("\nYeni model daha iyi performans gösteriyor.")
            user_input = input("Modeli güncellemek istiyor musunuz? (e/h): ")
            if user_input.lower() == 'e':
                # Modeli kaydet
                torch.save(self.object_model.state_dict() if model_type == 'object' else self.space_model.state_dict(),
                           f"{model_type}_model_updated.pt")
                print(f"{model_type.capitalize()} modeli güncellendi ve kaydedildi.")
                # Performans geçmişini güncelle
                self.performance_history[model_type].append(updated_performance)
                # auto_labeled_data'yı temizle
                self.auto_labeled_data[model_type] = []
                self.save_auto_labeled_data()
            else:
                print("Model güncelleme iptal edildi.")
        else:
            print("\nYeni model daha iyi performans göstermiyor. Güncelleme önerilmez.")
def main():
    print("Main function started.")
    object_model_path = 'E:/yolov9_3/pythonProject1/yolov9/best (2).pt'
    space_model_path = 'E:/yolov9_2/pythonProject1/yolov9/best (1).pt'
    output_folder = 'E:/yolov9_3/pythonProject1/output'

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs("low_confidence", exist_ok=True)
    os.makedirs("detected_objects", exist_ok=True)  # Yeni tespit edilen nesneler için klasör
    model_system = DualModelSystem(object_model_path, space_model_path)

    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, object_count, space_count = model_system.process_frame(frame)

        cv2.imshow('Gerçek Zamanlı Tespit', processed_frame)
        cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:06d}.jpg'), processed_frame)

        print(f"Kare {frame_count}: Ürün sayısı: {object_count}, Boşluk sayısı: {space_count}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            print("Manuel inceleme modu etkinleştirildi.")
            cv2.destroyAllWindows()  # Mevcut pencereleri kapat
            model_system.manual_review_interface()
            cv2.namedWindow('Gerçek Zamanlı Tespit')  # Yeni pencere aç
        elif key == ord('s'):
            print("Auto-labeled veriler manuel olarak kaydediliyor...")
            model_system.save_auto_labeled_data()
            print("Kayıt tamamlandı.")
        elif key == ord('y'):
            print("Sınıfa özel YOLO veri seti hazırlama modu etkinleştirildi.")
            cv2.destroyAllWindows()
            model_system.manual_class_specific_yolo_dataset_preparation()
            cv2.namedWindow('Gerçek Zamanlı Tespit')

        # Otomatik güncelleme kontrolü

        frame_count += 1

        # Program kapanmadan önce son kez kaydet
    cap.release()
    cv2.destroyAllWindows()
    model_system.shutdown()

if __name__ == "__main__":
    main()