import cv2
import numpy as np
import time
import sys
import os

# ------------------------------------------------------------
# Класс для отслеживания HP-бара
# ------------------------------------------------------------
class HPBarDetector:
    def __init__(self):
        self.hsv_lower = np.array([35, 50, 50])
        self.hsv_upper = np.array([85, 255, 255])
        self.roi = None
        self.full_width = None
        self.smooth_percent = 0
        self.alpha = 0.3

    def set_roi(self, roi, full_width=None):
        self.roi = roi
        if full_width is not None:
            self.full_width = full_width

    def set_hsv_range(self, lower, upper):
        self.hsv_lower = lower
        self.hsv_upper = upper

    def detect(self, frame):
        if self.roi is None:
            return 0
        x, y, w, h = self.roi
        roi_img = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        col_sum = np.sum(mask == 255, axis=0)
        threshold = h * 0.3
        filled_cols = col_sum > threshold
        filled_width = np.sum(filled_cols)
        if self.full_width is None:
            self.full_width = w
        percent = int(filled_width / self.full_width * 100)
        self.smooth_percent = self.alpha * percent + (1 - self.alpha) * self.smooth_percent
        return int(self.smooth_percent)

    def get_mask(self, frame):
        if self.roi is None:
            return None
        x, y, w, h = self.roi
        roi_img = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return mask

# ------------------------------------------------------------
# Класс для детекции активации умения (по миганию)
# ------------------------------------------------------------
class SkillFlashDetector:
    def __init__(self):
        self.roi = None
        self.prev_brightness = 0
        self.smooth = 0.8
        self.threshold = 30
        self.cooldown = 0
        self.cooldown_frames = 10
        self.is_active = False

    def set_roi(self, roi):
        self.roi = roi
        self.prev_brightness = 0

    def update(self, frame):
        if self.roi is None:
            return False
        x, y, w, h = self.roi
        roi_img = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        diff = abs(mean - self.prev_brightness)
        self.prev_brightness = self.smooth * mean + (1 - self.smooth) * self.prev_brightness
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        if diff > self.threshold:
            self.is_active = True
            self.cooldown = self.cooldown_frames
            return True
        else:
            self.is_active = False
            return False

# ------------------------------------------------------------
# Основной класс приложения
# ------------------------------------------------------------
class ArucoRectifier:
    def __init__(self, camera_id=0, screen_width=1920, screen_height=1080,
                 base_marker_size=70, base_margin=80,
                 base_grid_cols=14, base_grid_rows=8,
                 aruco_dict_id=cv2.aruco.DICT_6X6_250):
        """
        aruco_dict_id: можно сменить на DICT_5X5_100 или DICT_4X4_50,
                       если 6x6 плохо читается.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.base_marker_size = base_marker_size
        self.base_margin = base_margin
        self.base_grid_cols = base_grid_cols
        self.base_grid_rows = base_grid_rows
        self.grid_scale = 1.0

        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Камера не найдена")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(0.3)

        # --- ИНИЦИАЛИЗАЦИЯ ArUco С ПОВЫШЕННОЙ ЧУВСТВИТЕЛЬНОСТЬЮ ---
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        except AttributeError:
            self.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_id)

        # Настраиваем параметры детектора для лучшего распознавания
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Увеличиваем диапазон адаптивного порога (лучше для разных условий освещения)
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        # Разрешаем более мелкие маркеры
        self.aruco_params.minMarkerPerimeterRate = 0.02
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        # Точность аппроксимации
        self.aruco_params.polygonalApproxAccuracyRate = 0.05
        # Дополнительные параметры для устойчивости
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.perspectiveRemovePixelPerCell = 4
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13

        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            # Если ArucoDetector недоступен (старая версия OpenCV), используем старый API
            self.detector = None

        self.H = None
        self.global_marker_positions = {}
        self.calibration_image = None
        self.physical_markers = {}
        self.use_physical_tracking = False

        self.hp_detector = HPBarDetector()
        self.skill_detector = SkillFlashDetector()

        self.roi_file = "hp_roi.txt"
        self.hp_full_file = "hp_full_width.txt"
        self._load_hp_data()
        self._generate_current_grid()

        print("ArUco детектор настроен на повышенную чувствительность.")

    # ----------------------------------------------------------
    # Генерация сетки для калибровки
    # ----------------------------------------------------------
    def _generate_current_grid(self):
        grid_cols = int(self.base_grid_cols * self.grid_scale)
        grid_rows = int(self.base_grid_rows * self.grid_scale)
        marker_size = int(self.base_marker_size / self.grid_scale)
        margin = self.base_margin

        img = np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 255
        if grid_cols > 1 and grid_rows > 1:
            step_x = (self.screen_width - 2 * margin) / (grid_cols - 1)
            step_y = (self.screen_height - 2 * margin) / (grid_rows - 1)
        else:
            step_x = step_y = 0

        self.global_marker_positions.clear()
        for row in range(grid_rows):
            for col in range(grid_cols):
                center_x = margin + col * step_x
                center_y = margin + row * step_y
                top_left_x = int(center_x - marker_size // 2)
                top_left_y = int(center_y - marker_size // 2)
                marker_id = row * grid_cols + col
                self.global_marker_positions[marker_id] = (center_x, center_y)

                try:
                    marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, marker_size)
                except AttributeError:
                    marker_img = cv2.aruco.drawMarker(self.aruco_dict, marker_id, marker_size)
                if len(marker_img.shape) == 2:
                    marker_img = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

                if (0 <= top_left_x and 0 <= top_left_y and
                    top_left_x + marker_size <= self.screen_width and
                    top_left_y + marker_size <= self.screen_height):
                    img[top_left_y:top_left_y+marker_size,
                        top_left_x:top_left_x+marker_size] = marker_img

        self.calibration_image = img
        cv2.imwrite("arucoboard.png", img)
        print(f"Сетка: {len(self.global_marker_positions)} маркеров")

    # ----------------------------------------------------------
    # Загрузка/сохранение HP данных
    # ----------------------------------------------------------
    def _load_hp_data(self):
        if os.path.exists(self.roi_file):
            try:
                with open(self.roi_file, 'r') as f:
                    data = f.read().strip().split()
                    if len(data) == 4:
                        self.hp_detector.set_roi(tuple(map(int, data)))
                        print(f"Загружен HP ROI: {self.hp_detector.roi}")
            except: pass
        if os.path.exists(self.hp_full_file):
            try:
                with open(self.hp_full_file, 'r') as f:
                    w = int(f.read().strip())
                    self.hp_detector.full_width = w
                    print(f"Загружена эталонная ширина HP: {w}")
            except: pass

    def _save_hp_roi(self):
        if self.hp_detector.roi:
            with open(self.roi_file, 'w') as f:
                f.write(f"{self.hp_detector.roi[0]} {self.hp_detector.roi[1]} {self.hp_detector.roi[2]} {self.hp_detector.roi[3]}")

    def _save_hp_full_width(self):
        if self.hp_detector.full_width:
            with open(self.hp_full_file, 'w') as f:
                f.write(str(self.hp_detector.full_width))

    # ----------------------------------------------------------
    # Подавление бликов (CLAHE + гамма)
    # ----------------------------------------------------------
    def reduce_glare(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        gamma = 1.2
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, lut)
        return result

    # ----------------------------------------------------------
    # Фокусировка
    # ----------------------------------------------------------
    def _focus_measure(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def focus_calibration(self):
        print("\n=== НАСТРОЙКА ФОКУСА ===")
        cv2.namedWindow("Focus", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Focus", 800, 600)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            focus_val = self._focus_measure(gray)
            h, w = frame.shape[:2]
            overlay = frame.copy()
            text_str = f"{focus_val:.0f}"
            font_scale = 3
            thickness = 3
            (tw, th), baseline = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (w - tw) // 2
            text_y = (h + th) // 2
            cv2.rectangle(overlay, (text_x-10, text_y-th-10), (text_x+tw+10, text_y+baseline+10), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, text_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
            cv2.putText(frame, "Focus: turn ring, then ENTER, ESC skip", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.imshow("Focus", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == 13:
                break
            if key == 27:
                print("Фокус пропущен")
                break
        cv2.destroyWindow("Focus")
        print("Настройка фокуса завершена.")

    # ----------------------------------------------------------
    # КАЛИБРОВКА (исправленная)
    # ----------------------------------------------------------
    def calibrate(self, timeout=30):
        print("\n=== КАЛИБРОВКА ПО ArUco СЕТКЕ ===")
        print("Перетащите окно 'Calib' на экран, который видит камера, затем нажмите ENTER.")
        cv2.namedWindow("Calib", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calib", 1280, 720)
        cv2.imshow("Calib", self.calibration_image)
        cv2.waitKey(0)  # ждём нажатия клавиши
        cv2.destroyWindow("Calib")

        # Теперь начинаем захват кадров и поиск маркеров
        cv2.namedWindow("Calib", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calib", 1280, 720)
        cv2.imshow("Calib", self.calibration_image)

        start = time.time()
        best_frame = None
        best_focus = -1
        best_corners = None
        best_ids = None
        found = False

        while time.time() - start < timeout:
            ret, frame = self.cap.read()
            if not ret:
                print("Ошибка захвата кадра")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f = self._focus_measure(gray)

            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None and len(ids) >= 8:
                print(f"Найдено маркеров: {len(ids)}, резкость: {f:.1f}")
                if f > best_focus:
                    best_focus = f
                    best_frame = frame.copy()
                    best_corners = corners
                    best_ids = ids
                    found = True
                if f > 500:
                    break

            cv2.imshow("Calib", self.calibration_image)
            if cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyWindow("Calib")
                print("Калибровка прервана.")
                return False

        cv2.destroyWindow("Calib")

        if not found or best_frame is None:
            print("Не удалось найти достаточно маркеров. Попробуйте изменить плотность (+/-) и повторить.")
            return False

        print(f"Выбран лучший кадр: маркеров {len(best_ids)}, резкость {best_focus:.1f}")
        cv2.imwrite("calib_debug.png", best_frame)

        src_pts = []
        dst_pts = []
        for i, mid in enumerate(best_ids.flatten()):
            if mid not in self.global_marker_positions:
                continue
            c = best_corners[i][0]
            cx, cy = np.mean(c[:,0]), np.mean(c[:,1])
            src_pts.append([cx, cy])
            rx, ry = self.global_marker_positions[mid]
            dst_pts.append([rx, ry])

        if len(src_pts) < 4:
            print("Недостаточно соответствий для гомографии.")
            return False

        H, _ = cv2.findHomography(np.array(src_pts, dtype=np.float32),
                                  np.array(dst_pts, dtype=np.float32),
                                  cv2.RANSAC, 5.0)
        if H is None:
            print("Не удалось вычислить гомографию.")
            return False

        self.H = H
        print("Гомография успешно вычислена!")
        return True

    # ----------------------------------------------------------
    # ЗАПОМИНАНИЕ ФИЗИЧЕСКИХ МАРКЕРОВ (требует минимум 4)
    # ----------------------------------------------------------
    def remember_physical_markers(self):
        print("\n=== ЗАПОМИНАНИЕ ФИЗИЧЕСКИХ МАРКЕРОВ ===")
        print("Окно камеры покажет, видит ли она маркеры.")
        print("Нажмите 'm', чтобы запомнить текущие маркеры (нужно минимум 4), или 's', чтобы пропустить.")
        print("Нажмите 'q' для выхода из программы.")

        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 800, 600)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, f"Found {len(ids)} markers", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for i, mid in enumerate(ids.flatten()):
                    c = corners[i][0]
                    cx, cy = int(np.mean(c[:,0])), int(np.mean(c[:,1]))
                    cv2.putText(frame, f"ID:{mid}", (cx-20, cy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "No markers found", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Camera", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('m'):
                if ids is not None and len(ids) >= 4:
                    # Запоминаем маркеры
                    world_pts = {}
                    for i, mid in enumerate(ids.flatten()):
                        c = corners[i][0]
                        cx, cy = np.mean(c[:,0]), np.mean(c[:,1])
                        pts = np.array([[[cx, cy]]], dtype=np.float32)
                        world_pt = cv2.perspectiveTransform(pts, self.H).flatten()
                        world_pts[mid] = (int(world_pt[0]), int(world_pt[1]))
                    self.physical_markers = world_pts
                    self.use_physical_tracking = True
                    print(f"Запомнено {len(self.physical_markers)} маркеров: {self.physical_markers}")
                    break
                else:
                    print(f"Недостаточно маркеров для запоминания (найдено {len(ids) if ids is not None else 0}, нужно минимум 4).")
            elif key == ord('s'):
                print("Пропускаем запоминание физических маркеров.")
                break
            elif key == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyWindow("Camera")

    # ----------------------------------------------------------
    # Выпрямление с динамической коррекцией по физическим маркерам
    # ----------------------------------------------------------
    def rectify_frame_dynamic(self, frame):
        if self.H is None:
            return frame
        # Используем физические маркеры только если их >= 4
        if self.use_physical_tracking and len(self.physical_markers) >= 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is not None:
                src_pts = []
                dst_pts = []
                for i, mid in enumerate(ids.flatten()):
                    if mid in self.physical_markers:
                        c = corners[i][0]
                        cx, cy = np.mean(c[:,0]), np.mean(c[:,1])
                        src_pts.append([cx, cy])
                        dst_pts.append(self.physical_markers[mid])
                if len(src_pts) >= 4:
                    H_new, _ = cv2.findHomography(np.array(src_pts, dtype=np.float32),
                                                  np.array(dst_pts, dtype=np.float32),
                                                  cv2.RANSAC, 5.0)
                    if H_new is not None:
                        self.H = H_new
        warped = cv2.warpPerspective(frame, self.H, (self.screen_width, self.screen_height))
        warped = self.reduce_glare(warped)
        return warped

    # ----------------------------------------------------------
    # Интерфейс для выбора ROI
    # ----------------------------------------------------------
    def select_roi(self, frame, title="Select ROI"):
        roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(title)
        if roi[2] > 0 and roi[3] > 0:
            return tuple(map(int, roi))
        return None

    # ----------------------------------------------------------
    # Настройка HSV через трекбары
    # ----------------------------------------------------------
    def setup_hsv(self, frame):
        win = "HSV Tuning"
        cv2.namedWindow(win)
        cv2.createTrackbar('H_low', win, self.hp_detector.hsv_lower[0], 180, lambda x: None)
        cv2.createTrackbar('S_low', win, self.hp_detector.hsv_lower[1], 255, lambda x: None)
        cv2.createTrackbar('V_low', win, self.hp_detector.hsv_lower[2], 255, lambda x: None)
        cv2.createTrackbar('H_high', win, self.hp_detector.hsv_upper[0], 180, lambda x: None)
        cv2.createTrackbar('S_high', win, self.hp_detector.hsv_upper[1], 255, lambda x: None)
        cv2.createTrackbar('V_high', win, self.hp_detector.hsv_upper[2], 255, lambda x: None)

        print("Настройте диапазоны HSV для HP-бара. Нажмите ESC для выхода.")
        while True:
            h_low = cv2.getTrackbarPos('H_low', win)
            s_low = cv2.getTrackbarPos('S_low', win)
            v_low = cv2.getTrackbarPos('V_low', win)
            h_high = cv2.getTrackbarPos('H_high', win)
            s_high = cv2.getTrackbarPos('S_high', win)
            v_high = cv2.getTrackbarPos('V_high', win)
            lower = np.array([h_low, s_low, v_low])
            upper = np.array([h_high, s_high, v_high])
            self.hp_detector.set_hsv_range(lower, upper)
            ret, cur = self.cap.read()
            if ret:
                warped = self.rectify_frame_dynamic(cur)
                mask = self.hp_detector.get_mask(warped)
                if mask is not None:
                    cv2.imshow("Mask", mask)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        cv2.destroyWindow(win)
        print("Настройка HSV завершена.")

    # ----------------------------------------------------------
    # Основной цикл
    # ----------------------------------------------------------
    def run(self):
        self.focus_calibration()
        print("\n=== НАЧАЛЬНАЯ КАЛИБРОВКА ===")
        print("Перетащите окно 'Calib' на экран, который видит камера.")
        print("Используйте +/- для изменения плотности маркеров, затем 'c' для калибровки.")
        print("Нажмите 'q' для выхода.")

        cv2.namedWindow("Calib", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calib", 1280, 720)

        calibrated = False
        while not calibrated:
            img_display = self.calibration_image.copy()
            cv2.putText(img_display, "Press 'c' to calibrate, '+'/'-' density, 'q' quit",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Calib", img_display)

            key = cv2.waitKey(100) & 0xFF
            if key == ord('c'):
                cv2.destroyWindow("Calib")
                if self.calibrate():
                    calibrated = True
                else:
                    print("Калибровка не удалась.")
                    cv2.namedWindow("Calib", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Calib", 1280, 720)
            elif key == ord('+'):
                self.grid_scale = min(2.0, self.grid_scale + 0.1)
                self._generate_current_grid()
            elif key == ord('-'):
                self.grid_scale = max(0.5, self.grid_scale - 0.1)
                self._generate_current_grid()
            elif key == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

        # Запоминание физических маркеров (теперь с видео)
        self.remember_physical_markers()

        print("\n=== ОСНОВНОЙ РЕЖИМ ===")
        print("Управление:")
        print("  'h' – выделить ROI HP-бара (без калибровки 100%)")
        print("  'k' – калибровка HP при 100% (запомнить эталонную ширину)")
        print("  'u' – выделить ROI для умения")
        print("  't' – настройка HSV для HP (трекбары)")
        print("  'm' – перезапомнить физические маркеры")
        print("  'c' – перекалибровка по сетке")
        print("  'f' – фокус")
        print("  's' – скриншот")
        print("  'q' – выход")

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Rectified", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original", 800, 600)
        cv2.resizeWindow("Rectified", 800, 600)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            warped = self.rectify_frame_dynamic(frame)

            hp = self.hp_detector.detect(warped)
            cv2.putText(warped, f"HP: {hp}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.hp_detector.roi:
                x, y, w, h = self.hp_detector.roi
                cv2.rectangle(warped, (x, y), (x+w, y+h), (0, 255, 255), 2)

            if self.skill_detector.roi is not None:
                if self.skill_detector.update(warped):
                    cv2.putText(warped, "SKILL ACTIVATED!", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                x, y, w, h = self.skill_detector.roi
                cv2.rectangle(warped, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Original", frame)
            cv2.imshow("Rectified", warped)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                self.calibrate()
            if key == ord('f'):
                self.focus_calibration()
            if key == ord('h'):
                roi = self.select_roi(warped, "Select HP Bar")
                if roi:
                    self.hp_detector.set_roi(roi)
                    self._save_hp_roi()
            if key == ord('k'):
                roi = self.select_roi(warped, "Calibrate HP 100%")
                if roi:
                    self.hp_detector.set_roi(roi)
                    self.hp_detector.full_width = roi[2]
                    self._save_hp_roi()
                    self._save_hp_full_width()
            if key == ord('u'):
                roi = self.select_roi(warped, "Select Skill Icon")
                if roi:
                    self.skill_detector.set_roi(roi)
            if key == ord('t'):
                self.setup_hsv(warped)
            if key == ord('m'):
                self.remember_physical_markers()
            if key == ord('s'):
                cv2.imwrite("screenshot.png", warped)
                print("Скриншот сохранён")

        self.cap.release()
        cv2.destroyAllWindows()
        print("Программа завершена.")

# ------------------------------------------------------------
# Запуск
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Для улучшения распознавания можно попробовать DICT_5X5_100 или DICT_4X4_50
        # если маркеры всё ещё плохо читаются.
        app = ArucoRectifier(camera_id=0,
                             screen_width=1920,
                             screen_height=1080,
                             base_marker_size=70,
                             base_margin=80,
                             base_grid_cols=14,
                             base_grid_rows=8,
                             aruco_dict_id=cv2.aruco.DICT_6X6_250)  # можно заменить на DICT_5X5_100
        app.run()
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)