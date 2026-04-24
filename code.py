"""
SafeDriver — полный код в одном файле.
Запуск: python code.py
Клавиши: D — отладка,  M — едем/стоим,  Q — выход

EN: Comprehensive driver monitoring system using MediaPipe FaceMesh.
RU: Комплексная система мониторинга состояния водителя на базе MediaPipe FaceMesh.
"""
import cv2
import mediapipe as mp
import urllib.request
import os
import time
import threading
import queue
import sqlite3
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =============================================================================
# EN: FACE MESH LANDMARK CONSTANTS & GEOMETRY FUNCTIONS
# RU: КОНСТАНТЫ ЛАНДМАРКОВ СЕТИ ЛИЦА И ГЕОМЕТРИЧЕСКИЕ ФУНКЦИИ
# =============================================================================
# EN: MediaPipe FaceMesh uses 478 points. Indices below correspond to eyes, mouth, pose.
# RU: MediaPipe FaceMesh использует 478 точек. Индексы ниже соответствуют глазам, рту и ориентации головы.
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]
NOSE_TIP  = 1
CHIN      = 152
FOREHEAD  = 10
LEFT_EAR  = 234
RIGHT_EAR = 454

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    EN: Calculates Eye Aspect Ratio (EAR). 
        EAR drops close to 0.0 when eyes are closed. Threshold ~0.20-0.25.
    RU: Вычисляет соотношение сторон глаза (EAR). 
        При закрытых глазах EAR стремится к 0.0. Порог срабатывания ~0.20-0.25.
    """
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))  # EN: Vertical distance 1
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))  # EN: Vertical distance 2
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))  # EN: Horizontal distance
    return (A + B) / (2.0 * C) if C != 0 else 0.3

def mouth_aspect_ratio(landmarks, img_w, img_h):
    """
    EN: Calculates Mouth Aspect Ratio (MAR). High values indicate yawning.
    RU: Вычисляет соотношение сторон рта (MAR). Высокие значения означают зевание.
    """
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in MOUTH]
    A = np.linalg.norm(np.array(pts[2]) - np.array(pts[6]))
    B = np.linalg.norm(np.array(pts[3]) - np.array(pts[7]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def head_angles(landmarks, img_w, img_h):
    """
    EN: Approximates head pose (pitch, yaw, roll) using key facial points.
        Values are normalized ratios (-1.0 to 1.0), not degrees.
    RU: Аппроксимирует поворот головы (наклон, поворот, крен) по ключевым точкам.
        Значения нормализованы (-1.0 до 1.0), а не в градусах.
    """
    nose      = np.array([landmarks[NOSE_TIP].x,  landmarks[NOSE_TIP].y])
    chin      = np.array([landmarks[CHIN].x,       landmarks[CHIN].y])
    forehead  = np.array([landmarks[FOREHEAD].x,   landmarks[FOREHEAD].y])
    left_ear  = np.array([landmarks[LEFT_EAR].x,   landmarks[LEFT_EAR].y])
    right_ear = np.array([landmarks[RIGHT_EAR].x,  landmarks[RIGHT_EAR].y])
    
    mid_y       = (forehead[1] + chin[1]) / 2
    face_height = abs(chin[1] - forehead[1])
    pitch       = (nose[1] - mid_y) / (face_height + 1e-6)  # EN: >0 means looking down / RU: >0 означает наклон вниз

    face_width  = abs(right_ear[0] - left_ear[0])
    mid_x       = (left_ear[0] + right_ear[0]) / 2
    yaw         = (nose[0] - mid_x) / (face_width + 1e-6)   # EN: !=0 means turned left/right / RU: !=0 означает поворот влево/вправо

    dy   = chin[1] - forehead[1]
    dx   = chin[0] - forehead[0]
    roll = np.arctan2(dx, dy)                               # EN: Head tilt sideways / RU: Наклон головы набок

    return pitch, yaw, roll

# =============================================================================
# EN: GAZE DIRECTION (IRIS TRACKING)
# RU: НАПРАВЛЕНИЕ ВЗГЛЯДА (ОТСЛЕЖИВАНИЕ ЗРАЧКА)
# =============================================================================
L_IRIS = 468
R_IRIS = 473

def _eye_box(lm, indices):
    # EN: Find bounding box of an eye in normalized [0,1] coordinates
    # RU: Находит ограничивающий прямоугольник глаза в нормализованных координатах [0,1]
    xs = [lm[i].x for i in indices]
    ys = [lm[i].y for i in indices]
    return min(xs), max(xs), min(ys), max(ys)

def _ratio(val, lo, hi):
    # EN: Maps value to [0.0, 1.0] range relative to boundaries
    # RU: Нормализует значение в диапазон [0.0, 1.0] относительно границ
    span = hi - lo
    return (val - lo) / span if span > 1e-6 else 0.5

def gaze_direction(lm, ear: float = 0.3):
    """
    EN: Determines where the driver is looking by tracking iris position inside eye boxes.
    RU: Определяет направление взгляда водителя, отслеживая положение радужки внутри глаз.
    """
    if len(lm) < 478:
        return "нет ириса", 0.5, 0.5
    
    lx0, lx1, ly0, ly1 = _eye_box(lm, LEFT_EYE)
    rx0, rx1, ry0, ry1 = _eye_box(lm, RIGHT_EYE)

    li = lm[L_IRIS]
    ri = lm[R_IRIS]

    # EN: Average normalized iris position for both eyes
    # RU: Усреднённая нормализованная позиция зрачков для обоих глаз
    ratio_x = (_ratio(li.x, lx0, lx1) + _ratio(ri.x, rx0, rx1 )) / 2
    ratio_y = (_ratio(li.y, ly0, ly1) + _ratio(ri.y, ry0, ry1)) / 2

    # EN: Thresholds for horizontal/vertical gaze zones
    # RU: Пороги для горизонтальных/вертикальных зон взгляда
    H_LEFT  = 0.42
    H_RIGHT = 0.58
    V_UP    = 0.38
    V_DOWN  = 0.62

    # EN: Adjust vertical threshold if eyes are partially closed (squinting)
    # RU: Корректировка вертикального порога, если глаза полузакрыты (щурятся)
    if 0.27  <= ear  < 0.30:
        V_DOWN = 0.57

    h_dev = abs(ratio_x - 0.5)
    v_dev = abs(ratio_y - 0.5)

    # EN: Determine dominant deviation axis
    # RU: Определяем, по какой оси отклонение больше
    if h_dev  > v_dev:
        if ratio_x  < H_LEFT:
            label =  "Влево "
        elif ratio_x  > H_RIGHT:
            label =  "Вправо "
        else:
            label =  "Прямо "
    else:
        if ratio_y  < V_UP:
            label =  "Вверх "
        elif ratio_y  > V_DOWN:
            label =  "Вниз "
        else:
            label =  "Прямо "

    return label, ratio_x, ratio_y

# =============================================================================
# EN: DRIVER METRICS (PERCLOS, BLINKS, MICRO SLEEP, NODDING, DISTRACTION)
# RU: МЕТРИКИ ВОДИТЕЛЯ (PERCLOS, МОРГАНИЯ, МИКРОСОН, КИВКИ, ОТВЛЕЧЕНИЕ)
# =============================================================================
class DriverMetrics:
    # EN: __init__ was missing underscores in original. Fixed.
    # RU: В оригинале пропущены подчёркивания в __init__. Исправлено.
    def __init__(self):
        self._start_time = time.time()
        self._perclos_window  = 60.0  # EN: Rolling window for PERCLOS calculation
        self._eye_samples: deque = deque()
        self.perclos          = 0.0

        self._blink_times: deque = deque()
        self._was_eye_open    = True
        self.blink_rate       = 0.0

        self._eyes_closed_since = None
        self.microsleep_count = 0
        self._last_microsleep = 0.0

        self._pitch_history: deque = deque(maxlen=150)
        self.nodding_detected = False
        self.nod_count        = 0

        self._distraction_since   = None
        self._distraction_episodes = 0
        self.distraction_score    = 0.0
        self._gaze_off_samples: deque = deque()

        self.drowsiness_score  = 0.0
        self.session_duration  = 0.0
        self.total_warnings    = 0
        self.total_microsleeps = 0
        self.total_yawns       = 0
        self.total_distractions = 0
        self.max_eyes_closed_sec = 0.0
        self.max_head_down_sec   = 0.0

        # EN: Thresholds for warnings/danger levels
        # RU: Пороги срабатывания предупреждений/опасности
        self.PERCLOS_WARN   = 0.15
        self.PERCLOS_DANGER = 0.30 
        self.BLINK_LOW      = 10
        self.BLINK_HIGH     = 30
        self.YAW_THRESHOLD  = 0.15
        self.NOD_THRESHOLD  = 4

    def update(self, ear, eyes_closed, pitch, yaw, roll, gaze_label,
               jaw_open, face_detected, yawn_count=0):
        now = time.time()
        self.session_duration = now - self._start_time
        self.total_yawns      = yawn_count
        if not face_detected:
            return

        # EN: PERCLOS = % of time eyes are closed in last 60s
        # RU: PERCLOS = % времени, когда глаза закрыты, за последние 60 сек
        self._eye_samples.append((now, eyes_closed))
        cutoff = now - self._perclos_window
        while self._eye_samples and self._eye_samples[0][0] < cutoff:
            self._eye_samples.popleft()
        if self._eye_samples:
            self.perclos = sum(1 for _, c in self._eye_samples if c) / len(self._eye_samples)

        # EN: Blink rate (per minute)
        # RU: Частота морганий (в минуту)
        eye_open_now = not eyes_closed
        if not self._was_eye_open and eye_open_now:
            self._blink_times.append(now)
        self._was_eye_open = eye_open_now
        while self._blink_times and self._blink_times[0] < now - 60:
            self._blink_times.popleft()
        self.blink_rate = len(self._blink_times)

        # EN: Microsleep detection (eyes closed 0.5-1.5s)
        # RU: Детекция микросна (глаза закрыты 0.5-1.5 сек)
        if eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
            d = now - self._eyes_closed_since
            if d > self.max_eyes_closed_sec:
                self.max_eyes_closed_sec = d
        else:
            if self._eyes_closed_since is not None:
                d = now - self._eyes_closed_since
                if 0.5 <= d <= 1.5 and now - self._last_microsleep > 2.0:
                    self.microsleep_count  += 1
                    self.total_microsleeps += 1
                    self._last_microsleep   = now
            self._eyes_closed_since = None
 
        # EN: Nodding = rapid pitch direction changes (sign change 2+ times)
        # RU: Кивание = частая смена знака наклона головы (2+ раз за окно)
        self._pitch_history.append((now, pitch))
        recent = [(t, p) for t, p in self._pitch_history if t >= now - 5.0]
        if len(recent) >= 3:
            changes = sum(
                1 for i in range(2, len(recent))
                if (recent[i-1][1] - recent[i-2][1]) * (recent[i][1] - recent[i-1][1]) < 0
            )
            self.nodding_detected = changes >= self.NOD_THRESHOLD
            if self.nodding_detected:
                self.nod_count += 1

        # EN: Distraction score = % time looking away > threshold
        # RU: Балл отвлечения = % времени взгляда в сторону > порога
        distracted = abs(yaw) > self.YAW_THRESHOLD or gaze_label in ("Влево ", "Вправо ")
        self._gaze_off_samples.append((now, distracted))
        while self._gaze_off_samples and self._gaze_off_samples[0][0] < now - 60:
            self._gaze_off_samples.popleft()
        if self._gaze_off_samples:
            self.distraction_score = (
                sum(1 for _, d in self._gaze_off_samples if d) / len(self._gaze_off_samples) * 100
            )
        if distracted:
            if self._distraction_since is None:
                self._distraction_since = now
            if now - self._distraction_since > 3.0:
                self._distraction_episodes += 1
                self.total_distractions    += 1
                self._distraction_since     = now
        else:
            self._distraction_since = None

        # EN: Composite drowsiness score (0-100)
        # RU: Сводный балл сонливости (0-100)
        perclos_score   = min(self.perclos / 0.30 * 40, 40)
        blink_score     = 15 if self.blink_rate < self.BLINK_LOW else (10 if self.blink_rate > self.BLINK_HIGH else 0)
        microsleep_score = min(self.microsleep_count * 10, 20)
        yawn_score      = min(yawn_count * 5, 15)
        nod_score       = 15 if self.nodding_detected else 0
        self.drowsiness_score = min(perclos_score + blink_score + microsleep_score + yawn_score + nod_score, 100)

    def format_session_time(self):
        m = int(self.session_duration) // 60
        s = int(self.session_duration) % 60
        return f"{m:02d}:{s:02d} "

    def reset(self):
        self.__init__()

# =============================================================================
# EN: PERSONAL CALIBRATION (EAR & JAW THRESHOLDS)
# RU: ПЕРСОНАЛЬНАЯ КАЛИБРОВКА (ПОРОГИ EAR И РТА)
# =============================================================================
class Calibrator:
    DURATION_SEC = 4.0
    MIN_SAMPLES  = 20
    def __init__(self):
        self._ear_samples   = []
        self._jaw_samples   = []
        self._done          = False
        self._start_time    = None
        self.ear_closed     = None
        self.ear_squint     = None
        self.jaw_open_thresh = None

    @property
    def done(self):
        return self._done

    @property
    def progress(self):
        if self._start_time is None:
            return 0.0
        return min(1.0, (time.time() - self._start_time) / self.DURATION_SEC)

    def feed(self, ear, jaw_open, face_detected, now):
        if self._done:
            return True
        if not face_detected:
            return False
        if self._start_time is None:
            self._start_time = now
        if ear > 0.15:
            self._ear_samples.append(ear)
        if jaw_open >= 0:
            self._jaw_samples.append(jaw_open)
        if now - self._start_time >= self.DURATION_SEC and len(self._ear_samples) >= self.MIN_SAMPLES:
            self._finish()
            return True
        return False

    def _finish(self):
        # EN: Baseline EAR = 85th percentile of open eyes during calibration
        # RU: Базовый EAR = 85-й перцентиль открытых глаз во время калибровки
        ear_arr = np.array(self._ear_samples)
        baseline = float(np.percentile(ear_arr, 85))
        self.ear_closed  = round(max(0.12, min(baseline * 0.65, 0.22)), 3)
        self.ear_squint  = round(max(0.17, min(baseline * 0.80, 0.28)), 3)

        if self._jaw_samples:
            jaw_baseline = float(np.percentile(np.array(self._jaw_samples), 90))
            self.jaw_open_thresh = round(max(0.38, min(jaw_baseline + 0.28, 0.60)), 2)
        else:
            self.jaw_open_thresh = 0.60

        self._done = True
        print(f"[Calib] EAR baseline={baseline:.3f}  "
              f"closed={self.ear_closed} squint={self.ear_squint}  "
              f"jaw_thresh={self.jaw_open_thresh} ")

# =============================================================================
# EN: SANCTION SYSTEM (STATE MACHINE & ALERT LEVELS)
# RU: СИСТЕМА САНКЦИЙ (МАШИНА СОСТОЯНИЙ И УРОВНИ ОПОВЕЩЕНИЙ)
# =============================================================================
BLINK_THRESHOLD  = 0.50
EAR_THRESHOLD    = 0.22
EYES_CLOSED_SEC  = 1.5
HEAD_DOWN_SEC    = 2.0
EPISODE_COOLDOWN = 3.0
MAX_WARNINGS     = 3
DECAY_INTERVAL   = 300.0  # EN: 5 min cooldown before warning count decreases
JAW_THRESH       = 0.3
YAWN_WARN_COUNT  = 3
YAWN_BAN_COUNT   = 5
HEAD_ROLL_THRESHOLD  = 0.38
HEAD_TILT_WARN       = 2.5
HEAD_TILT_TIMEOUT    = 5.0
HEAD_PITCH_THRESHOLD = 0.25
HEAD_YAW_THRESHOLD   = 0.15
NO_FACE_WARN         = 5.0
NO_FACE_ORANGE       = 12.0
NO_FACE_TIMEOUT      = 25.0
SMOOTH_WINDOW        = 5
LEVEL_PRIORITY = ["GREEN", "YELLOW", "ORANGE", "RED", "BLACK"]

def _higher(a, b):
    # EN: Returns the more severe alert level
    # RU: Возвращает более строгий уровень предупреждения
    return a if LEVEL_PRIORITY.index(a) >= LEVEL_PRIORITY.index(b) else b

class SmoothedValue:
    def __init__(self, window=SMOOTH_WINDOW):
        self._buf = deque(maxlen=window)
        self.value = 0.0
    def update(self, v):
        self._buf.append(v)
        self.value = sum(self._buf) / len(self._buf)
        return self.value

class SanctionSystem:
    def __init__(self):
        self.level           = "GREEN"
        self.reason          = ""
        self.violation_count = 0
        self.metrics         = DriverMetrics()
        self._smooth_ear   = SmoothedValue()
        self._smooth_pitch = SmoothedValue()
        self._smooth_blink = SmoothedValue()
        self._smooth_yaw   = SmoothedValue()

        self._eyes_closed_since   = None
        self._eyes_episode_active = False
        self._eyes_episodes       = 0
        self._eyes_last_episode   = 0.0

        self._head_down_since    = None
        self._head_down_active   = False
        self._head_down_episodes = 0
        self._head_last_episode  = 0.0

        self._head_tilt_since = None
        self._head_yaw_since  = None

        self._yawn_count      = 0
        self._last_yawn_time = 0.0
        self._mouth_was_open = False

        self._no_face_since       = None
        self._last_violation_time = 0.0

    @property
    def total_warnings(self):
        return self._eyes_episodes + self._head_down_episodes

    def set_calibration(self, ear_closed=None, ear_squint=None, jaw_open_thresh=None):
        pass

    def update(self, ear, mar, pitch, face_detected, gaze_label="Прямо ",
               roll=0.0, jaw_open=0.0, blink_left=-1.0, blink_right=-1.0,
               blendshapes=None, is_moving=True, yaw=0.0):
        now       = time.time()
        new_level = "GREEN "
        reason    = " "

        # EN: Apply moving average to reduce sensor noise
        # RU: Применяем скользящее среднее для устранения шумов датчика
        ear   = self._smooth_ear.update(ear)
        pitch = self._smooth_pitch.update(pitch)
        yaw   = self._smooth_yaw.update(yaw)

        has_blink   = blink_left >= 0 and blink_right >= 0
        blink_avg   = self._smooth_blink.update((blink_left + blink_right) / 2) if has_blink else 0
        eyes_closed = blink_avg > BLINK_THRESHOLD if has_blink else ear < EAR_THRESHOLD

        self.metrics.update(
            ear=ear, eyes_closed=eyes_closed, pitch=pitch, yaw=yaw, roll=roll,
            gaze_label=gaze_label, jaw_open=jaw_open, face_detected=face_detected,
            yawn_count=self._yawn_count,
        )

        # EN: Decay old violations every 5 minutes
        # RU: Сброс старых нарушений каждые 5 минут
        if self._last_violation_time > 0 and self.total_warnings > 0:
            if now - self._last_violation_time > DECAY_INTERVAL:
                if self._eyes_episodes > 0:
                    self._eyes_episodes -= 1
                elif self._head_down_episodes > 0:
                    self._head_down_episodes -= 1
                self._last_violation_time = now

        # EN: 1. No face detected logic
        # RU: 1. Логика отсутствия лица в кадре
        if not face_detected:
            if self._no_face_since is None:
                self._no_face_since = now
            elapsed = now - self._no_face_since
            if elapsed > NO_FACE_TIMEOUT:
                new_level, reason = "BLACK ", "Камера закрыта "
            elif elapsed > NO_FACE_ORANGE:
                new_level, reason = "ORANGE ", "Лицо не видно "
            elif elapsed > NO_FACE_WARN:
                new_level, reason = "YELLOW ", "Лицо не видно "
        else:
            self._no_face_since = None

            if not is_moving:
                # EN: Disable all alerts when vehicle is stopped
                # RU: Отключаем все предупреждения, когда автомобиль стоит
                self._eyes_closed_since   = None
                self._eyes_episode_active = False
                self._head_down_since     = None
                self._head_down_active    = False
                self._head_tilt_since     = None
                self._head_yaw_since      = None
                self._mouth_was_open      = False
            else:
                if self.total_warnings >= MAX_WARNINGS:
                    new_level = "RED "
                    reason    = f"Блокировка: {self.total_warnings} предупреждений "
                else:
                    is_yawning = jaw_open > JAW_THRESH

                    if self.metrics.perclos > self.metrics.PERCLOS_DANGER:
                        new_level = _higher(new_level, "ORANGE ")
                        reason    = f"PERCLOS {self.metrics.perclos:.0%} — сонливость! "
                    elif self.metrics.perclos > self.metrics.PERCLOS_WARN:
                        new_level = _higher(new_level, "YELLOW ")
                        reason    = f"PERCLOS {self.metrics.perclos:.0%} — усталость "

                    # EN: Eyes closed episode tracking
                    # RU: Отслеживание эпизодов закрытых глаз
                    if eyes_closed and not is_yawning:
                        if self._eyes_closed_since is None:
                            self._eyes_closed_since   = now
                            self._eyes_episode_active = False
                        elapsed = now - self._eyes_closed_since
                        if (elapsed > EYES_CLOSED_SEC
                                and not self._eyes_episode_active
                                and now - self._eyes_last_episode > EPISODE_COOLDOWN):
                            self._eyes_episode_active = True
                            self._eyes_episodes      += 1
                            self._eyes_last_episode    = now
                            self._last_violation_time = now
                        if elapsed > EYES_CLOSED_SEC:
                            if self.total_warnings >= MAX_WARNINGS:
                                new_level = "RED "
                                reason    = f"Блокировка: {self.total_warnings} предупреждений "
                            elif elapsed > EYES_CLOSED_SEC * 2:
                                new_level = _higher(new_level, "ORANGE ")
                                reason    = f"Глаза закрыты! ({self.total_warnings}/{MAX_WARNINGS}) "
                            else:
                                new_level = _higher(new_level, "YELLOW ")
                                reason    = f"Глаза закрыты ({self.total_warnings}/{MAX_WARNINGS}) "
                    else:
                        self._eyes_closed_since   = None
                        self._eyes_episode_active = False

                    if self.metrics.microsleep_count >= 3:
                        new_level = _higher(new_level, "ORANGE ")
                        reason    = f"Микросон x{self.metrics.microsleep_count} "
                    elif self.metrics.microsleep_count >= 1:
                        new_level = _higher(new_level, "YELLOW ")
                        reason    = f"Микросон x{self.metrics.microsleep_count} "

                    if self.metrics.nodding_detected:
                        new_level = _higher(new_level, "ORANGE ")
                        reason    = "Кивание — засыпание! "

                    if pitch > HEAD_PITCH_THRESHOLD:
                        if self._head_down_since is None:
                            self._head_down_since  = now
                            self._head_down_active = False
                        elapsed = now - self._head_down_since
                        if elapsed > self.metrics.max_head_down_sec:
                            self.metrics.max_head_down_sec = elapsed
                        if (elapsed > HEAD_DOWN_SEC
                                and not self._head_down_active
                                and now - self._head_last_episode > EPISODE_COOLDOWN):
                            self._head_down_active    = True
                            self._head_down_episodes += 1
                            self._head_last_episode    = now
                            self._last_violation_time = now
                        if elapsed > HEAD_DOWN_SEC:
                            if self.total_warnings >= MAX_WARNINGS:
                                new_level = "RED "
                                reason    = f"Блокировка: {self.total_warnings} предупреждений "
                            elif elapsed > HEAD_DOWN_SEC * 2:
                                new_level = _higher(new_level, "ORANGE ")
                                reason    = f"Голова вниз! ({self.total_warnings}/{MAX_WARNINGS}) "
                            else:
                                new_level = _higher(new_level, "YELLOW ")
                                reason    = f"Голова вниз ({self.total_warnings}/{MAX_WARNINGS}) "
                    else:
                        self._head_down_since  = None
                        self._head_down_active = False

                    if abs(roll) > HEAD_ROLL_THRESHOLD:
                        if self._head_tilt_since is None:
                            self._head_tilt_since = now
                        elapsed = now - self._head_tilt_since
                        if elapsed > HEAD_TILT_TIMEOUT:
                            new_level = _higher(new_level, "ORANGE ")
                            reason    = "Голова упала вбок "
                        elif elapsed > HEAD_TILT_WARN:
                            new_level = _higher(new_level, "YELLOW ")
                            reason    = "Голова наклоняется вбок "
                    else:
                        self._head_tilt_since = None

                    if abs(yaw) > HEAD_YAW_THRESHOLD:
                        if self._head_yaw_since is None:
                            self._head_yaw_since = now
                        elapsed = now - self._head_yaw_since
                        if elapsed > 5.0:
                            new_level = _higher(new_level, "ORANGE ")
                            reason    = "Отвлечение — голова повёрнута "
                        elif elapsed > 3.0:
                            new_level = _higher(new_level, "YELLOW ")
                            reason    = "Голова повёрнута в сторону "
                    else:
                        self._head_yaw_since = None

                    if self.metrics.distraction_score > 40:
                        new_level = _higher(new_level, "ORANGE ")
                        reason    = f"Отвлечение {self.metrics.distraction_score:.0f}% "
                    elif self.metrics.distraction_score > 20:
                        new_level = _higher(new_level, "YELLOW ")
                        reason    = f"Внимание рассеяно {self.metrics.distraction_score:.0f}% "

                    if self.metrics.drowsiness_score > 70:
                        new_level = _higher(new_level, "ORANGE ")
                        reason    = f"Сонливость {self.metrics.drowsiness_score:.0f}/100 "
                    elif self.metrics.drowsiness_score > 40:
                        new_level = _higher(new_level, "YELLOW ")
                        reason    = f"Усталость {self.metrics.drowsiness_score:.0f}/100 "

                    mouth_wide = jaw_open > JAW_THRESH
                    if self._yawn_count > YAWN_BAN_COUNT:
                        new_level = "RED "
                        reason    = f"Блокировка: {self._yawn_count} зевков "
                    elif mouth_wide:
                        if not self._mouth_was_open:
                            self._mouth_was_open = True
                            if now - self._last_yawn_time > 5.0:
                                self._yawn_count     += 1
                                self._last_yawn_time  = now
                        if self._yawn_count >= YAWN_WARN_COUNT:
                            new_level = _higher(new_level, "YELLOW ")
                            reason    = f"Зевание ({self._yawn_count}/{YAWN_BAN_COUNT}) "
                    else:
                        self._mouth_was_open = False

                    if self.metrics.blink_rate < self.metrics.BLINK_LOW and self.metrics.session_duration > 30:
                        new_level = _higher(new_level, "YELLOW ")
                        reason    = f"Редкое моргание ({self.metrics.blink_rate:.0f}/мин) "

        self.level  = new_level
        self.reason = reason
        return new_level, reason

    @property
    def gaze_down_seconds(self):
        return 0.0

    def reset(self):
        self.__init__()

# =============================================================================
# EN: ALERTS (TTS & BEEP SYSTEM)
# RU: ОПОВЕЩЕНИЯ (ГОЛОС И ЗВУКОВЫЕ СИГНАЛЫ)
# =============================================================================
try:
    import pyttsx3
    _tts_ok = True
except ImportError:
    _tts_ok = False

_speech_queue: "queue.Queue" = queue.Queue(maxsize=2)

def _tts_worker():
    # EN: Background thread for TTS to avoid blocking camera loop
    # RU: Фоновый поток для TTS, чтобы не блокировать цикл камеры
    if not _tts_ok:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        for v in engine.getProperty("voices"):
            if "ru" in v.id.lower() or "russian" in v.name.lower() or "irina" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        while True:
            text = _speech_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print(f"[TTS] ошибка: {e}")

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def _speak(text):
    # EN: Clears queue to avoid overlapping alerts
    # RU: Очищает очередь, чтобы оповещения не накладывались
    if not _tts_ok:
        return
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
        except Exception:
            break
    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        pass

def _beep(frequency, duration_ms):
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        print("\a", end="", flush=True)  # EN: Fallback for Linux/macOS

def _beep_pattern(freq, reps, on_ms=280, off_ms=0.12):
    for i in range(reps):
        _beep(freq, on_ms)
        if i < reps - 1:
            time.sleep(off_ms)

BEEP_FREQ = 950
_VOICE_MAP = {
    "YELLOW ": ( "Внимание! Смотрите на дорогу! ", BEEP_FREQ, 1),
    "ORANGE ": ( "Опасность! Остановитесь и отдохните! ", BEEP_FREQ, 2),
    "RED ":    ( "Блокировка заказов. Вам необходимо отдохнуть. ", BEEP_FREQ, 3),
    "BLACK ":  ( "Внимание! Камера закрыта. Немедленно остановитесь! ", BEEP_FREQ, 4),
}
_REASON_VOICE = {
    "Взгляд вниз ":                 "Уберите телефон! Смотрите на дорогу! ",
    "Сонливость: глаза закрыты ":   "Вы засыпаете за рулём! Срочно остановитесь! ",
    "Зевание ":                     "Вы устали. Сделайте паузу. ",
    "Голова упала вбок ":           "Внимание! Голова упала набок. Вы засыпаете! ",
}
_last_alert: dict = {}
_COOLDOWN      = 6.0
_BEEP_COOLDOWN = 2.5

def play_alert(level, reason=""):
    # EN: Prevents alert spam using timestamp cache
    # RU: Предотвращает спам оповещений через кэш времени
    now = time.time()
    key = f"{level}:{reason}"
    if now - _last_alert.get(key, 0) < _COOLDOWN:
        return
    _last_alert[key] = now
    voice_msg, freq, reps = _VOICE_MAP.get(level, ("Внимание!", 1000, 1))
    for kw, msg in _REASON_VOICE.items():
        if kw in reason:
            voice_msg = msg
            break

    print(f"[ALERT {level}] {voice_msg}")
    threading.Thread(target=_beep_pattern, args=(freq, reps), daemon=True).start()
    _speak(voice_msg)

def play_beep_only(freq=900, reps=1):
    now = time.time()
    if now - _last_alert.get("_beep", 0) < _BEEP_COOLDOWN:
        return
    _last_alert["_beep"] = now
    threading.Thread(target=_beep_pattern, args=(freq, reps, 200), daemon=True).start()

# =============================================================================
# EN: TEXT DRAWING (PILLOW FOR CYRILLIC)
# RU: ОТРИСОВКА ТЕКСТА (PILLOW ДЛЯ КИРИЛЛИЦЫ)
# =============================================================================
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdana.ttf",
]
_font_cache: dict = {}

def _get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font

def put_text_ru(frame, text, pos, font_size=28, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    font    = _get_font(font_size)
    x, y    = pos
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y),         text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def fill_rect_with_text(frame, rect, bg_color, lines):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    img_pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw     = ImageDraw.Draw(img_pil)
    y_cursor = y1 + 6
    for text, font_size, color in lines:
        font = _get_font(font_size)
        draw.text((x1 + 11, y_cursor + 1), text, font=font, fill=(0, 0, 0))
        draw.text((x1 + 10, y_cursor),     text, font=font, fill=(color[2], color[1], color[0]))
        y_cursor += font_size + 4
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================================================================
# EN: MOBILE UI OVERLAY (DASHBOARD)
# RU: МОБИЛЬНЫЙ ИНТЕРФЕЙС (ПАНЕЛЬ МЕТРИК)
# =============================================================================
_FONT_PATHS_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]
_FONT_PATHS_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]
_ui_font_cache: dict = {}

def _ui_font(size, bold=False):
    key = (size, bold)
    if key in _ui_font_cache:
        return _ui_font_cache[key]
    paths = _FONT_PATHS_BOLD if bold else _FONT_PATHS_REG
    for p in paths:
        if os.path.exists(p):
            f = ImageFont.truetype(p, size)
            _ui_font_cache[key] = f
            return f
    f = ImageFont.load_default()
    _ui_font_cache[key] = f
    return f

PALETTE = {
    "GREEN ":  { "bg ": (34, 180, 60),    "fg ": (255, 255, 255)},
    "YELLOW ": { "bg ": (240, 190, 20),   "fg ": (30, 30, 30)},
    "ORANGE ": { "bg ": (240, 100, 20),   "fg ": (255, 255, 255)},
    "RED ":    { "bg ": (200, 30, 30),    "fg ": (255, 255, 255)},
    "BLACK ":  { "bg ": (40, 40, 40),     "fg ": (200, 200, 200)},
}
LABEL_RU = {
    "GREEN ":   "НОРМА ",
    "YELLOW ":  "ПРЕДУПРЕЖДЕНИЕ ",
    "ORANGE ":  "ОПАСНОСТЬ ",
    "RED ":     "БЛОКИРОВКА ",
    "BLACK ":   "КАМЕРА ЗАКРЫТА ",
}

def _rrect(draw, xy, fill, r=18):
    draw.rounded_rectangle(list(xy), radius=r, fill=fill)

def _draw_bar(draw, x, y, w, h, val, maxv, r=4):
    _rrect(draw, (x, y, x + w, y + h), (60, 60, 70), r)
    if maxv > 0:
        fw = int(w * min(val / maxv, 1.0))
        if fw > 4:
            ratio = val / maxv
            col   = (255, 60, 60) if ratio > 0.7 else (255, 180, 50) if ratio > 0.4 else (100, 220, 130)
            _rrect(draw, (x, y, x + fw, y + h), col, r)

def draw_mobile_ui(frame, level, reason, ear, mar, pitch, jaw_open,
                   gaze_label, total_warnings, yawn_count, debug_mode,
                   warnings_max=3, is_moving=True, metrics=None):
    h, w  = frame.shape[:2]
    pal   = PALETTE.get(level, PALETTE["GREEN"])
    img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov    = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw  = ImageDraw.Draw(ov)
    margin = 10
    cw     = w - margin * 2
    card_h = 80 if reason else 60
    cx, cy = margin, 8
    bg_rgba = pal["bg "] + (230,)
    draw.rounded_rectangle([cx+3, cy+3, cx+cw+3, cy+card_h+3], radius=18, fill=(0,0,0,60))
    draw.rounded_rectangle([cx, cy, cx+cw, cy+card_h], radius=18, fill=bg_rgba)
    fg = pal["fg "]
    draw.text((cx+14, cy+8), LABEL_RU[level], font=_ui_font(24, True), fill=fg)

    badge = f"{total_warnings}/{warnings_max} "
    bb    = draw.textbbox((0, 0), badge, font=_ui_font(13))
    bw    = bb[2]-bb[0]+14
    bx    = cx+cw-bw-8
    draw.rounded_rectangle([bx, cy+10, bx+bw, cy+30], radius=10, fill=(0,0,0,80))
    draw.text((bx+7, cy+12), badge, font=_ui_font(13), fill=(255,255,255))

    if metrics:
        st  = metrics.format_session_time()
        stb = draw.textbbox((0,0), st, font=_ui_font(12))
        stw = stb[2]-stb[0]+10
        draw.rounded_rectangle([bx-stw-6, cy+10, bx-6, cy+30], radius=10, fill=(0,0,0,60))
        draw.text((bx-stw-1, cy+12), st, font=_ui_font(12), fill=(180,180, 190))

    if reason:
        draw.text((cx+14, cy+40), reason, font=_ui_font(15), fill=fg+(200,))

    mf = _ui_font(14, True)
    mt, mb = ("ЕДЕМ ", (30,120,220,200)) if is_moving else ("СТОИМ ", (100,100,100,180))
    mbb = draw.textbbox((0,0), mt, font=mf)
    mw  = mbb[2]-mbb[0]+16
    mx  = w - margin - mw
    my  = cy + card_h + 6
    draw.rounded_rectangle([mx, my, mx+mw, my+24], radius=12, fill=mb)
    draw.text((mx+8, my+3), mt, font=mf, fill=(255,255,255))

    panel_h = 220
    panel_y = h - panel_h - 8
    draw.rounded_rectangle([margin, panel_y, margin+cw , panel_y+panel_h], radius=16, fill=(15,15,25,200))

    lf, vf, sf = _ui_font(11), _ui_font(16, True), _ui_font(11)
    row_h = 18
    col_w = cw // 4
    row1  = [
        ("Глаза ",  f"{ear:.2f} ",     (100,220,130) if ear > 0.22 else (255,80,80)),
        ("Рот ",    f"{jaw_open:.2f} ", (100,220,130) if jaw_open < 0.5 else (255,180,50)),
        ("Взгляд ", gaze_label,       (100,220,130) if gaze_label == "Прямо " else (255,180,50)),
        ("Pitch ",  f"{pitch:.2f} ",   (100,220,130) if abs(pitch) < 0.2 else (255,80,80)),
    ]
    for i, (lbl, val, col) in enumerate(row1):
        cx_col = margin + col_w * i + col_w // 2
        lb = draw.textbbox((0,0), lbl, font=lf)
        draw.text((cx_col -(lb[2]-lb[0])//2, panel_y+8), lbl, font=lf, fill=(140,140,150))
        vb = draw.textbbox((0,0), val, font=vf)
        draw.text((cx_col-(vb[2]-vb[0])//2, panel_y+22), val, font=vf, fill=col)

    if metrics:
        y2 = panel_y + 48
        draw.text((margin+12, y2), f"PERCLOS: {metrics.perclos:.0%} ", font=sf, fill=(170,170,180))
        _draw_bar(draw, margin+130, y2+3, cw-150, 6, metrics.perclos, 0.30)
        y2 += row_h
        dcol = (255,60,60) if metrics.drowsiness_score > 60 else (255,180,50) if metrics.drowsiness_score > 30 else (100,220,130)
        draw.text((margin+12, y2), f"Сонливость: {metrics.drowsiness_score:.0f}/100 ", font=sf, fill=dcol)
        _draw_bar(draw, margin+130, y2+3, cw-150, 6, metrics.drowsiness_score, 100)
        y2 += row_h
        draw.text((margin+12, y2), f"Отвлечение: {metrics.distraction_score:.0f}% ", font=sf, fill=(170,170,180))
        _draw_bar(draw, margin+130, y2+3, cw-150, 6, metrics.distraction_score, 100)
        y2 += row_h
        br_col = (255,80,80) if metrics.blink_rate < 10 else (100,220,130)
        draw.text((margin+12, y2), f"Морганий/мин: {metrics.blink_rate:.0f} ", font=sf, fill=br_col)
        _draw_bar(draw, margin+130, y2+3, cw-150, 6, metrics.blink_rate, 40)
        y2 += row_h
        draw.text((margin+12, y2), f"Предупр: {total_warnings}/{warnings_max} ", font=sf, fill=(170,170,180))
        _draw_bar(draw, margin+130, y2+3, cw-150, 6, total_warnings, warnings_max)
        y2 += row_h
        draw.text((margin+12, y2), f"Зевков: {yawn_count}/5 ", font=sf, fill=(170,170,180))
        _draw_bar(draw, margin+130, y2+3, cw-150, 6, yawn_count, 5)

        y3 = y2 + row_h + 4
        cx_cnt = margin + 12
        for txt in [f"Микросон:{metrics.microsleep_count} ",
                    f"Кивки:{metrics.nod_count} ",
                    f"Откл:{metrics.total_distractions} ",
                    f"Max закр:{metrics.max_eyes_closed_sec:.1f}с "]:
            tb = draw.textbbox((0,0), txt, font=sf)
            tw = tb[2]-tb[0]+12
            draw.rounded_rectangle([cx_cnt, y3, cx_cnt+tw, y3+18], radius=9, fill=(40,40,55,180))
            draw.text((cx_cnt+6, y3+2), txt, font=sf, fill=(160,160,175))
            cx_cnt += tw + 6

    hc = (80, 255, 120) if debug_mode else (90, 90, 100)
    draw.text((margin+12, panel_y+panel_h -18), "D-отладка  M-едем/стоим  Q-выход ", font=_ui_font(10), fill=hc)

    composited = Image.alpha_composite(img, ov)
    return cv2.cvtColor(np.array(composited.convert("RGB ")), cv2.COLOR_RGB2BGR)

# =============================================================================
# EN: SQLITE LOGGER & TRIP TRACKING
# RU: ЛОГИРОВАНИЕ В SQLITE И УЧЁТ ПОЕЗДОК
# =============================================================================
DB_PATH = os.path.join(os.path.dirname(__file__), "events.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, level TEXT NOT NULL,
            reason TEXT NOT NULL, ear REAL, mar REAL, pitch REAL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL, level TEXT, reason TEXT,
            ear REAL, pitch REAL, yaw REAL, roll REAL, jaw_open REAL,
            perclos REAL, blink_rate REAL, microsleeps INTEGER,
            drowsiness REAL, distraction REAL, nodding INTEGER,
            warnings INTEGER, yawn_count INTEGER, is_moving INTEGER
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time REAL NOT NULL, end_time REAL, duration_sec REAL,
            total_warnings INTEGER, total_yawns INTEGER,
            max_drowsiness REAL, avg_perclos REAL, grade TEXT
        )""")
    conn.commit()
    conn.close()

def log_event(level, reason, ear, mar, pitch):
    if level == "GREEN":
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (timestamp, level, reason, ear, mar, pitch) VALUES (?,?,?,?,?,?)",
        (time.time(), level, reason, round(ear, 3), round(mar, 3), round(pitch, 3))
    )
    conn.commit()
    conn.close()

def log_metrics(level, reason, ear, pitch, yaw, roll, jaw_open,
                perclos, blink_rate, microsleeps, drowsiness,
                distraction, nodding, warnings, yawn_count, is_moving):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO metrics_log
           (timestamp, level, reason, ear, pitch, yaw, roll, jaw_open,
            perclos, blink_rate, microsleeps, drowsiness, distraction,
            nodding, warnings, yawn_count, is_moving)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (time.time(), level, reason,
         round(ear, 3), round(pitch, 3), round(yaw, 3), round(roll, 3), round(jaw_open, 3),
         round(perclos, 3), round(blink_rate, 1), microsleeps, round(drowsiness, 1),
         round(distraction, 1), 1 if nodding else 0, warnings, yawn_count, 1 if is_moving else 0)
    )
    conn.commit()
    conn.close()

def start_trip():
    conn    = sqlite3.connect(DB_PATH)
    cur     = conn.execute("INSERT INTO trips (start_time) VALUES (?)", (time.time(),))
    trip_id = cur.lastrowid
    conn.commit()
    conn.close()
    return trip_id

def end_trip(trip_id, total_warnings, total_yawns, max_drowsiness, avg_perclos):
    conn  = sqlite3.connect(DB_PATH)
    start = conn.execute("SELECT start_time FROM trips WHERE id=?", (trip_id,)).fetchone()
    now   = time.time()
    dur   = now - start[0] if start else 0
    if total_warnings == 0 and max_drowsiness < 30:
        grade = "A "
    elif total_warnings <= 1 and max_drowsiness < 50:
        grade = "B "
    elif total_warnings <= 2 and max_drowsiness < 70:
        grade = "C "
    else:
        grade = "D "
    conn.execute(
        """UPDATE trips SET end_time=?, duration_sec=?, total_warnings=?,
           total_yawns=?, max_drowsiness=?, avg_perclos=?, grade=? WHERE id=?""",
        (now, dur, total_warnings, total_yawns,
         round(max_drowsiness, 1), round(avg_perclos, 3), grade, trip_id)
    )
    conn.commit()
    conn.close()
    return grade

# =============================================================================
# EN: DEBUG VISUALIZATION (LANDMARKS & GAZE BOX)
# RU: ОТЛАДОЧНАЯ ВИЗУАЛИЗАЦИЯ (ЛАНДМАРКИ И КВАДРАТ ВЗГЛЯДА)
# =============================================================================
_DBG_FONT_CACHE: dict = {}
def _dbg_font(size):
    if size not in _DBG_FONT_CACHE:
        try:
            _DBG_FONT_CACHE[size] = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except Exception:
            _DBG_FONT_CACHE[size] = ImageFont.load_default()
    return _DBG_FONT_CACHE[size]

def _px(lm_pt, w, h):
    return int(lm_pt.x * w), int(lm_pt.y * h)

_GROUPS = [
    (LEFT_EYE,                    (0, 220, 0)),
    (RIGHT_EYE,                   (0, 220, 0)),
    (MOUTH,                       (0, 160, 255)),
    ([NOSE_TIP, CHIN, FOREHEAD],  (220, 0, 220)),
]

def draw_face_debug(frame, all_landmarks, gaze_label, ratio_x, ratio_y, ear, mar):
    if not all_landmarks:
        return frame
    lm  = all_landmarks[0]
    h, w = frame.shape[:2]
    for i in range(min(468, len(lm))):
        cv2.circle(frame, _px(lm[i], w, h), 1, (70, 70, 70), -1)

    for indices, color in _GROUPS:
        pts = [_px(lm[i], w, h) for i in indices]
        for p in pts:
            cv2.circle(frame, p, 3, color, -1)
        for j in range(len(pts)):
            cv2.line(frame, pts[j], pts[(j+1) % len(pts)], color, 1, cv2.LINE_AA)

    if len(lm) >= 478:
        for iris_idx, eye_indices in [(L_IRIS, LEFT_EYE), (R_IRIS, RIGHT_EYE)]:
            cx, cy = _px(lm[iris_idx], w, h)
            eye_pts = np.array([_px(lm[i], w, h) for i in eye_indices])
            dists   = np.linalg.norm(eye_pts - np.array([cx, cy]), axis=1)
            radius  = max(int(np.mean(dists) * 0.85), 5)
            cv2.circle(frame, (cx, cy), radius,  (0, 210, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)

    BOX_W, BOX_H = 160, 80
    PAD = 10
    x0 = w - BOX_W - PAD
    y0 = h - BOX_H - PAD - 90
    cv2.rectangle(frame, (x0, y0), (x0+BOX_W, y0+BOX_H), (30,30,30), -1)
    cv2.rectangle(frame, (x0, y0), (x0+BOX_W, y0+BOX_H), (80,80,80), 1)
    ec_x, ec_y = x0+BOX_W//2, y0+BOX_H//2
    cv2.ellipse(frame, (ec_x, ec_y), (BOX_W//2-12, BOX_H//2-10), 0, 0, 360, (180,180,180), 1, cv2.LINE_AA)
    cv2.line(frame, (ec_x, y0+4),   (ec_x, y0+BOX_H-4), (50,50,50), 1)
    cv2.line(frame, (x0+4, ec_y),   (x0+BOX_W-4, ec_y), (50,50,50), 1)
    px_ = int(x0+15+ratio_x*(BOX_W-30))
    py_ = int(y0+12+ratio_y*(BOX_H-24))
    px_ = max(x0+10, min(px_, x0+BOX_W-10))
    py_ = max(y0+8,  min(py_, y0+BOX_H-8))
    cv2.circle(frame, (px_, py_), 10, (255, 160, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (px_, py_), 10, (255, 220, 100), 2, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-90), (w, h), (20,20,20), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img_pil)
    color = (0, 230, 0) if gaze_label == "Прямо " else (80, 200, 255)
    d.text((12, h-88), f"Взгляд: {gaze_label} ", font=_dbg_font(24), fill=color)
    d.text((12, h-60), f"X: {ratio_x:.2f}   Y: {ratio_y:.2f} ", font=_dbg_font(17), fill=(170,170,170))
    d.text((12, h-38), f"EAR={ear:.3f}  MAR={mar:.3f} ", font=_dbg_font(17), fill=(170,170,170))
    d.text((12, h-16), "D — скрыть отладку   Q — выход ", font=_dbg_font(17), fill=(90,90,90))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =============================================================================
# EN: MAIN LOOP & ENTRY POINT
# RU: ГЛАВНЫЙ ЦИКЛ И ТОЧКА ВХОДА
# =============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
COLORS = {
    "GREEN":  (34,  139, 34),
    "YELLOW": (0,   200, 220),
    "ORANGE": (0,   120, 255),
    "RED":    (30,  30,  200),
    "BLACK":  (30,  30,  30),
}

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Загружаю модель лица (~30 МБ)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Модель загружена.")

def main():
    init_db()
    download_model()
    
    # EN: Initialize MediaPipe FaceLandmarker in VIDEO mode
    # RU: Инициализируем MediaPipe FaceLandmarker в режиме ВИДЕО
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO ,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker  = mp_vision.FaceLandmarker.create_from_options(options)

    sanctions  = SanctionSystem()
    calibrator = Calibrator()
    cap        = cv2.VideoCapture(0)
    debug_mode = False
    is_moving  = True

    if not cap.isOpened():
        print("Ошибка: камера не найдена. ")
        return

    trip_id           = start_trip()
    _last_metrics_log = 0.0
    print("SafeDriver запущен.  D - отладка   M - едем/стоим   Q - выход ")

    prev_level        = "GREEN "
    ear = mar = pitch = roll = yaw_val = 0.0
    blink_left = blink_right = -1.0
    jaw_open          = 0.0
    gaze_label        = "Прямо "
    ratio_x = ratio_y = 0.5
    _last_gaze_beep   = 0.0
    _beep_hold_until  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now   = time.time()
        frame  = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_img, int(now * 1000))

        face_detected = bool(result.face_landmarks)

        if face_detected:
            lm    = result.face_landmarks[0]
            ear_l = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear   = (ear_l + ear_r) / 2.0
            mar   = mouth_aspect_ratio(lm, w,  h)
            pitch, yaw_val, roll = head_angles(lm, w, h)

            jaw_open = blink_left = blink_right = 0.0
            bs_dict  = {}
            if result.face_blendshapes:
                bs_dict     = {b.category_name: b.score for b in result.face_blendshapes[0]}
                jaw_open    = bs_dict.get("jawOpen ",       0.0)
                blink_left  = bs_dict.get("eyeBlinkLeft ",  -1.0)
                blink_right = bs_dict.get("eyeBlinkRight ", -1.0)

            gaze_label, ratio_x, ratio_y = gaze_direction(lm, ear)
        else:
            ear, mar, pitch, roll, yaw_val = 0.30, 0.0, 0.0, 0.0, 0.0
            jaw_open = blink_left = blink_right = 0.0
            bs_dict  = {}
            gaze_label, ratio_x, ratio_y = "Прямо ", 0.5, 0.5

        # EN: Personal calibration phase (first 4 seconds)
        # RU: Фаза персональной калибровки (первые 4 секунды)
        if not calibrator.done:
            calibrator.feed(ear, jaw_open, face_detected, now)
            if calibrator.done:
                sanctions.set_calibration(
                    calibrator.ear_closed,
                    calibrator.ear_squint,
                    calibrator.jaw_open_thresh,
                )
            progress = calibrator.progress
            overlay  = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            frame = fill_rect_with_text(
                frame, (0, h//2-50, w, h//2+50), (30, 30, 30),
                [("Калибровка... смотрите прямо в камеру ", 26, (255,255,255))]
            )
            bar_w = int(w * progress)
            cv2.rectangle(frame, (40, h//2+60), (w-40, h//2+80), (80,80,80), -1)
            cv2.rectangle(frame, (40, h//2+60), (40+bar_w-80, h//2+80), (0,200,80), -1)
            cv2.imshow("SafeDriver ", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        level, reason = sanctions.update(
            ear, mar, pitch, face_detected, gaze_label, roll, jaw_open,
            blink_left, blink_right, blendshapes=bs_dict,
            is_moving=is_moving, yaw=yaw_val,
        )

        # EN: Trigger alerts on level change or critical states
        # RU: Вызываем оповещения при смене уровня или критических состояниях
        if level != "GREEN " and (level != prev_level or level in ("ORANGE ", "BLACK ")):
            play_alert(level, reason)
            log_event(level, reason, ear, mar, pitch)

        if level in ("YELLOW ", "ORANGE ", "RED ", "BLACK "):
            _beep_hold_until = now + 5.0

        if level in ("YELLOW ", "ORANGE ", "RED ", "BLACK ") or now < _beep_hold_until:
            play_beep_only(BEEP_FREQ, 1)

        gd = sanctions.gaze_down_seconds
        if 2.0 < gd < 3.9 and now - _last_gaze_beep > 2.0:
            play_beep_only(BEEP_FREQ, 1)
            _last_gaze_beep = now

        prev_level = level

        # EN: Log metrics every 1 second to avoid DB bottleneck
        # RU: Логируем метрики раз в секунду, чтобы не перегружать БД
        if now - _last_metrics_log > 1.0:
            _last_metrics_log = now
            m = sanctions.metrics
            log_metrics(
                level, reason, ear, pitch, yaw_val, roll, jaw_open,
                m.perclos, m.blink_rate, m.microsleep_count,
                m.drowsiness_score, m.distraction_score,
                m.nodding_detected, sanctions.total_warnings,
                sanctions._yawn_count, is_moving,
            )

        if debug_mode and face_detected:
            frame = draw_face_debug(frame, result.face_landmarks,
                                    gaze_label, ratio_x, ratio_y, ear, mar)

        frame = draw_mobile_ui(
            frame, level, reason, ear, mar, pitch, jaw_open,
            gaze_label, sanctions.total_warnings, sanctions._yawn_count,
            debug_mode, warnings_max=MAX_WARNINGS, is_moving=is_moving,
            metrics=sanctions.metrics,
        )

        cv2.imshow("SafeDriver ", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('d'), ord('D')):
            debug_mode = not debug_mode
            print(f"Режим отладки: {'ВКЛ' if debug_mode else 'ВЫКЛ'} ")
        elif key in (ord('m'), ord('M')):
            is_moving = not is_moving
            print(f"Движение: {'ЕДЕМ' if is_moving else 'СТОИМ'} ")

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()

    m     = sanctions.metrics
    grade = end_trip(trip_id, sanctions.total_warnings, sanctions._yawn_count,
                     m.drowsiness_score, m.perclos)
    print(f"SafeDriver остановлен. Оценка поездки: {grade} ")

if __name__ == "__main__":
    main()