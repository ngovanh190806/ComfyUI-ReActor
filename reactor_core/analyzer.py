import os
import sys
import subprocess
import zipfile
import urllib.request
import folder_paths
from scripts.reactor_logger import logger
from insightface.model_zoo import model_zoo

# Tự động kiểm tra và cài đặt thư viện insightface
try:
    import insightface
except ImportError:
    print("[ReActor] Thiếu module 'insightface'. Đang tiến hành cài đặt tự động...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "insightface"])

from insightface.app import FaceAnalysis

# Xác định đường dẫn gốc
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
ANTELOPEV2_DIR = os.path.join(INSIGHTFACE_DIR, "models", "antelopev2")
BUFFALO_L_DIR = os.path.join(INSIGHTFACE_DIR, "models", "buffalo_l")
ANTELOPEV2_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

def prepare_buffalo_l():
    """Tự động tải và giải nén buffalo_l nếu chưa tồn tại."""
    if not os.path.exists(BUFFALO_L_DIR):
        os.makedirs(BUFFALO_L_DIR, exist_ok=True)
    w600k_path = os.path.join(BUFFALO_L_DIR, "w600k_r50.onnx")
    if not os.path.exists(w600k_path):
        logger.status("[ReActor] Không tìm thấy w600k_r50.onnx, đang tiến hành tự động tải buffalo_l...")
        zip_path = os.path.join(INSIGHTFACE_DIR, "models", "buffalo_l.zip")
        urllib.request.urlretrieve(BUFFALO_L_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BUFFALO_L_DIR)
        os.remove(zip_path)
        logger.status("[ReActor] Tải và nạp w600k_r50.onnx hoàn tất!")

class ReActorFaceAnalysis(FaceAnalysis):
    """
    Class quản lý nhận diện khuôn mặt thừa kế từ FaceAnalysis.
    Sử dụng antelopev2 làm mặc định và ghi đè recognition bằng buffalo_l.
    """
    def __init__(self, name='antelopev2', root=INSIGHTFACE_DIR, allowed_modules=None, providers=None, **kwargs):
        prepare_buffalo_l() # Kích hoạt auto-download buffalo_l
        self.root = root
        self.providers = providers or ["CPUExecutionProvider"]
        
        # Tự động tải antelopev2 nếu chưa tồn tại
        self._prepare_antelopev2()
        
        # 1. Khởi tạo toàn bộ pipeline bằng antelopev2
        super().__init__(name=name, root=root, allowed_modules=allowed_modules, providers=self.providers, **kwargs)
        
        # 2. Ghi đè (override) mô hình recognition bằng w600k_r50.onnx của buffalo_l
        rec_model_path = os.path.join(BUFFALO_L_DIR, 'w600k_r50.onnx')
        
        if os.path.exists(rec_model_path):
            logger.status("[ReActor] Đang ghi đè Recognition Model bằng buffalo_l/w600k_r50...")
            rec_model = model_zoo.get_model(rec_model_path, providers=self.providers)
            self.models['recognition'] = rec_model
        else:
            logger.error("[ReActor] CẢNH BÁO: Không tìm thấy w600k_r50.onnx trong buffalo_l!")

    def _prepare_antelopev2(self):
        if not os.path.exists(ANTELOPEV2_DIR):
            logger.status(f"[ReActor] Không tìm thấy antelopev2. Đang tự động tải...")
            os.makedirs(os.path.join(self.root, "models"), exist_ok=True)
            zip_path = os.path.join(self.root, "models", "antelopev2.zip")
            
            urllib.request.urlretrieve(ANTELOPEV2_URL, zip_path)
            
            logger.status(f"[ReActor] Đang giải nén antelopev2...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(self.root, "models"))
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
            logger.status("[ReActor] Tải và giải nén antelopev2 hoàn tất.")

    def prepare(self, ctx_id=0, det_size=(640, 640), det_thresh=0.5):
        self.det_thresh = det_thresh
        super().prepare(ctx_id=ctx_id, det_size=det_size)

    def get(self, img, max_num=0):
        faces = super().get(img, max_num=max_num)
        if hasattr(self, 'det_thresh') and self.det_thresh > 0:
            faces = [f for f in faces if f.det_score >= self.det_thresh]
        return faces
