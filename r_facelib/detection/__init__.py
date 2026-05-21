import numpy as np

class HybridDetectorWrapper:
    def __init__(self):
        # Móc nối trực tiếp vào bộ nhớ Hybrid đang chạy của ReActor Swapper
        # Việc import bên trong __init__ giúp tránh lỗi Circular Import
        from scripts.reactor_swapper import getAnalysisModel
        self.analyzer = getAnalysisModel((640, 640))
        
    def __call__(self, img, *args, **kwargs):
        # Lấy threshold từ cấu hình truyền vào (nếu có), mặc định 0.5
        conf_threshold = kwargs.get('conf_threshold', 0.5)
        if len(args) > 0:
            # Xử lý trường hợp args truyền vào là tuple (thường gặp trong facexlib)
            conf_threshold = args[0] if isinstance(args, (list, tuple)) else args
            
        # img được FaceRestoreHelper đưa vào dạng BGR array chuẩn
        faces = self.analyzer.get(img)
        bboxes = []
        landmarks = []
        
        for face in faces:
            if face.det_score >= conf_threshold:
                # Đóng gói theo chuẩn format mà FaceRestoreHelper cần (x1, y1, x2, y2, score)
                bboxes.append(np.append(face.bbox, face.det_score))
                landmarks.append(face.kps)
                
        if len(bboxes) == 0:
            # Trả về mảng rỗng tương thích với định dạng của Facexlib
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)
            
        return np.array(bboxes, dtype=np.float32), np.array(landmarks, dtype=np.float32)
    
    def detect_faces(self, img, *args, **kwargs):
        return self.__call__(img, *args, **kwargs)
        
    # Tạo các hàm giả lập (dummy) để chống lỗi AttributeError khi facexlib gọi Pytorch functions
    def eval(self): return self
    def to(self, *args, **kwargs): return self
    def float(self): return self
    def half(self): return self
    def cpu(self): return self
    def cuda(self): return self

def init_detection_model(model_name, *args, **kwargs):
    """
    Hàm khởi tạo ghi đè: Bỏ qua RetinaFace/YOLO, ép 100% Face Restorer 
    dùng chung lõi Hybrid đang chạy của ReActor.
    """
    print(f"[Face Restore] Đã chặn tải model {model_name} cũ. Ép sử dụng 100% lõi Hybrid_AntelopeV2 trên RAM!")
    return HybridDetectorWrapper()
