import cv2
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize

# Import Hybrid Analyzer mới
import folder_paths
from scripts.reactor_swapper import getAnalysisModel

from r_facelib.parsing import init_parsing_model
from r_facelib.utils.misc import img2tensor, imwrite


def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class FaceRestoreHelper(object):
    """Helper for the face restoration pipeline using ReActor Hybrid Analyzer."""

    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50', # Tham số này bị bỏ qua
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None):
        
        self.template_3points = template_3points 
        self.upscale_factor = upscale_factor
        self.crop_ratio = crop_ratio
        self.face_size = (int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))

        # Cấu hình template khuôn mặt
        if self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * (self.crop_ratio[1] - 1) / 2
            
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- ÉP SỬ DỤNG HYBRID ANALYZER TỪ REACTOR ---
        # Không khởi tạo lại analyzer, dùng getAnalysisModel để lấy instance đã cache
        self.analyzer = getAnalysisModel((640, 640))

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = init_parsing_model(model_name='parsenet', device=self.device)

    def read_image(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
        if np.max(img) > 256: img = img / 65535 * 255
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4: img = img[:, :, 0:3]
        self.input_img = img
        if min(self.input_img.shape[:2]) < 512:
            f = 512.0 / min(self.input_img.shape[:2])
            self.input_img = cv2.resize(self.input_img, (0,0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)

    def get_face_landmarks_5(self, only_keep_largest=False, only_center_face=False, resize=None, blur_ratio=0.01, eye_dist_threshold=None):
        self.all_landmarks_5 = []
        self.det_faces = []
        
        img = self.input_img
        if resize:
            h, w = img.shape[:2]
            scale = resize / min(h, w)
            scale = max(1, scale)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1

        # Lấy kết quả từ Hybrid Analyzer
        faces = self.analyzer.get(img)
        
        if len(faces) == 0:
            return 0

        # Lưu lại để xử lý logic filter
        bboxes = []
        landmarks = []
        for face in faces:
            bboxes.append(np.append(face.bbox, face.det_score) / scale)
            landmarks.append(face.kps / scale)

        self.det_faces = np.array(bboxes)
        self.all_landmarks_5 = list(landmarks)
            
        # Logic chọn mặt
        if only_keep_largest:
            largest_face, idx = get_largest_face(self.det_faces, *self.input_img.shape[:2])
            self.det_faces, self.all_landmarks_5 = [largest_face], [self.all_landmarks_5[idx]]
        elif only_center_face:
            center_face, idx = get_center_face(self.det_faces, *self.input_img.shape[:2])
            self.det_faces, self.all_landmarks_5 = [center_face], [self.all_landmarks_5[idx]]
            
        return len(self.all_landmarks_5)

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align và sửa lỗi landmark 68 -> 5 điểm nếu cần."""
        for idx, landmark in enumerate(self.all_landmarks_5):
            if landmark is None or len(landmark) == 0: continue
            
            # Nếu nhận được 68 điểm từ AntelopeV2, quy đổi về 5 điểm cho cv2.estimateAffinePartial2D
            pts = np.array(landmark, dtype=np.float32)
            if pts.shape == (68, 2):
                pts = np.array([
                    np.mean(pts[36:42], axis=0), np.mean(pts[42:48], axis=0), 
                    pts[30], pts[48], pts[54]
                ], dtype=np.float32)

            affine_matrix = cv2.estimateAffinePartial2D(pts, self.face_template, method=cv2.LMEDS)[0]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            cropped_face = cv2.warpAffine(
                input_img, affine_matrix, self.face_size, borderMode=border_mode, borderValue=(135, 133, 132))  # gray
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Get inverse affine matrix."""
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)


    def add_restored_face(self, face):
        self.restored_faces.append(face)


    def paste_faces_to_input_image(self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None):
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        if upsample_img is None:
            # simply resize the background
            # upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
            upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
        else:
            upsample_img = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')
        
        inv_mask_borders = []
        for restored_face, inverse_affine in zip(self.restored_faces, self.inverse_affine_matrices):
            if face_upsampler is not None:
                restored_face = face_upsampler.enhance(restored_face, outscale=self.upscale_factor)[0]
                inverse_affine /= self.upscale_factor
                inverse_affine[:, 2] *= self.upscale_factor
                face_size = (self.face_size[0]*self.upscale_factor, self.face_size[1]*self.upscale_factor)
            else:
                # Add an offset to inverse affine matrix, for more precise back alignment
                if self.upscale_factor > 1:
                    extra_offset = 0.5 * self.upscale_factor
                else:
                    extra_offset = 0
                inverse_affine[:, 2] += extra_offset
                face_size = self.face_size
            inv_restored = cv2.warpAffine(restored_face, inverse_affine, (w_up, h_up))

            # if draw_box or not self.use_parse:  # use square parse maps
            #     mask = np.ones(face_size, dtype=np.float32)
            #     inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            #     # remove the black borders
            #     inv_mask_erosion = cv2.erode(
            #         inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            #     pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            #     total_face_area = np.sum(inv_mask_erosion)  # // 3
            #     # add border
            #     if draw_box:
            #         h, w = face_size
            #         mask_border = np.ones((h, w, 3), dtype=np.float32)
            #         border = int(1400/np.sqrt(total_face_area))
            #         mask_border[border:h-border, border:w-border,:] = 0
            #         inv_mask_border = cv2.warpAffine(mask_border, inverse_affine, (w_up, h_up))
            #         inv_mask_borders.append(inv_mask_border)
            #     if not self.use_parse:
            #         # compute the fusion edge based on the area of face
            #         w_edge = int(total_face_area**0.5) // 20
            #         erosion_radius = w_edge * 2
            #         inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            #         blur_size = w_edge * 2
            #         inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            #         if len(upsample_img.shape) == 2:  # upsample_img is gray image
            #             upsample_img = upsample_img[:, :, None]
            #         inv_soft_mask = inv_soft_mask[:, :, None]

            # always use square mask
            mask = np.ones(face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask, np.ones((int(2 * self.upscale_factor), int(2 * self.upscale_factor)), np.uint8))
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)  # // 3
            # add border
            if draw_box:
                h, w = face_size
                mask_border = np.ones((h, w, 3), dtype=np.float32)
                border = int(1400/np.sqrt(total_face_area))
                mask_border[border:h-border, border:w-border,:] = 0
                inv_mask_border = cv2.warpAffine(mask_border, inverse_affine, (w_up, h_up))
                inv_mask_borders.append(inv_mask_border)
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
            if len(upsample_img.shape) == 2:  # upsample_img is gray image
                upsample_img = upsample_img[:, :, None]
            inv_soft_mask = inv_soft_mask[:, :, None]

            # parse mask
            if self.use_parse:
                # inference
                face_input = cv2.resize(restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)
                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()

                parse_mask = np.zeros(out.shape)
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for idx, color in enumerate(MASK_COLORMAP):
                    parse_mask[out == idx] = color
                #  blur the mask
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                # remove the black borders
                thres = 10
                parse_mask[:thres, :] = 0
                parse_mask[-thres:, :] = 0
                parse_mask[:, :thres] = 0
                parse_mask[:, -thres:] = 0
                parse_mask = parse_mask / 255.

                parse_mask = cv2.resize(parse_mask, face_size)
                parse_mask = cv2.warpAffine(parse_mask, inverse_affine, (w_up, h_up), flags=3)
                inv_soft_parse_mask = parse_mask[:, :, None]
                # pasted_face = inv_restored
                fuse_mask = (inv_soft_parse_mask<inv_soft_mask).astype('int')
                inv_soft_mask = inv_soft_parse_mask*fuse_mask + inv_soft_mask*(1-fuse_mask)

            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:  # alpha channel
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + (1 - inv_soft_mask) * upsample_img

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)

        # draw bounding box
        if draw_box:
            # upsample_input_img = cv2.resize(input_img, (w_up, h_up))
            img_color = np.ones([*upsample_img.shape], dtype=np.float32)
            img_color[:,:,0] = 0
            img_color[:,:,1] = 255
            img_color[:,:,2] = 0
            for inv_mask_border in inv_mask_borders:
                upsample_img = inv_mask_border * img_color + (1 - inv_mask_border) * upsample_img
                # upsample_input_img = inv_mask_border * img_color + (1 - inv_mask_border) * upsample_input_img

        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        return upsample_img

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []
