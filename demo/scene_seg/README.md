# Hướng dẫn chạy Phân đoạn Cảnh Video (Video Scene Segmentation)

Thư mục này chứa script chạy phân đoạn cảnh phim end-to-end dùng mô hình **SCRL (Scene Consistency Representation Learning)** kết hợp với phân loại **BiLSTM**.

---

## 1. Môi trường chạy (Environment)
Vui lòng sử dụng môi trường Conda đã được cấu hình sẵn tại:
*   **Đường dẫn Python:** `/data/ndloc_bk/ntVan/demo_env/bin/python` (Khuyên dùng)

---

## 2. Checkpoints của Mô hình
Script mặc định sử dụng các checkpoint đã được cấu hình tại:
1.  **SCRL Visual Encoder:** `/data/ndloc_bk/ntVan/video_caption/scene_seg/SceneSegmentation-SCRL/output/checkpoint_0099.pth.tar`
2.  **BiLSTM Scene Segmenter:** `/data/ndloc_bk/ntVan/video_caption/scene_seg/SceneSegmentation-SCRL/SceneSeg/output/seg_checkpoints/best/model_best.pth.tar`

*(Có thể tùy biến đường dẫn thông qua tham số dòng lệnh).*

---

## 3. Cách chạy chính (Usage)

Chạy phân đoạn video trực tiếp bằng cách thực thi lệnh sau (khuyên dùng cấu hình thư mục tạm trên RAM Disk `/dev/shm` để tránh nghẽn I/O trên ổ mạng NFS):

```bash
/data/ndloc_bk/ntVan/demo_env/bin/python predict_scenes.py \
  --video_path /đường_dẫn/video_phim.mkv \
  --output_json output/result.json \
  --temp_dir /dev/shm/temp_predict_demo
```

---

## 4. Các tham số dòng lệnh (CLI Arguments)

| Tham số | Kiểu dữ liệu | Mặc định | Chức năng |
| :--- | :--- | :--- | :--- |
| `--video_path` | `str` | *(Bắt buộc)* | Đường dẫn tới file video đầu vào. |
| `--output_json` | `str` | `output/result.json` | Đường dẫn lưu file kết quả JSON chứa timestamp các cảnh. |
| `--gpu_id` | `str` | `"0"` | ID của GPU sử dụng. |
| `--cut_scenes` | `bool` | `True` | Tự động cắt video thành các clip nhỏ `.mp4` tương ứng với mỗi phân cảnh. |
| `--output_scenes_dir`| `str` | `None` | Thư mục lưu các video cảnh đã cắt. Nếu để mặc định (`None`), thư mục con sẽ được tạo tự động cùng cấp với file JSON kết quả (ví dụ: `output/result_scenes/`). |
| `--temp_dir` | `str` | `./temp_predict_demo`| Thư mục chứa các file trung gian (ảnh keyframe, embedding). Khuyên dùng `/dev/shm/temp_predict_demo` để lưu trên RAM Disk. |
| `--keep_temp` | `flag` | *(Không bật)* | Giữ lại thư mục tạm sau khi chạy xong để kiểm tra (mặc định sẽ tự động xóa). |
| `--scrl_checkpoint`| `str` | `/data/ndloc...` | Đường dẫn tới checkpoint SCRL encoder. |
| `--bilstm_checkpoint`| `str` | `/data/ndloc...` | Đường dẫn tới checkpoint BiLSTM decoder. |
| `--thresh_high` | `float` | `0.5` | Ngưỡng xác suất cao để xác định biên cảnh (anchor boundary). |
| `--max_scene_shots` | `int` | `40` | Số shot tối đa trong một cảnh. Nếu vượt quá, cảnh sẽ bị chia nhỏ đệ quy. |
| `--split_prob_thresh`| `float` | `0.02` | Ngưỡng xác suất tối thiểu tại điểm cực đại địa phương (local peak) để chia nhỏ cảnh dài. |

---

## 5. Định dạng đầu ra (Outputs)

Khi chạy thành công, script sẽ sinh ra 2 kết quả đầu ra chính:

### File JSON chứa các mốc thời gian (`result.json`)
```json
[
  {
    "scene_id": 0,
    "start_time": 0.0,
    "end_time": 65.31525,
    "duration": 65.31525
  },
  {
    "scene_id": 1,
    "start_time": 65.31525,
    "end_time": 124.707917,
    "duration": 59.392667
  }
]
```

### Thư mục chứa các video cảnh đã cắt (`result_scenes/`)
Các video phân cảnh định dạng `.mp4` được cắt trực tiếp ở mức độ lossless (không re-encode):
*   `scene_0000.mp4`
*   `scene_0001.mp4`
*   ...


