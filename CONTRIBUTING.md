# Contributing

## Phân công module (vertical slice — end-to-end backend + UI)

| Module | Người phụ trách | Backend (`backend/app/modules/`) | Frontend (`frontend/pages/`) | FR trong docs/spec.md |
| --- | --- | --- | --- | --- |
| **A — Source & Batch** | Khúc Việt Anh (`BanhKhuc04`) | `source/`, `batch/` | `1_Projects.py`, `2_Source_Baseline.py`, `3_Batch_Analysis.py` | 2.1, 2.2, 2.3, 2.4 |
| **B — Shift & Slice** | Nhữ Đình Chiến (`Chien27803`) | `shift/`, `slice/` | `4_Slice_Explorer.py` (+ risk dashboard trong Batch Analysis) | 2.5, 2.6, 2.7 |
| **C — Performance & Report** | Trịnh Quang Trung (`toilatrung`) | `evaluation/`, `correlation/`, `failure/`, `report/` | `5_Performance_Evaluation.py`, `6_Correlation_Report.py`, `7_Failure_Analysis.py`, `8_Export_Report.py` | 2.8, 2.9, 2.10, 2.11 |

## Vai trò hỗ trợ (không code trong repo)

| Vai trò | Người phụ trách | Việc chính |
| --- | --- | --- |
| Slide Designer | Võ Trọng Nghĩa | Chuẩn bị slide thuyết trình / demo |
| Data Collector | Lê Minh Trí | Thu thập, chuẩn bị dataset source và target cho 3 dev dùng |
| BA | Ngô Văn Hưng | Đặc tả yêu cầu chi tiết, review tính năng so với `docs/spec.md` |
| Tester | Nguyễn Đăng Huấn | Kiểm thử chức năng từng module, báo bug qua GitHub Issues |

Mỗi người sở hữu module của mình end-to-end (route FastAPI + logic + trang Streamlit tương ứng). Ranh giới giữa các module là API/schema — không sửa trực tiếp code trong module của người khác, mở issue hoặc PR riêng nếu cần thay đổi.

### Phụ thuộc giữa các module

* Module B đọc embedding/prediction stats do Module A tạo ra (từ Embedding & Inference Engine) — thống nhất schema output sớm.
* Module C đọc shift score/slice ranking từ Module B để tính Correlation Validator — thống nhất schema `Domain/Slice → shift_score` sớm.
* Khi đổi schema dùng chung, cập nhật trong `backend/app/core/schemas.py` và báo trong PR description để 2 người còn lại review.

## Git workflow

* Nhánh chính: `main` — luôn ở trạng thái chạy được (không bắt buộc hoàn chỉnh tính năng).
* Mỗi người làm trên nhánh riêng đã tạo sẵn:
  * `feature/source-batch` — BanhKhuc04
  * `feature/shift-slice` — Chien27803
  * `feature/performance-report` — toilatrung
* Commit nhỏ, message rõ ràng theo dạng `<module>: <mô tả ngắn>` (vd: `shift: implement MMD embedding distance`).
* Mở Pull Request vào `main`, cần ít nhất 1 người khác review trước khi merge.
* Không force-push lên `main`.

## Trước khi code

1. Đọc kỹ `docs/spec.md`, phần FR của module mình phụ trách.
2. Thống nhất schema input/output với 2 module liền kề trong pipeline (xem sơ đồ ở `docs/spec.md` mục 1).
3. Cập nhật stub trong module của mình (`backend/app/modules/<module>/router.py`, `service.py`) và trang Streamlit tương ứng.

## Môi trường

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` thành `.env` nếu cần chỉnh cấu hình local (DB path, model checkpoint dir...).
