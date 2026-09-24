# AIA-T003-LAB07 — Domain Shift Radar

MVP tool: nhận một batch dữ liệu production chưa có nhãn, đo mức domain shift so với source, xác định slice rủi ro, và kiểm chứng shift signal có tương quan với performance drop thật sự của model hay không.

Đề bài chi tiết: [docs/spec.md](docs/spec.md).

## Team — Nhóm 3, C2 Domain Shift Radar

| Thành viên | Vai trò | Module / Phạm vi |
| --- | --- | --- |
| Nhữ Đình Chiến (Nhóm trưởng, GitHub: `Chien27803`) | Dev — Module B | Shift Score Engine, Slice Analyzer, Risk Dashboard |
| Khúc Việt Anh (GitHub: `BanhKhuc04`) | Dev — Module A | Project Setup, Source Domain Registry, Target Batch Ingestion, Embedding/Inference Engine |
| Trịnh Quang Trung (GitHub: `toilatrung`) | Dev — Module C | Ground Truth/Performance Evaluation, Correlation Validator, False Alarm/Miss Explorer, Report Export |
| Võ Trọng Nghĩa | Slide Designer | Chuẩn bị slide thuyết trình / demo |
| Lê Minh Trí | Data Collector | Thu thập, chuẩn bị dataset source và target |
| Ngô Văn Hưng | BA | Đặc tả yêu cầu, review tính năng so với đề bài |
| Nguyễn Đăng Huấn | Tester | Kiểm thử chức năng, báo bug |

Chỉ 3 dev (Chiến/Việt Anh/Trung) code trực tiếp trong repo, theo 3 module dọc ở trên. Chi tiết quy trình làm việc, nhánh git, ranh giới module: [CONTRIBUTING.md](CONTRIBUTING.md).

## Stack

* UI: Streamlit (`frontend/`)
* Backend: FastAPI (`backend/app/`)
* ML: PyTorch, Ultralytics, NumPy, SciPy, scikit-learn
* DB: SQLite (demo) qua SQLAlchemy
* Deployment: Docker Compose

## Cấu trúc thư mục

```text
backend/
  app/
    main.py            # FastAPI entrypoint, gắn router từng module
    core/               # config, db session, schema dùng chung
    modules/
      source/           # Module A — Project Setup + Source Domain Registry
      batch/             # Module A — Target Batch Ingestion + Embedding/Inference
      shift/               # Module B — Shift Score Engine
      slice/               # Module B — Slice Analyzer + Risk Dashboard
      evaluation/          # Module C — Ground Truth + Performance Evaluation
      correlation/         # Module C — Correlation Validator
      failure/             # Module C — False Alarm/Miss Explorer
      report/               # Module C — Report Export
frontend/
  app.py               # Streamlit entrypoint
  pages/                # 1 file Streamlit / màn hình MVP
data/
  source/               # dataset nguồn (không commit ảnh thật, xem .gitignore)
  target/                 # batch target upload
  artifacts/              # embedding, report xuất ra
docs/
  spec.md               # đề bài gốc
```

## Chạy thử local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Chạy end-to-end demo script của Module A
python scripts/demo_module_a.py

# backend
uvicorn backend.app.main:app --reload --port 8000

# frontend (terminal khác)
streamlit run frontend/app.py
```

Hoặc bằng Docker Compose:

```bash
docker compose up --build
```

## Chạy bộ kiểm thử (Pytest)

```bash
pytest backend -v
```

## Trạng thái

- **Module A (BanhKhuc04) — Source & Batch**: ✅ **COMPLETED / PRODUCTION READY**
  - Project Setup (FR 2.1): UI + API + Cấu hình lưu trữ
  - Source Domain Registry (FR 2.2): Baseline mAP, task-aware embeddings, bootstrap reference subsets
  - Target Batch Ingestion (FR 2.3): Upload ảnh thật + tạo batch domain shift nhân tạo tất định (Dark, Blur, Noise, JPEG...)
  - Embedding & Inference Engine (FR 2.4): Batched YOLO inference + trích xuất vector embedding L2-normalized 256-D
  - Handoff artifacts và API endpoints sẵn sàng cho Module B (`Chien27803`)
  - Tài liệu chi tiết: [docs/module_a.md](docs/module_a.md)
- **Module B (Chien27803) — Shift & Slice**: 🟡 STUB
- **Module C (toilatrung) — Performance & Report**: ✅ Performance Evaluation (mAP), Correlation Validator (Spearman ρ), False Alarm/Miss Explorer, Report Export (HTML) đã implement — xem `backend/app/modules/{evaluation,correlation,failure,report}/`
