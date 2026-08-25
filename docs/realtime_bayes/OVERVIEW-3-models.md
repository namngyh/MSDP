# Mang tầng real-time sang PC — tổng quan 3 model

Hướng dẫn chi tiết nằm **trong từng repo** (để đi theo git khi bạn clone trên PC):

| Model | Tài liệu | Trạng thái |
| --- | --- | --- |
| RAFF (VN-Index) | `docs/realtime_bayes/deploy_on_pc.md` | ✅ **Chạy được thật.** 115 test, verify end-to-end 6306 phiên, 4.15 s/phiên |
| Dynamic Graph (VN30) | `docs/realtime_bayes/deploy_on_pc.md` | 🔵 **Chưa xong.** 325 test, nhưng **chưa ghi `artifacts/latest/`** → website chưa cập nhật theo phiên |
| MSDP | `docs/realtime_bayes/phase3_status.md` | ⬜ **Chưa bắt đầu.** Mới có khảo sát |

> **Trả lời thẳng: chưa xong hết 3 mô hình.** Chỉ RAFF sẵn sàng nối website theo phiên.
> Dynamic Graph còn thiếu bước publish. MSDP chưa động tới.

File này và `realtime-bayes-roadmap.md` nằm ở thư mục cha **không thuộc repo nào**, nên chúng
sẽ **không** đi theo khi bạn clone. Bản copy của roadmap đã được đặt vào
`docs/realtime_bayes/` của cả ba repo. Cân nhắc tạo một repo điều phối nhỏ (`vind-models`) chỉ
chứa roadmap + docs chung, xem lý do ở cuối file.

---

## Thứ tự vận hành, giống nhau cho cả ba

```
run-all              ← tầng batch, chu kỳ dài (tuần với RAFF/Graph, tháng với MSDP)
   └─ ghi batch handoff
init-online-state    ← chạy NGAY SAU mỗi run-all, seed state từ handoff
update-latest        ← mỗi phiên mới, vài giây, KHÔNG refit gì
```

Mỗi `run-all` **reset** tầng online. Quên `init-online-state` thì `update-latest` vẫn chạy trên
state cũ mà không báo lỗi — đối chiếu `source_run_metadata` trong manifest để phát hiện.

Ba model **độc lập**, không cần chờ nhau. Không có daemon Python — Windows Task Scheduler gọi
`.bat`, đúng triết lý sẵn có của cả ba repo.

---

## Ba việc cần làm đầu tiên trên PC

### 1. RAFF — điền `data.source` rồi chạy

```bash
cd RAFF/RARF-FHE-Regime-Aware-Random-Forest-Forecasting-Hybrid-Engine
python -m pip install -e . && python -m pytest -q        # kỳ vọng 115 passed
# sửa configs/default.yaml -> data.source (backend/path/table/column_map)
python -m vnindex_model.cli run-all           --config configs/default.yaml
python -m vnindex_model.cli init-online-state --config configs/default.yaml
python -m vnindex_model.cli update-latest     --config configs/default.yaml
```

Website đọc `artifacts/forecasts/latest_forecast.csv` + `latest_forecast_summary.json` như cũ —
schema không đổi, đã khóa bằng test.

### 2. Dynamic Graph — tạo `config/local.yaml` rồi chạy để **đo**

```bash
cd "Dynamic Graph"
python -m pip install -e ".[db,community]" && python -m pytest -q   # kỳ vọng 325 passed
cp config/local.example.yaml config/local.yaml
python -m dynamicgraph.cli discover-data      # dò DB, điền database_path
python -m dynamicgraph.cli run-all            --config config/local.yaml
python -m dynamicgraph.cli init-online-state  --config config/local.yaml
python -m dynamicgraph.cli update-latest      --config config/local.yaml
```

Báo lại `elapsed_seconds` và `number_of_nodes` — đó là ngân sách thời gian thật và kích thước
đồ thị thật, hai con số tôi chưa có.

### 3. MSDP — chưa cần làm gì

---

## Ba câu hỏi mở — Claude bên PC tự trả lời, không cần bạn biết

Mỗi repo có `docs/realtime_bayes/CLAUDE_HANDOFF.md` viết cho agent, kèm lệnh dò cụ thể.

| Câu hỏi | Cách tự trả lời |
| --- | --- |
| DB nằm đâu, bảng/cột tên gì? | RAFF: `python -m vnindex_model.cli discover-source <path>` — đọc read-only, in ra khối `data.source` dán thẳng được. Dynamic Graph: `python -m dynamicgraph.cli discover-data` |
| Website đọc dữ liệu kiểu gì? | **Đã trả lời cho Dynamic Graph**: FastAPI `dynamicgraph.api.app` phục vụ file tĩnh dưới `artifacts/latest/`, cache theo mtime → ghi đè file là API nhận ngay, không cần restart. Còn lại chỉ cần xác nhận frontend gọi API này hay đọc file trực tiếp |
| `production_model.pt` của MSDP ở đâu? | `Get-ChildItem -Recurse -Filter production_model.pt` trong repo MSDP |

## Còn lại bao nhiêu việc

| Hạng mục | Trạng thái |
| --- | --- |
| Phase 1 — RAFF | ✅ xong |
| Phase 2 — Dynamic Graph | 🔵 còn: đóng băng model dự báo stress + cache metric history → publish `artifacts/latest/` |
| Phase 3 — MSDP | ⬜ chưa bắt đầu (Hedge cho gate, port ACI từ RAFF, data source, CLI) |
| Bộ điều phối chung | ⬜ chưa bắt đầu (service nhỏ poll DB rồi gọi 3 CLI) |

---

## Về việc gộp repo

Ba repo hiện tách riêng, mỗi cái một remote GitHub (`namngyh/RARF-FHE-...`,
`namngyh/Dynamic-Graph`, `namngyh/MSDP` — repo MSDP nằm ở `MSDP/MSDP/`, sâu một cấp).

Khuyến nghị **giữ tách**: ba stack phụ thuộc gần như rời nhau (`arch`+`hmmlearn` /
`networkx`+`typer`+`igraph` / `torch`+`optuna`), cả ba đều commit `reports/` và `artifacts/`
(riêng RAFF đã 402/499 file tracked là reports), lịch sử đã public, và hiện **chưa có dòng code
nào dùng chung**. Gộp lại sẽ buộc mọi clone kéo cả ba bộ artifact và mọi env phải cài `torch` để
chạy test HMM.

Cái thật sự thiếu — tài liệu điều phối không được version control — giải bằng một repo nhỏ
`vind-models` chỉ chứa roadmap + docs chung + (sau này) bộ điều phối. Vài chục KB, không artifact.
