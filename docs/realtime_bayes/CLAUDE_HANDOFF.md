# Bàn giao cho phiên Claude chạy trên PC — MSDP

Viết cho **agent**, không phải cho người dùng. Người dùng không biết chi tiết hạ tầng và đã nói
rõ là để bạn tự dò. Đừng hỏi họ những gì bạn tự tìm được.

## 1. Đọc trước, đừng khảo sát lại

- `docs/realtime_bayes/phase3_status.md` — khảo sát, cái gì đã có sẵn, cái gì đã làm, cái gì còn.
- `docs/realtime_bayes/realtime-bayes-roadmap.md` — lộ trình 3 model.
- Kiến trúc tham chiếu đã chứng minh: `RAFF/RARF-FHE-.../docs/realtime_bayes/phase1_plan.md`.

## 2. Việc đầu tiên: xác nhận 2 test đỏ đã xanh

Trên laptop, **2 test đỏ sẵn từ trước Phase 3**, cả hai do môi trường:

```
tests/test_saved_artifacts.py::test_saved_calibrator_load_and_bundle_distinction
    → FileNotFoundError: artifacts/models/evaluation_model.pt
tests/test_production_ensemble.py::test_predict_latest_matches_run_all_latest
    → NotImplementedError (StringDtype) tại inference.py:42, scaler pickle không khớp pandas 2.3.3
```

Trên PC, model có sẵn và môi trường khớp nên chúng **phải xanh**. Chạy:

```bash
python -m pytest -q          # kỳ vọng 57 passed, 0 failed
```

Nếu vẫn đỏ:

- thiếu file → tìm bằng `Get-ChildItem -Recurse -Filter *.pt` trong repo;
- lỗi `StringDtype` → so version: `pip show pandas scikit-learn` rồi đối chiếu với version đã
  pickle ra `artifacts/scalers/*`. Đây là lỗi tương thích artifact, **không** phải bug logic.
  Đừng "sửa" bằng cách ép kiểu ở `inference.py` khi chưa hiểu; nhiều khả năng chỉ cần đúng env.

**Không tin bất cứ số nào cho tới khi hai test này xanh.**

## 3. Đã có gì (đừng viết lại)

| Thành phần | File | Trạng thái |
| --- | --- | --- |
| Bayesian combination trên gate (Hedge) | `src/msdp/online/hedge.py` | ✅ 11 test |
| `gate_override` để posterior điều khiển mạng | `src/msdp/models/msdp.py` | ✅ 5 test, có gate identity |
| Bookkeeping đáo hạn (chống leak) | `src/msdp/online/{state,session}.py` | ✅ 8 test, mutation-checked |
| ACI / rolling CQR | `src/msdp/calibration.py` | ✅ **đã có sẵn từ trước** (`RollingCQRCalibrator`, `AdaptiveConformalCalibrator`) |
| Inference-only latest | `scripts/predict_latest.py` | ✅ đã có sẵn từ trước |

## 4. Còn lại, theo thứ tự

### Bước 1 — nối online vào `predict_latest_ensemble`

Trong `src/msdp/inference.py`:

- lấy `gate_prior` từ output mạng, tính `posterior = state.hedge.posterior_matrix(prior)`, rồi
  gọi lại model với `gate_override=posterior` (API đã có);
- thay `StaticCQRCalibrator` bằng `AdaptiveConformalCalibrator` khi có online state, giữ static
  làm mặc định để hành vi batch không đổi;
- sau khi publish, gọi `session.record_forecast(...)` cho mỗi horizon với
  `aux_return_median[0, j]` (dự báo riêng từng expert) và cận dưới/trên đã hiệu chỉnh.

Cần production bundle thật nên chỉ làm được trên PC.

### Bước 2 — persistence + CLI

MSDP **không có** entry point CLI trong `pyproject.toml`; mọi thứ chạy qua `scripts/*.py`. Thêm
`scripts/init_online_state.py` và `scripts/update_latest.py` theo đúng hình dạng của RAFF
(`init-online-state` → `update-latest`, idempotent, ghi manifest có checksum + `source_run_metadata`).

`OnlineState` đã có `to_dict`/`from_dict`/`manifest` nên chỉ cần chọn chỗ lưu, ví dụ
`artifacts/online_state/`.

### Bước 3 — nguồn dữ liệu read-only

Port `src/vnindex_model/data_source.py` của RAFF (interface `MarketDataSource`, connector SQLite/
DuckDB read-only) và cả `source_discovery.py` (`discover-source` dò schema, in ra block config
dán thẳng được). RAFF và MSDP dùng **chung** chuỗi VN-Index nên cấu hình sẽ giống hệt.

### Bước 4 — chọn `eta`

`HedgeGateState` mặc định `eta=0.5`, **chưa tuning**. Đây là tham số mới duy nhất Phase 3 thêm.
Chọn bằng cách replay lịch sử trên validation (grid nhỏ, ví dụ 0.1/0.25/0.5/1.0/2.0), so tổn thất
tích lũy của tổ hợp Hedge với gate tĩnh. **Đừng đoán.**

## 5. Ràng buộc không được phá

1. **Mạng neural không train online.** Chỉ gate combination và conformal học theo phiên. Train
   lại toàn mạng ở tầng batch, chu kỳ dài (một run ~6000 giây → cân nhắc hàng tháng).
2. **Không leak.** Forecast chỉ được chấm điểm khi horizon thực sự trôi qua trong chuỗi giá.
   Test `test_a_pending_forecast_never_scores_itself` giữ điều này.
3. `gate_override=None` phải giữ hành vi cũ **y hệt** — test identity đang bảo vệ.
4. Đơn vị: return quantiles của mạng ở **phần trăm** (`inference.py` chiếu bằng `exp(q/100)`).
   `realized_return_percent` đã theo đúng đơn vị đó; đừng trộn lẫn.
5. DB read-only tuyệt đối.
6. Khi sửa test, **mutation-check**: cố ý làm hỏng code và xác nhận test fail. Cách này đã bắt
   được bug thật ở cả ba repo trong dự án này.
