# Lộ trình real-time Bayesian update cho 3 model

Kiến trúc chung (đã thống nhất): mỗi model có 1 **tầng online** (cập nhật state/trọng số vài
giây mỗi khi có phiên mới, dùng recursive/Bayesian update, không refit) và 1 **tầng batch**
(refit định kỳ như hiện tại, mỗi lần refit sẽ reset lại tầng online).

## Phase 1 — RAFF (đang làm)

Prompt chi tiết: [`RAFF/RARF-FHE-Regime-Aware-Random-Forest-Forecasting-Hybrid-Engine/docs/realtime_bayes/phase1_raff_prompt.md`](RAFF/RARF-FHE-Regime-Aware-Random-Forest-Forecasting-Hybrid-Engine/docs/realtime_bayes/phase1_raff_prompt.md)

- HMM: 1 bước forward-filter (`forward_filter_step`).
- EGARCH: 1 bước recursive log-variance update (`egarch_step`).
- Random Forest: không đổi, chỉ inference với regime posterior mới qua `predict_soft_gated`
  (bản chất đã là Bayesian model averaging theo regime).
- Conformal: nuôi score pool online + Adaptive Conformal Inference (ACI) cho interval.
- Việc mới hoàn toàn: kết nối đọc read-only tới DB trên VPS (RAFF hiện chỉ đọc CSV tĩnh).

Chọn RAFF làm phase 1 vì HMM/EGARCH đã có sẵn cấu trúc recursive trong code — rủi ro thấp nhất
để chứng minh kiến trúc trước khi làm 2 model còn lại.

## Phase 2 — Dynamic Graph (sau khi Phase 1 xong và chạy ổn)

Ý tưởng cốt lõi (sẽ viết prompt chi tiết riêng khi tới lượt):

- Graphical Lasso hiện refit toàn bộ mỗi lần (`src/dynamicgraph/graphs/graphical_lasso.py`,
  gọi từ `scripts/build_graphs.py`/`scripts/generate_latest.py`). Tầng online sẽ thay bằng
  recursive/EWMA covariance update (hoặc Bayesian Normal-Inverse-Wishart conjugate update) rồi
  tính lại `adjacency_raw`/`adjacency_inference`/`adjacency_display` từ precision matrix mới —
  rẻ vì community detection/node roles/break score đều là hàm của graph, không phải của dữ liệu
  thô.
- Structural break score (`network/stress_score.py` và tương đương) đã dùng expanding
  z-score/percentile theo đúng kiểu "chỉ dùng lịch sử trước ngày hiện tại" — gần như đã sẵn sàng
  cho online, chỉ cần state hóa mean/std thay vì tính lại từ đầu.
- Việc chọn lại `alpha` (penalty) cho Graphical Lasso, CV, và train lại model dự báo stress
  (`training/walk_forward.py`, `fit_final_model`) vẫn ở tầng batch.
- Dynamic Graph đã có sẵn kết nối DB read-only (SQLite/DuckDB) — tái dùng cho watcher, không
  cần viết mới như RAFF.

## Phase 3 — MSDP (sau khi Phase 2 xong)

Ý tưởng cốt lõi:

- Mạng neural (`src/msdp/models/msdp.py`, experts trong `models/experts.py`) **không** train
  online — quá rủi ro overfit/trôi model với 1 điểm dữ liệu/ngày.
- Trọng số gate (`models/gate.py`) cập nhật online bằng Bayesian model combination kiểu
  multiplicative-weights/Hedge: mỗi phiên mới, tăng/giảm trọng số mỗi expert theo predictive
  log-loss gần đây của chính expert đó — về mặt toán tương đương cập nhật posterior Bayes cho
  bài toán chọn model.
- Conformal (CQR, `src/msdp/calibration.py`) dùng lại đúng cơ chế ACI như Phase 1.
- Việc train lại toàn mạng (Optuna/GPU, `scripts/tune.py`/`scripts/train.py`) vẫn ở tầng batch,
  ít thường xuyên hơn (mạng train ~6000 giây/run nên không hợp để refit hàng tuần như RAFF/Graph
  — cân nhắc chu kỳ dài hơn, ví dụ hàng tháng).

## Bộ điều phối chung (làm sau khi cả 3 phase xong, hoặc song song nếu cần sớm)

Một service nhỏ poll DB trên VPS, khi thấy phiên mới thì gọi lần lượt `update-latest` (hoặc lệnh
tương đương) của cả 3 model, không cần chờ nhau (độc lập). Không cần daemon Python riêng —
Windows Task Scheduler/cron gọi CLI là đủ, đúng triết lý hiện tại của cả 3 repo.
