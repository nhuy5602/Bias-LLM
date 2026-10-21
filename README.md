# Nhóm nghiên cứu Bias VLM — Codex, Claude Code, Gemini

Chạy trong PowerShell tại thư mục này:

    .\run-collab.ps1

Bộ điều phối chạy local; các CLI gửi prompt tới dịch vụ model đã cấu hình (không phải inference offline). Có thể chạy từ thư mục khác bằng đường dẫn tuyệt đối tới script. Không lưu API key vào repository.

## Giới hạn mặc định

| Mục | Giới hạn |
|---|---|
| Vòng | 2: đề xuất độc lập, phản biện chéo + sửa đề xuất |
| Tổng hợp | 3 phần song song, mỗi thành viên sở hữu một phần |
| Song song | tối đa 3 thành viên |
| Retry | 3 lần sau lần gọi ban đầu, cho từng task lỗi |
| Tổng lượt khởi chạy CLI | 12, bao gồm retry |
| Timeout mỗi lượt | 240 giây |
| Tổng thời gian chạy | 1200 giây; lưu qua resume |
| Lỗi auth/quota/cấu hình | ngắt thành viên ngay, không retry vô ích |

Một thành viên hết retry bị vô hiệu hóa trong phiên. Thành viên khác tiếp tục; synthesis ghi rõ ai thiếu. Không tự đổi model, không mở thêm phiên và không gọi lại kết quả thành công. Lệnh mặc định resume phiên gần nhất nếu nguồn/cấu hình/code khớp. Ctrl+C dừng các tiến trình do bộ điều phối sở hữu và lưu tiến độ.

Lộ trình đầy đủ dùng 9 task (3 đề xuất + 3 phản biện + 3 tổng hợp), còn tối đa 3 lượt retry trong trần 12 lượt chung. Trần chung có thể dừng sớm trước khi một thành viên dùng đủ 3 retry. Claude phụ trách kiểm chứng/nhân quả; Gemini phụ trách novelty/generalization; Codex phụ trách pilot/implementation. Controller chỉ ghép các phần, không tự phong một model làm trọng tài.

Trần lượt CLI/thời gian không phải trần tiền tuyệt đối: mỗi CLI có thể gọi API nhiều lần do tools hoặc retry nội bộ. Claude có trần native tổng $2/run; từng request nhận phần ngân sách còn lại. Gemini không có USD cap theo yêu cầu người dùng, nhưng vẫn bị chặn theo số lượt/thời gian/retry. Codex CLI không báo chi phí; $0 reported trong ledger không có nghĩa là miễn phí. Prompt yêu cầu tối đa 1200 từ, đây là giới hạn mềm.

## Lệnh khác

    .\run-collab.ps1 -DryRun
    .\run-collab.ps1 -Status
    .\run-collab.ps1 -ContinueLatest -MaxCalls 12 -MemberMaxCalls 4 -MemberMaxSeconds 480 -MemberMaxUsd 2
    .\run-collab.ps1 -ExtendBudget -MaxRounds 1 -MaxCalls 12 -MemberMaxCalls 6 -MemberMaxSeconds 960 -MemberMaxUsd 2 -MaxSeconds 1200
    .\run-collab.ps1 -ExtendBudget -RepairGemini -MaxCalls 16 -MemberMaxCalls 8 -MemberMaxSeconds 1600 -MemberMaxUsd 2 -MaxSeconds 1800 -GeminiRepairTimeout 600
    .\run-collab.ps1 -ExtendBudget -UnlimitedBudget -TimeoutSeconds 600
    .\run-collab.ps1 -ContinueLatest -UnlimitedBudget -MaxRounds 1 -TimeoutSeconds 600
    .\run-collab.ps1 -ExtendBudget -UnlimitedBudget -ResumeRunId 20261012-012215-327254 -TimeoutSeconds 600
    .\run-series.ps1 -DryRun
    .\run-series.ps1
    .\run-semantic-label-pilot.ps1 -DryRun
    .\run-semantic-label-pilot.ps1
    .\run-semantic-label-pilot.ps1 -Status
    .\run-semantic-label-pilot.ps1 -RetryUnavailable -MaxNewLaunches 1
    .\run-collab.ps1 -NewRun
    .\run-collab.ps1 -NewRun -MaxRounds 1 -MaxCalls 6 -MaxSeconds 600
    .\run-collab.ps1 -Duo -NewRun -MaxRounds 1 -MaxCalls 6 -MaxSeconds 600
    .\run-collab.ps1 -Duo -ContinueLatest -MaxRounds 1 -MaxCalls 12
    python -m unittest discover -s tests -v
    python scripts/check_research_constraints.py .
    python scripts/logo_duo_pilot.py --include-mercedes --dry-run
    python scripts/logo_duo_pilot.py --run-dir <prepared-run-folder>
    python scripts/open_vlm_pilot.py --dry-run --run-dir <open-vlm-run-dir>
    ./run-open-vlm.sh smoke <open-vlm-run-dir>

`-Duo -NewRun` tạo một ledger mới chỉ với Codex và Gemini. Hai thành viên giữ
vai trò `pilot` và `evidence_gate`, phản biện chéo lẫn nhau; Claude không được
gọi, không xuất hiện trong task/report/synthesis, và ledger duo không tự tiếp
quản một run ba thành viên. Dùng `-Duo -ContinueLatest` để tạo ledger tiếp nối
một proposal round đã hoàn tất của duo.

## Open-VLM track

The `open-vlm` track is an additive Linux/GPU path for Qwen2.5-VL. It keeps
free-generation and candidate-scoring baselines separate, validates the
blur-as-prior assumption before contrastive decoding, and records image-token
parity, source-family splits, metrics, controls, and cleanup checks. The first
dry-run and fake-backend tests require no GPU or model download. Paid execution
must use `run-open-vlm.sh`, upload artifacts and verify checksums before pod
deletion, and remains scientifically `NEEDS_EVIDENCE`.
The pinned GPU requirements are availability-checked only; CUDA, attention
kernel, model license, and dataset license checks must pass on the target image
before any paid model call. A failed development gate redirects the remaining
budget to attention/logit-lens diagnostics instead of silently continuing.

-ContinueLatest lấy proposal đã xác minh của vòng gần nhất (3 ở full mode, 2 ở duo mode), chạy đúng một vòng phản biện/sửa đề xuất rồi tổng hợp. Nó tạo budget ledger mới, giữ PDF và role ownership, đồng thời từ chối nếu report/hash đầu vào không khớp. -ExtendBudget tiếp tục ledger mới nhất sau khi quota hết; chỉ cho giữ nguyên hoặc tăng thời gian/per-member-call ceilings, không giảm hoặc reset calls, retries, costs, đổi nguồn hay model. Mọi lần tăng thực tế được ghi trong manifest. Dùng -ResumeRunId cùng -ExtendBudget để tiếp tục một ledger cũ mà không đổi latest.json; controller vẫn kiểm tra source hash và report hash. -RepairGemini dùng prompt đã lưu để chạy riêng tác vụ Gemini đã hết retry, giữ nguyên lịch sử timeout và tạo synthesis reconciliation mới. -RepairGeminiSynthesis có thể chạy một tác vụ synthesis recovery riêng sau bốn timeout; lượt recovery có ledger riêng, không tăng số retry của task gốc. Timeout riêng mặc định 600 giây; task recovery vẫn tối đa 3 retry. -UnlimitedBudget chỉ dùng với -ExtendBudget hoặc -ContinueLatest; bỏ trần tổng/per-member calls, elapsed time và USD kể cả Claude native, nhưng giữ timeout từng request và tối đa 3 retry. Mọi launch, retry, elapsed time và chi phí CLI báo được vẫn ghi vào ledger. `run-series.ps1` tiếp tục tuần tự tới số vòng được yêu cầu, cập nhật series ledger sau mỗi run, và dừng ở trạng thái chưa hoàn tất. -NewRun tạo nghiên cứu độc lập từ đầu. Nếu sửa PDF, giới hạn hoặc controller, không tái sử dụng kết quả cũ một cách ngầm định. OS lock ngăn chạy hai controller đồng thời. Không kill cứng controller; dùng Ctrl+C để dọn cây tiến trình.

## Kết quả

`run-semantic-label-pilot.ps1` chạy pilot riêng trên 20 ảnh động vật counterfactual 384px (10 chim, 10 thú), hai điều kiện prompt (tên loài cụ thể hoặc "this animal"), mỗi điều kiện gọi model hai lần. Trần 80 model launches bao gồm retry; mỗi task tối đa 3 retry. Dataset lấy chọn lọc từ Hugging Face, lưu hash ảnh, prompt và từng attempt vào `collab/experiment/`. Pilot hiện dùng Claude Code CLI, model `claude-fable-5-1` qua proxy đã cấu hình trên máy (`gpt.teamsoclo.site`), chỉ bật Read tool và cap Claude native $2 cho run. Proxy đã báo tổng $2.3534 trước khi dừng; provider có thể vượt cap native. 23 Gemini launches trước đó vẫn nằm trong cùng ledger. Nếu provider trả lỗi quota, run dừng `DEGRADED`; chỉ dùng `-RetryUnavailable` sau khi quota được khôi phục. Đây là pilot mô tả trên ảnh counterfactual, không có original baseline và không xác nhận cơ chế nhân quả.

`python scripts/logo_duo_pilot.py` chạy exploratory replication nhỏ trên dataset VLMBias chính thức: 2 Audi source families, original/counterfactual ở 384px, cùng prompt trên Codex và Gemini, tối đa 8 launches. Kết quả hiện tại ở `collab/experiment/logo-duo-20261018-223746-146093/`: Codex hoàn tất 4/4; Gemini timeout 4/4; operational status `DEGRADED`, scientific status `NEEDS_EVIDENCE`. Đây chưa phải causal pilot: chưa có texture/edge-matched sham, hidden-state intervention, hay open-VLM comparison.

Pilot khái quát hóa thương hiệu thứ ba dùng `python scripts/logo_duo_pilot.py --include-mercedes --dry-run`, sau đó chạy Codex bằng `--run-dir` trên cùng manifest/prompt. Run hiện tại ở `collab/experiment/logo-duo-20261019-005644-359883/`: 14 families hoàn chỉnh (5 Audi, 5 Maserati, 4 Mercedes-Benz; Mercedes sedan bị loại vì thiếu original pair), 84 Codex tasks và 84 Gemini tasks. Codex theo prior trên 42/42 counterfactual; Gemini đếm đúng 42/42. Đây vẫn là behavioral generalization, không phải causal evidence; scientific status giữ `NEEDS_EVIDENCE`.

Pilot robustness dùng `python scripts/logo_robustness_pilot.py --dry-run`, sau đó
`python scripts/logo_robustness_pilot.py --run-dir <prepared-run-folder>`. Run
hiện tại ở `collab/experiment/logo-robustness-20261019-022124-977565/`: 6 fixed
families, original/counterfactual, max-edge `384/192/96`, 3 repeats, 108 Codex
tasks và 108 Gemini tasks. Cả hai model đều giữ pattern ở cả ba mức; phase
transition chưa quan sát được. Đây là robustness screen giới hạn bởi transform,
không phải causal evidence; scientific status giữ `NEEDS_EVIDENCE`.

Pilot Gaussian blur dùng `python scripts/logo_robustness_pilot.py --mode
gaussian-blur --dry-run`, sau đó chạy `--run-dir` trên prompt đã tạo. Run hiện
tại ở `collab/experiment/logo-robustness-20261019-231529-364544/`: sigma
`1/3/6`, Gemini `108/108` valid, Codex `gpt-6.1-sol` với `104/108` valid và 4
task missing do timeout/stream disconnect. Gemini cho thấy prior alignment tăng
từ `0/18` ở sigma 1 lên `12/18` ở sigma 3 và `18/18` ở sigma 6. Đây là kết quả
hành vi provisional, không phải causal evidence; scientific status giữ
`NEEDS_EVIDENCE`.

Protocol Gaussian blur cũ hiện được đánh dấu `PROVISIONAL-CONFOUNDED`: ảnh
original thường có kích thước nguồn lớn hơn ảnh counterfactual 384px, trong khi
`gaussian-blur` áp dụng blur mà không resize. Bản hiệu chỉnh dùng
`python scripts/logo_robustness_pilot.py --mode gaussian-blur-matched --dry-run`,
sau đó chạy `--run-dir` trên cùng thư mục. Mode này chuẩn hóa mọi source về
`max_edge=384` bằng EXIF transpose + RGB + LANCZOS trước khi áp dụng sigma
`1/3/6`, lưu riêng source/normalized/derived SHA256 và kích thước; đây là
protocol opt-in và không thay đổi các mode cũ.

Run matched hiện tại ở
`collab/experiment/logo-robustness-20261021-181603-868764/`: Codex hoàn tất
108/108 valid và Gemini hoàn tất 108/108 valid. Sau audit, source/normalized/
derived hashes và dimensions đều khớp (0 mismatch). Gemini cho thấy
counterfactual prior-aligned `0/18`, `12/18`, `18/18` tại sigma `1/3/6`, còn
Codex prior-aligned `18/18` ở cả ba sigma. Đây là bất đồng hành vi giữa hai
model, không phải transition đã được replication liên-model; scientific status
vẫn `NEEDS_EVIDENCE`.

Vòng tiếp theo dùng protocol matched `v2` tại
`collab/experiment/logo-robustness-20261021-215013-759070/`, thêm sigma `0` làm
baseline no-blur trong cùng manifest (48 variants, 144 task mỗi model). Ảnh
sigma-zero là bản copy byte-identical của ảnh normalized, nên hash normalized và
derived phải bằng nhau; các mode Gaussian blur cũ không bị thay đổi. Codex
prior-aligned `18/18` counterfactual ở cả sigma `0/1/3/6`; Gemini exact
`18/18`, `18/18`, `6/18`, `0/18` tương ứng. Đây là model disagreement sau
matched baseline, chưa phải replication liên-model; scientific status vẫn
`NEEDS_EVIDENCE`.

Pilot pixel-paired/sham-control corrected ở
`collab/experiment/logo-paired-sham-20261021-225532-353740/`: 6 families × 3
conditions × 3 repeats, cùng ảnh normalized cho mỗi family. Codex và Gemini đều
`54/54` valid; original/sham đạt `18/18` exact ở cả hai model. Logo-erased cho
Codex `4/18` exact, `14/18` prior-aligned; Gemini `18/18` exact. Hash, dimension,
ROI và task-ID audit đều `0` mismatch. Đây là synthetic behavioral control,
chưa phải bằng chứng nhân quả; scientific status vẫn `NEEDS_EVIDENCE`.

collab/latest.json trỏ đến collab/runs/<id>/. Trong đó:
- manifest.json: hash nguồn và giới hạn.
- inputs/paper.txt: văn bản toàn PDF; context.txt: trang chọn lọc gửi các agent.
- reports/: đề xuất, phản biện và tổng hợp.
- logs/: prompt, stdout/stderr và kết quả từng attempt.
- state.json: budget/counters/status để resume.
- SUMMARY.md: kết quả cuối và các vấn đề chưa giải quyết.

Các file chạy được gitignore vì có thể chứa nội dung nghiên cứu riêng. Không có kết quả thí nghiệm được tạo giả. Phiên kết thúc ở NEEDS_EVIDENCE về mặt khoa học; COMPLETED chỉ nghĩa là thảo luận đã chạy xong. Chưa có executor thí nghiệm hay bộ viết manuscript tự động trong phiên bản này.

## Skills và plugins

Native CLI tiếp tục đọc cấu hình tương ứng. Để giảm context/tool overhead, pha thảo luận mặc định tắt tool calls và MCP của Claude; dữ liệu được truyền trực tiếp. Codex dùng read-only sandbox và tắt MCP/desktop plugin cho lượt này; Gemini dùng plan mode. Các skill đã cài không bị xóa, nhưng khả năng gọi skill/tool trong chế độ tiết kiệm có giới hạn. Không dùng --bare hoặc bypass-permissions. Công cụ chỉ có trong Codex desktop không được đảm bảo có trong codex exec. Tự khai của model về skill/tool là chưa xác minh nếu không có log tương ứng.

Máy này cần python, pypdf, node và ba CLI đã cài; nếu đổi máy: python -m pip install -r requirements.txt, sau đó cấu hình các CLI riêng. Gemini dùng GEMINI_API_KEY/GOOGLE_API_KEY. Không in secrets trong preflight.

