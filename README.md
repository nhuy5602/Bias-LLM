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

Trần lượt CLI/thời gian không phải trần tiền tuyệt đối: mỗi CLI có thể gọi API nhiều lần do tools hoặc retry nội bộ. Claude có thêm giới hạn native $0.80/lượt, nhưng có thể vượt nhẹ do request đang chạy và giá proxy có thể khác. Không thể suy ra hóa đơn Gemini/Codex từ số lượt CLI. Muốn ngân sách tiền tuyệt đối cần quota phía nhà cung cấp. Prompt yêu cầu tối đa 1200 từ, đây là giới hạn mềm.

## Lệnh khác

    .\run-collab.ps1 -DryRun
    .\run-collab.ps1 -Status
    .\run-collab.ps1 -NewRun
    .\run-collab.ps1 -NewRun -MaxRounds 1 -MaxCalls 6 -MaxSeconds 600
    python -m unittest discover -s tests -v

-NewRun là quyết định cấp ngân sách mới. Nếu sửa PDF, giới hạn hoặc controller, không tái sử dụng kết quả cũ một cách ngầm định. OS lock ngăn chạy hai controller đồng thời. Không kill cứng controller; dùng Ctrl+C để dọn cây tiến trình.

## Kết quả

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

