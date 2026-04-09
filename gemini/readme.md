# Hướng dẫn

## Mục tiêu
Bảo tính sẽ call API của Gemini-2.5-Flash về để inference trên tập test của Manual, FOLIO, ProofWriter, LogiQA, ReClor, RuleTaker, với phương pháp One-shot prompting. Mỗi tập mình chạy riêng để xem tỉ lệ predict đúng, sai là bao nhiêu, sau đó tổng hợp lại sau.

Hiện tại Bảo đã inference xong tập Manual và FOLIO, giờ còn 4 tập: ProofWriter, LogiQA, ReClor, RuleTaker. Mình sẽ phân ra như này:
- Khôi: LogiQA và RuleTaker 
- Bảo: ProofWriter và ReClor

## Cấu trúc thư mục
```
gemini/
├── results/            # Lưu kết quả inference trên tập test
│   ├── *.json
│   └── result.txt
├── inference.py        # File để call API của Gemini, để nó predict tập test  
├── test_data.json      # File chứa dataset test                   
├── predictions.json    # File ghi ra kết quả predict của Gemini
├── utils.ipynb         # File chứa vài tiện ích hữu dụng
├── .env                # Chứa API_KEY của Gemini (Khôi tạo API_KEY rồi điền vào đây)
└── readme.md  
```         

Trong thư mục results/, Khôi có thể thấy mấy file json là kết quả predict trên tập Manual và FOLIO Bảo đã chạy, và trong result.txt chứa tỉ lệ predict đúng, sai. Giờ Khôi cũng sẽ chạy trên tập LogiQA và RuleTaker để thêm 2 file json vào results/, và viết số lượng dự đoán đúng/sai trên tập đó vào results.txt như Bảo đã viết. 

Bảo đã code sẵn rồi, giờ Khôi thực hiện theo các bước Bảo ghi ở dưới.

## Các bước thực hiện
Trước hết, Khôi pip install google-genai. Sau đó Khôi làm theo các bước sau cho mỗi tập LogiQA và RuleTaker:
1. Khôi vào dán dataset test vào test_data.json.
2. Khôi vào utils.ipynb, chạy đoạn code Reformat để reformat file test_data.json về dạng đúng.
3. Xóa nội dung cũ (nếu có) trong file predictions.json trước khi chạy ở bước 4. Mình ko xóa file, chỉ xóa nội dung để file ko chứa gì thôi. Do Gemini chạy xong thì kết quả sẽ được "ghi tiếp" vào file chứ ko phải ghi lại từ đầu, nên làm vậy để file predictions.json được sạch.
4. Khôi chạy file inference.py, đợi nó chạy rồi sẽ ghi kết quả ra predictions.json. 
- Ở bước này, Gemini sẽ inference trên từng batch (Bảo mặc định batch_size = 150, để tiết kiệm quota), mỗi batch là mỗi lần gọi API. Mỗi lần chạy tốn tầm 2ph nên ráng đợi xíu nha.
- Lúc chạy có thể Khôi sẽ bị lỗi 503 UNAVAILABLE, chuyện này là bình thường. Cái này do Gemini đang quá tải (nhiều người gọi API). Khi bị vầy thì Khôi đợi tầm 10s xong chạy lại. 
    + Nếu bị lỗi này thì chương trình sẽ dừng tại batch hiện tại. 
    + Tin vui là khi bị lỗi này thì mình ko bị mất quota oan.
- Lưu ý, trong file predictions.json, Khôi có thể linh hoạt sửa dòng cuối (`run_large_scale_test(data, 150)`) thành `run_large_scale_test(data[start:end], 150)` với start và end là 2 sample đầu cuối Khôi muốn inference. 
5. Sau khi hoàn tất file predictions.json, Khôi vào utils.ipynb, chạy đoạn code **Compare predictions & True labels** để đếm bao nhiều predictions đúng/sai. 
6. Cuối cùng, Khôi thêm file *.json vào results/, và ghi kết quả số predictions đúng/sai vào results/

Khôi làm 6 bước trên cho 2 dataset test: LogiQA và RuleTaker.

## Bàn luận
Mỗi API key cho phép 20 call/ngày. Mỗi lần call API inference được 150 samples. 
=> Mỗi key được 3000 samples một ngày.
=> 5 key được 15000 samples, là đạt chỉ tiêu. (hoặc mình dùng 3 key cho 2 ngày)

Vậy nên quota không còn là vấn đề nữa :>