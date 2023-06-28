
import pickle

file_name = 'embedding_model.h5'
loaded_model = pickle.load(open(file_name, "rb"))

def model_predict(Buoi = 'Sáng', Cam_xuc = 'Vui', Do_tuoi = 10, So_nguoi = 1, Thoi_tiet = 'Mát',
                Phong_cach_am_thuc = 'Việt nam', Loai_hinh_quan_an = 'Nhà Hàng Sang Trọng',
                Che_do_an = 'ăn thoải mái', Dac_biet = 'Có chỗ để xe', Do_pho_bien = 'hot trend'):

    Buoi = Buoi.lower()
    Cam_xuc = Cam_xuc.lower()
    Thoi_tiet = Thoi_tiet.lower()
    Phong_cach_am_thuc = Phong_cach_am_thuc.lower()
    Loai_hinh_quan_an = Loai_hinh_quan_an.lower()
    Che_do_an = Che_do_an.lower()
    Dac_biet = Dac_biet.lower()
    Do_pho_bien = Do_pho_bien.lower()
     
    if Buoi == 'sáng':
        Buoi = 0
    elif Buoi == 'trưa':
        Buoi = 1
    elif Buoi == 'chiều':
        Buoi = 2
    elif Buoi == 'tối':
        Buoi = 3
    else:
        Buoi = 4

    if Cam_xuc == 'vui':
        Cam_xuc = 0
    elif Cam_xuc == 'buồn':
        Cam_xuc = 1
    elif Cam_xuc == 'khó chịu':
        Cam_xuc = 2
    else:
        Cam_xuc = 3
    
    if Thoi_tiet == 'lạnh' or Thoi_tiet == 'hơi lạnh' or Thoi_tiet == 'qúa lạnh' or Thoi_tiet == 'quá lạnh':
        Thoi_tiet = 0
    elif Thoi_tiet == 'nóng' or Thoi_tiet == 'hơi nóng' or Thoi_tiet == 'quá nóng':
        Thoi_tiet = 1
    else:
        Thoi_tiet = 2
   
    if Phong_cach_am_thuc == 'trung quốc':
        Phong_cach_am_thuc = 0
    elif Phong_cach_am_thuc == 'hàn quốc':
        Phong_cach_am_thuc = 1
    elif Phong_cach_am_thuc == 'món cho trẻ em':
        Phong_cach_am_thuc = 2
    elif Phong_cach_am_thuc == 'đồ ăn vặt':
        Phong_cach_am_thuc = 3
    elif Phong_cach_am_thuc == 'miền nam':
        Phong_cach_am_thuc = 4
    else:
        Phong_cach_am_thuc = 5
    
    if Loai_hinh_quan_an == 'quán vỉa hè' or Loai_hinh_quan_an == 'quán vỉa hè':
        Loai_hinh_quan_an = 0
    elif Loai_hinh_quan_an == 'nhà hàng bình dân':
        Loai_hinh_quan_an = 1
    else:
        Loai_hinh_quan_an = 2

    if Che_do_an == 'ăn kiêng':
        Che_do_an = 0
    elif Che_do_an == 'ăn chay':
        Che_do_an = 1
    else:
        Che_do_an = 2
    
    if Dac_biet == 'có chỗ để xe':
        Dac_biet = 0
    elif Dac_biet == 'có phòng trông trẻ':
        Dac_biet = 1
    elif Dac_biet == 'có phòng hút thuốc':
        Dac_biet = 2
    elif Dac_biet == 'đang giảm giá':
        Dac_biet = 3
    elif Dac_biet == 'có không gian ngoài trời':
        Dac_biet = 4
    elif Dac_biet == 'wifi miễn phí':
        Dac_biet = 5
    else:
        Dac_biet = 6
    
    if Do_pho_bien == 'hot trend':
        Do_pho_bien = 0
    elif Do_pho_bien == 'đặc sản':
        Do_pho_bien = 1
    elif Do_pho_bien == 'bestseller':
        Do_pho_bien = 2
    else:
        Do_pho_bien = 3
    list_of_input = [[Buoi, Cam_xuc, Do_tuoi, So_nguoi, Thoi_tiet, Phong_cach_am_thuc, Loai_hinh_quan_an, Che_do_an, Dac_biet, Do_pho_bien]]


    return loaded_model.predict(list_of_input)

def load_predict(model_predict_result):
    if model_predict_result[0] == 0:
        return 'món nước'
    elif model_predict_result[0] == 1:
        return 'món lạnh'
    elif model_predict_result[0] == 2:
        return 'món tráng miệng'
    elif model_predict_result[0] == 3:
        return 'đồ ăn vặt'
    elif model_predict_result[0] == 4:
        return 'Món cho trẻ em'
    else:
        return 'bánh tráng trộn'

