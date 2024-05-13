import cv2

# Đọc hình ảnh
image = cv2.imread(r"C:\Scrape\Untitled.jpeg")
height, width = image.shape[:2]
# Tính toán kích thước mới cho phần của hình ảnh bạn muốn giữ lại
left_right_crop = int(0.2 * width)  # 20% từ mỗi phía bên trái và bên phải
bottom_crop = int(0.3 * height)  # 30% từ dưới lên

# Cắt phần của hình ảnh bạn muốn giữ lại
cropped_image = image[bottom_crop:, left_right_crop:(width - left_right_crop)]

# Lấy kích thước mới của hình ảnh cắt
cropped_height, cropped_width = cropped_image.shape[:2]

# Mở tệp văn bản để ghi dữ liệu
with open(r"C:\Scrape\pixel_colors.txt", "w") as file:
    # Duyệt qua từng pixel trong phần của hình ảnh cắt và ghi giá trị màu RGB vào tệp
    for y in range(cropped_height):
        for x in range(cropped_width):
            # Lấy giá trị màu RGB của pixel tại tọa độ (x, y)
            blue, green, red = cropped_image[y, x]

            # Ghi giá trị màu của pixel vào tệp văn bản
            file.write("Pixel at (x={}, y={}): R={}, G={}, B={}\n".format(x, y, red, green, blue))

print("File 'pixel_colors.txt' has been created.")