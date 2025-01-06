import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.transform import resize

# Пути к файлам
prediction_path = "trunet_segmentation.nii.gz"
label_path = "ct_train_1001_label.nii.gz"

# Загрузка файлов с помощью nibabel
nifti_prediction = nib.load(prediction_path)
nifti_label = nib.load(label_path)

# Преобразование данных в NumPy массив
data_prediction = nifti_prediction.get_fdata()
data_label = nifti_label.get_fdata()

print(data_prediction.shape, data_label.shape)

data_label = resize(data_label, data_prediction.shape, anti_aliasing=False, order=0)

# Транспонирование для корректного отображения срезов
data_prediction = np.swapaxes(data_prediction, 0, 2)
data_label = np.swapaxes(data_label, 0, 2)

# Масштабируем данные для корректного отображения (если нужно)
data_min, data_max = np.min(data_prediction), np.max(data_prediction)
if data_max > 1:
    data_prediction = (data_prediction - data_min) / (data_max - data_min)

# Начальный индекс среза
slice_index = 0

# Создание фигуры и начальных изображений
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.2)  # Добавляем место для слайдера

# Отображение начального среза
image1 = ax1.imshow(data_label[slice_index], vmin=0, vmax=np.max(data_label))
ax1.set_title(f"Label: Slice {slice_index}")

image2 = ax2.imshow(data_prediction[slice_index], vmin=0, vmax=1)
ax2.set_title(f"Prediction: Slice {slice_index}")

# Добавление слайдера
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor="lightgoldenrodyellow")
slider = Slider(ax_slider, "Slice", 0, data_prediction.shape[0] - 1, valinit=slice_index, valstep=1)

# Функция обновления среза
def update(val):
    slice_idx = int(slider.val)  # Получаем текущее значение слайдера
    image1.set_data(data_label[slice_idx])  # Обновляем изображение для label
    ax1.set_title(f"Label: Slice {slice_idx}")
    image2.set_data(data_prediction[slice_idx])  # Обновляем изображение для prediction
    ax2.set_title(f"Prediction: Slice {slice_idx}")
    fig.canvas.draw_idle()  # Перерисовываем фигуру

# Связываем слайдер с функцией обновления
slider.on_changed(update)

plt.show()
