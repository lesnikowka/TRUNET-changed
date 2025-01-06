import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Путь к файлу
prediction_path = "trunet_segmentation.nii.gz"

# Загрузка файла с помощью nibabel
nifti = nib.load(prediction_path)

# Преобразование данных в NumPy массив
data = nifti.get_fdata()

# Транспонирование для корректного отображения срезов
data = np.swapaxes(data, 0, 2)

# Масштабируем данные для корректного отображения (если нужно)
data_min, data_max = np.min(data), np.max(data)
if data_max > 1:
    data = (data - data_min) / (data_max - data_min)  # Нормализация к [0, 1]

# Начальный индекс среза
slice_index = 0

# Создание фигуры и отображение начального среза
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # Добавляем место для слайдера
image = ax.imshow(data[slice_index], cmap="gray", vmin=0, vmax=1)
ax.set_title(f"Slice {slice_index}")

# Добавление слайдера
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor="lightgoldenrodyellow")
slider = Slider(ax_slider, "Slice", 0, data.shape[0] - 1, valinit=slice_index, valstep=1)

# Функция обновления среза
def update(val):
    slice_idx = int(slider.val)  # Получаем текущее значение слайдера
    image.set_data(data[slice_idx])  # Обновляем изображение
    ax.set_title(f"Slice {slice_idx}")  # Обновляем заголовок
    fig.canvas.draw_idle()  # Перерисовываем фигуру

# Связываем слайдер с функцией обновления
slider.on_changed(update)

plt.show()
