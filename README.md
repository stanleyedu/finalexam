pythonAI-final
南華大學跨領域-人工智慧期末報告

組員
10924316
人工智慧內捲神經
簡介
卷積是大多數現代神經網路的基礎電腦視覺網路。
卷積核是空間不可知且特定於通道。正因為如此，它無法適應不同的視覺模式不同的空間位置。
除了位置相關的問題外，卷積的感受野為捕獲帶來了挑戰 長程空間相互作用。

為了解決上述問題。
重新考慮屬性的捲積對合：反轉視覺辨識中卷積的固有性。
提出了“對合核”，即特定位置和 與通路無關。由於操作的地點特定性，自註意力屬於以下設計範式內捲化。

這個例子描述了卷積核，比較兩個影像分類模型，一個有捲積，另一個有內卷，並嘗試與自註意力進行類比層。

設定
import os:
導入 Python 中的 os 模組，該模組用於與操作系統進行交互，這在後續可能用於設定環境變數等。

os.environ["KERAS_BACKEND"] = "tensorflow":
設定 Keras 的後端為 TensorFlow。Keras 是一個高階的深度學習框架，而 TensorFlow 是其中的一個後端引擎。這行程式碼確保使用 TensorFlow 作為 Keras 的後端。

import tensorflow as tf:
導入 TensorFlow 模組，這是一個用於構建和訓練深度學習模型的開源機器學習框架。

import keras:
導入 Keras 模組，這是一個高階的深度學習框架，通常與 TensorFlow 一起使用，用於快速構建深度學習模型。

import matplotlib.pyplot as plt:
導入 Matplotlib 模組的 pyplot 子模組，用於繪製圖表和視覺化資料。

tf.random.set_seed(42):
設定 TensorFlow 的隨機種子為 42，這樣可以確保在執行模型時獲得可重複的結果。這對於模型的可重複性和調試很有用。

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import matplotlib.pyplot as plt

# Set seed for reproducibility.
tf.random.set_seed(42)
卷積
卷積仍然是電腦視覺深度神經網路的支柱。 要理解對合，有必要談談 卷積運算。

MSKLsm5

考慮一個具有維度H、W和C_in的輸入張量X。我們使用一組C_out卷積核，每個核的形狀為K、K、C_in。透過輸入張量和卷積核之間的乘加操作，我們得到一個具有維度H、W、C_out的輸出張量Y。

在上面的圖表中，C_out=3。這使得輸出張量的形狀為H、W和3。可以注意到卷積核不依賴輸入張量的空間位置，這使其位置無關。另一方面，輸出張量中的每個通道都基於特定的捲積濾波器，這使其通道特定。

內卷化
這個想法是要設計一種既是位置特定又是通道無關的操作。嘗試實現這些特定的屬性是一個挑戰。
如果對於每個空間位置使用固定數量的involution核，我們將無法處理可變解析度的輸入張量。

為了解決這個問題，作者考慮了在特定空間位置條件下產生每個核。透過這種方法，我們應該能夠輕鬆處理可變解析度的輸入張量。
下面的圖表提供了關於這種核生成方法的直觀理解。

jtrGGQg

自定義的 Keras 層
這個自定義層實現了一種稱為 Involution 的卷積變種操作，它在卷積過程中引入了一些新的概念和操作。

class Involution(keras.layers.Layer):
    def __init__(
        self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output, kernel
測試內捲層
使用之前定義的 Involution 自定義層，並對一個隨機產生的輸入張量進行不同參數設置的 Involution 操作，並印出每個操作的輸出形狀。
創建輸入張量、進行 Involution 操作、印出輸出形狀

# Define the input tensor.
input_tensor = tf.random.normal((32, 256, 256, 3))

# Compute involution with stride 1.
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=1, reduction_ratio=1, name="inv_1"
)(input_tensor)
print(f"with stride 1 ouput shape: {output_tensor.shape}")

# Compute involution with stride 2.
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=2, reduction_ratio=1, name="inv_2"
)(input_tensor)
print(f"with stride 2 ouput shape: {output_tensor.shape}")

# Compute involution with stride 1, channel 16 and reduction ratio 2.
output_tensor, _ = Involution(
    channel=16, group_number=1, kernel_size=5, stride=1, reduction_ratio=2, name="inv_3"
)(input_tensor)
print(
    "with channel 16 and reduction ratio 2 ouput shape: {}".format(output_tensor.shape)
)
影像分類
在本節中，我們將建立一個影像分類器模型。這裡將 是兩個模型，一個有捲積，另一個有內捲化。
Get the CIFAR10 Dataset

# Load the CIFAR10 dataset.
print("loading the CIFAR10 dataset...")
(
    (train_images, train_labels),
    (
        test_images,
        test_labels,
    ),
) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1.
(train_images, test_images) = (train_images / 255.0, test_images / 255.0)

# Shuffle and batch the dataset.
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    .shuffle(256)
    .batch(256)
)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(256)
視覺化數據
通過 Matplotlib 視覺化了訓練集中的前25張圖片，每張圖片都標有對應的類別名稱。
這樣的視覺化有助於了解圖片數據的特點，並且在深度學習中經常用於檢查數據的品質和分布。

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
卷積神經網絡
# Build the conv model.
print("building the convolution model...")
conv_model = keras.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"),
        keras.layers.ReLU(name="relu1"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.ReLU(name="relu2"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), padding="same"),
        keras.layers.ReLU(name="relu3"),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10),
    ]
)

# Compile the mode with the necessary loss function and optimizer.
print("compiling the convolution model...")
conv_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model.
print("conv model training...")
conv_hist = conv_model.fit(train_ds, epochs=20, validation_data=test_ds)
內捲神經網絡
# Build the involution model.
print("building the involution model...")

inputs = keras.Input(shape=(32, 32, 3))
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_1"
)(inputs)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_2"
)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x, _ = Involution(
    channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_3"
)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation="relu")(x)
outputs = keras.layers.Dense(10)(x)

inv_model = keras.Model(inputs=[inputs], outputs=[outputs], name="inv_model")

# Compile the mode with the necessary loss function and optimizer.
print("compiling the involution model...")
inv_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# train the model
print("inv model training...")
inv_hist = inv_model.fit(train_ds, epochs=20, validation_data=test_ds)
參數
可以看到，CNN 中的參數具有類似的架構比INN（對合神經網路）大得多。

conv_model.summary()

inv_model.summary()
損失和準確度圖
這裡，損失和準確率圖顯示 INN 很慢 學習者（參數較低）。

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title("Convolution Loss")
plt.plot(conv_hist.history["loss"], label="loss")
plt.plot(conv_hist.history["val_loss"], label="val_loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Involution Loss")
plt.plot(inv_hist.history["loss"], label="loss")
plt.plot(inv_hist.history["val_loss"], label="val_loss")
plt.legend()

plt.show()

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.title("Convolution Accuracy")
plt.plot(conv_hist.history["accuracy"], label="accuracy")
plt.plot(conv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Involution Accuracy")
plt.plot(inv_hist.history["accuracy"], label="accuracy")
plt.plot(inv_hist.history["val_accuracy"], label="val_accuracy")
plt.legend()

plt.show()
可視化卷積核
為了可視化內核，我們取每個內核的 K×K 值總和 卷積核。 不同空間的所有代表 位置框出對應的熱圖。

「我們提出的內卷讓人想起自我關注和 本質上可以成為它的通用版本。」 透過內核的可視化，我們確實可以獲得關注 圖像的地圖。學習到的對合核提供了對 輸入張量的各個空間位置。這 位置特定屬性使對合成為模型的通用空間 自註意力屬於其中。

layer_names = ["inv_1", "inv_2", "inv_3"]
outputs = [inv_model.get_layer(name).output[1] for name in layer_names]
vis_model = keras.Model(inv_model.input, outputs)

fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(10, 30))

for ax, test_image in zip(axes, test_images[:10]):
    (inv1_kernel, inv2_kernel, inv3_kernel) = vis_model.predict(test_image[None, ...])
    inv1_kernel = tf.reduce_sum(inv1_kernel, axis=[-1, -2, -3])
    inv2_kernel = tf.reduce_sum(inv2_kernel, axis=[-1, -2, -3])
    inv3_kernel = tf.reduce_sum(inv3_kernel, axis=[-1, -2, -3])

    ax[0].imshow(keras.utils.array_to_img(test_image))
    ax[0].set_title("Input Image")

    ax[1].imshow(keras.utils.array_to_img(inv1_kernel[0, ..., None]))
    ax[1].set_title("Involution Kernel 1")

    ax[2].imshow(keras.utils.array_to_img(inv2_kernel[0, ..., None]))
    ax[2].set_title("Involution Kernel 2")

    ax[3].imshow(keras.utils.array_to_img(inv3_kernel[0, ..., None]))
    ax[3].set_title("Involution Kernel 3")
