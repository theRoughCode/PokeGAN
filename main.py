import tensorflow as tf
from dcgan import DCGAN

batch_size = 64
img_dim = 28
num_channels = 3
num_epochs = 2000
save_interval = 200

def process_path(file_path):
  # Decode image as RGB
  image = tf.image.decode_png(tf.io.read_file(file_path), channels=num_channels)
  # some mapping to constant size - be careful with distorting aspect ratios
  image = tf.image.resize(image, (img_dim, img_dim))
  # color normalization to [-1, 1]
  image = (tf.cast(image, tf.float32) - 127.5) / 127.5
  return image

ds = tf.data.Dataset.list_files(str('pokemon/*.png'), shuffle=True).map(process_path)
model = DCGAN(ds, img_dim, num_channels, batch_size, model_dir='./dcgan28/models/', img_dir='./dcgan28/images/')
model.train(epochs=num_epochs, save_interval=save_interval)
