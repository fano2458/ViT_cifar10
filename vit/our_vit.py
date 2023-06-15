import typing
import warnings
import tensorflow as tf
import typing_extensions as tx

from . import layers, utils


ImageSizeArg = typing.Union[typing.Tuple[int, int], int]


def interpret_image_size(image_size_arg):
	"""Process the image_size argument whether a tuple or int."""
	if isinstance(image_size_arg, int):
		return (image_size_arg, image_size_arg)
	if (
		isinstance(image_size_arg, tuple)
		and len(image_size_arg) == 2
		and all(map(lambda v: isinstance(v, int), image_size_arg))
	):
		return image_size_arg
	raise ValueError(
		f"The image_size argument must be a tuple of 2 integers or a single integer. Received: {image_size_arg}"
	)


def ViT(image_size,patch_size,num_classes,hidden_size,num_layers,
		num_heads,mlp_dim,dropout,emb_dropout,pool=False):
	image_size_tuple = interpret_image_size(image_size)
	assert (image_size_tuple[0] % patch_size == 0) and (
		image_size_tuple[1] % patch_size == 0
	), "image_size must be a multiple of patch_size"
	inputs = tf.keras.layers.Input(shape=(32, 32, 3))
	x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image,(image_size,image_size)))(inputs)
	y = tf.keras.layers.Conv2D(
		filters=hidden_size,
		kernel_size=patch_size,
		strides=patch_size,
		padding="valid",
		name="embedding",
	)(x)
	y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
	y = layers.ClassToken(name="class_token")(y)
	y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
	for n in range(num_layers):
		y, _ = layers.TransformerBlock(
			num_heads=num_heads,
			mlp_dim=mlp_dim,
			dropout=dropout,
			name=f"Transformer/encoderblock_{n}",)(y)
	y = tf.keras.layers.LayerNormalization(
		epsilon=1e-6, name="Transformer/encoder_norm")(y)
	if pool:
		y = tf.keras.layers.Lambda(lambda v: tf.reduce_mean(x,axis=1))(y) # TODO
	else:
		y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y) 
	y = tf.keras.layers.Dense(num_classes, name="head")(y)
	return tf.keras.models.Model(inputs=x, outputs=y, name="ViT")
