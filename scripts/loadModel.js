WIDTH = 16;
HEIGHT = 16;
const modelName = 'proggan_16x16'
const modelURL = `https://raw.githubusercontent.com/theRoughCode/PokeGAN/master/models/${modelName}/model.json`;
const weightsURLPrefix = `https://github.com/theRoughCode/PokeGAN/tree/master/models/${modelName}`;
const modelIndexDbUrl = "indexeddb://pokegan-model:v1"; // include version for cache versioning

const getImage = (model, input) => {
  // Get output: values in [-1, 1]
  let output = model.predict(input);
  // Map tp [0, 1]
  output = output.add(1.0).div(2.0);
  // Map to [0, 255]
  output = output.mul(255.0).round();
  // Convert to array
  output = output.arraySync()[0];
  return output;
};

const fetchModel = async () => {
  console.log("Loading PokeGAN...");
  try {
    // Try loading locally saved model
    const model = await tf.loadGraphModel(modelIndexDbUrl, {
      strict: true,
      weightPathPrefix: weightsURLPrefix,
    });
    console.log("Model loaded from IndexedDB");

    return model;
  } catch (error) {
    console.log(error);
    // If local load fails, get it from the server
    try {
      const model = await tf.loadGraphModel(modelURL, {
        strict: true,
        weightPathPrefix: weightsURLPrefix,
      });
      console.log("Model loaded from HTTP.");

      // Store the downloaded model locally for future use
      await model.save(modelIndexDbUrl);
      console.log("Model saved to IndexedDB.");

      return model;
    } catch (error) {
      console.error(error);
    }
  }
};

const predict = (model, input=null) => {
  if (input == null) input = tf.randomUniform([1, 100]);
  const img = getImage(model, input);
  const flattened = flatten(img);
  loadImageData(flattened);
  return input;
};
