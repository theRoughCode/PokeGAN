WIDTH = 64;
HEIGHT = 64;
const modelURL = 'https://raw.githubusercontent.com/theRoughCode/PokeGAN/master/models/dcgan64/model.json';
const weightsURLPrefix = 'https://github.com/theRoughCode/PokeGAN/tree/master/models/dcgan64';
const modelIndexDbUrl = "indexeddb://pokegan-model:v2"; // include v1 for cache versioning

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
    const model = await tf.loadLayersModel(modelIndexDbUrl);
    console.log("Model loaded from IndexedDB");

    return model;
  } catch (error) {
    console.log(error);
    // If local load fails, get it from the server
    try {
      const model = await tf.loadLayersModel(modelURL, { strict: true });
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

const predict = (model) => {
  const input = tf.randomUniform([1, 100]);
  const img = getImage(model, input);
  const flattened = flatten(img);
  loadImageData(flattened);
};
