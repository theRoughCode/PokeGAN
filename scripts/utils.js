const flatten = (arr) => {
  const res = [];
  for (let x = 0; x < arr.length; x++) {
    const row = arr[x];
    for (let y = 0; y < row.length; y++) {
      const rgb = row[y];
      res.push(...rgb);
      res.push(255);
    }
  }
  return res;
};

const loadImageData = (rgbArr) => {
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(WIDTH, HEIGHT);
  for (let i = 0; i < rgbArr.length; i++) {
    imageData.data[i] = rgbArr[i];
  }
  ctx.putImageData(imageData, 0, 0);
  const imgEl = document.getElementById("poke-img");
  imgEl.src = canvas.toDataURL();
  $('#info').fadeOut(400);
  setTimeout(() => $('#image-card').fadeIn(400), 400);
};

const numToAlpha = (num) => {
  if (num === 1) return "$";
  if (num >= 1) throw Error("Invalid number!");
  // Remove decimal point
  num = num.toString().slice(2);
  // Remove leading zeros
  let num2 = parseInt(num);
  // Count leading zeros
  const numZeros = num.length - num2.toString().length;
  let str = `${numZeros}_`;
  while (num2 > 0) {
    const rem = num2 % 36;
    const alpha = rem.toString(36);
    num2 = Math.floor(num2 / 36);
    str += alpha;
  }
  return str;
};

const alphaToNum = (str) => {
  if (str === "$") return 1;
  // Get leading zeros
  let [numZeros, numStr] = str.split("_");
  let num = 0;
  for (let i = numStr.length - 1; i >= 0; i--) {
    const alpha = numStr[i];
    const rem = parseInt(alpha, 36);
    num = num * 36 + rem;
  }
  // Add decimal point
  num = "0." + "0".repeat(numZeros) + num.toString();
  num = parseFloat(num);
  return num;
};

const encodeTensor = (tensor) => {
  let arr = tensor.dataSync();
  // Convert to alpabet
  arr = Array.from(arr).map(numToAlpha);
  const str = arr.join("-");
  return str;
};

const decodeStr = (str) => {
  let arr = str.split("-");
  arr = arr.map(alphaToNum);
  const tensor = tf.tensor(arr, [1, arr.length], "float32");
  return tensor;
};
