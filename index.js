function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  let s = sigmoid(x);
  return s * (1 - s);
}

let input1 = 0.1;
let input2 = 0.9;
let target = 1.0;
let lr = 0.1;

let w1 = Math.random();
let b1 = Math.random();

let w2 = Math.random();
let b2 = Math.random();

let w_out1 = Math.random();
let w_out2 = Math.random();
let b_out = Math.random();

for (let i = 0; i < 1000; i++) {
  let z1 = w1 * input1 + b1;
  let h1 = sigmoid(z1);

  let z2 = w2 * input2 + b2;
  let h2 = sigmoid(z2);

  let z_out = h1 * w_out1 + h2 * w_out2 + b_out;
  let output = sigmoid(z_out);

  let error = output - target;
  let dOutput = error * sigmoidDerivative(z_out);

  w_out1 -= lr * dOutput * h1;
  w_out2 -= lr * dOutput * h2;
  b_out  -= lr * dOutput;

  let dH1 = dOutput * w_out1 * sigmoidDerivative(z1);
  w1 -= lr * dH1 * input1;
  b1 -= lr * dH1;

  let dH2 = dOutput * w_out2 * sigmoidDerivative(z2);
  w2 -= lr * dH2 * input2;
  b2 -= lr * dH2;
}

let h1_final = sigmoid(w1 * input1 + b1);
let h2_final = sigmoid(w2 * input2 + b2);
let finalOutput = sigmoid(h1_final * w_out1 + h2_final * w_out2 + b_out);

console.log("Final Output:", finalOutput.toFixed(4));
