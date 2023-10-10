# Neural networks but with Rust

A small library for training and running vanilla neural nets.  To be used precisely when you want your basic learning algorithms to be kinda cool (written from scratch in Rust) and highly unoptomized (written from scratch in Rust).

This library utilizes the `nalgebra` crate for matrix operations.

## Setup 

To get started, 

    cargo run

will execute a very simple example neural net with 3 layers -- of sizes 2, 2, and 1 -- trained on a robust 2 examples.  You will be prompted to enter 2D vectors that you wish classified.  You can modify this example, including network architecture, network initializations, and training data, in `src/examples/feed_forward.rs`.

## Usage

There are two library components: a feed-forward neural net in `src/lib.rs` and a perceptron in `src/modules/perceptron.rs`.  

Initialize a neural network:

    use nalgebra::{dvector, dmatrix, DVector, DMatrix}

    let feed_forward = FeedForward(
        weights: weights,
        bias: bias,
        size: size,
    )

    let weights: Vec<DMatrix<f64>> = vec![dmatrix![ ... ], ..., dmatrix![...]];
    let bias: Vec<DVector<f64>> = vec![dvector![ ... ], ..., dvector![ ... ]];
    let size: usize = weights.len();

Train a neural network:

    feed_forward.gradient(inputs, outputs, step_size);

    let inputs: Vec<DVector<f64>> = vec![dvector![ ... ], ..., dvector![ ... ]]; 
    let outputs: Vec<DVector<f64>> = vec![dvector![ ... ], ..., dvector![ ... ]];
    let step_size: f64 = ... ;

Use a neural network:

    feed_forward.classify(data);

    let data: DVector<f64> = ... ;

Note that there is no error checking to make sure that the dimensions you enter for anything are compatible!  (TODO)

The perceptron can be used similarly.  It uses the static types from `SMatrix` and `SVector` instead of the dynamic vectors and matrices of the neural network.