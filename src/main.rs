use std::io;
use nalgebra::{SVector, SMatrix};
use crate::perceptron::Perceptron;
mod perceptron;


const INPUTS: SMatrix<f64, 2, 4> = SMatrix::<f64, 2, 4>::new(
    1.0, 2.0, 0.0, 1.0,
    -1.0, 0.0, 2.0, 1.0,
);
const OUTPUTS: SVector<f64, 4> = SVector::<f64, 4>::new(0.0, 1.0, 1.0, 1.0);
const STEPS: f64 = 0.1;
const NUM_STEPS: i32 = 10;


fn main() {
    let feed_forward = FeedForward{
        weights: vec!(
            dmatrix![
                0.0, 0.0;
                0.0, 0.0;
            ],
            dmatrix![0.0, 0.0],
        ),
        bias: vec!(
            dvector!(0.0, 0.0),
            dvector!(0.0),
        ),
        size: 2,
    };
}


