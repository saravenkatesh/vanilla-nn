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
    let mut perceptron = Perceptron{
        weights: SVector::<f64, 2>::new(0.0, 1.0),
        bias: 3.,
    };
    
    perceptron.gradient_descent(
        NUM_STEPS,
        INPUTS,
        OUTPUTS,
        STEPS,
    );

    // TODO: Train perceptron on input data, and
    // ask for an n-dimensional vector
    while true {
        println!("Enter a 2 dimensional vector");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");
        let sequence: Vec<f64> = 
            input.split(",")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap())
                .collect();
        let data: SVector<f64, 2> = SVector::<f64, 2>::from_vec(sequence);
        
        let classification: f64 = perceptron.classifier(data);
        println!("{}", classification);
    }
}


