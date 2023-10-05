use std::io;
use nalgebra::SVector;
use crate::perceptron::Perceptron;
mod perceptron;

fn main() {
    let mut perceptron = Perceptron{
        weights: SVector::<f64, 2>::new(0.0, 1.0),
        bias: 3.,
    };
    
    perceptron.train_perceptron();

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


