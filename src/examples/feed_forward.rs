use std::io;
use nalgebra::{dmatrix, dvector, DVector};

use vanilla_nn::FeedForward;

pub fn feed_forward() {
    // A (2, 2, 1) - network
    let mut feed_forward = FeedForward{
        weights: vec!(
            dmatrix![
                3.0, 0.5;
                -1.0, 0.2;
            ],
            dmatrix![0.1, -5.0;
                0.2, 0.2],
            dmatrix![-3.0, 0.1],
        ),
        bias: vec!(
            dvector!(1.0, 0.3),
            dvector!(0.3, 0.5),
            dvector!(0.2),
        ),
        size: 3,
    };
    
    let input: Vec<DVector<f64>> = vec![dvector![1.0, -1.0], dvector![4.0, 10.0]];
    let output: Vec<DVector<f64>> = vec![dvector![2.0], dvector![-4.0]];
    feed_forward.gradient(input, output, 0.5);
 
    loop {
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
        let data: DVector<f64> = DVector::<f64>::from_vec(sequence);
        
        let classification: DVector<f64> = feed_forward.classify(data);
        println!("{}", classification);
    }   
}