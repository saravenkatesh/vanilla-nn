use nalgebra::{DVector, DMatrix, dvector, dmatrix};
use vanilla_nn::FeedForward;

fn main() {
    // A (2, 2, 1) - network
    let mut feed_forward = FeedForward{
        weights: vec!(
            dmatrix![
                3.0, 0.5;
                1.0, 0.2;
            ],
            dmatrix![0.1, 0.0;
                0.2, 0.2],
            dmatrix![0.3, 0.1],
        ),
        bias: vec!(
            dvector!(1.0, 0.3),
            dvector!(0.3, 0.5),
            dvector!(0.2),
        ),
        size: 3,
    };
    
    let input: Vec<DVector<f64>> = vec![dvector![1.0, -1.0]];
    let output: Vec<DVector<f64>> = vec![dvector![2.0]];
    feed_forward.gradient(input, output, 0.5);
    println!("{:?}, {:?}", feed_forward.weights, feed_forward.bias)
}
