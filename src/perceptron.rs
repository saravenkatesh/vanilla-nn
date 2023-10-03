use nalgebra::{SVector, SMatrix};

fn threshold(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0;
    }
    return 0.0;
}

struct Perceptron<const X: usize, const N: usize> {
    weights: SVector<f64, X>,
    bias: f64,
}

impl<const X: usize, const N: usize> Perceptron<X, N> {
    fn one_step(
        &mut self, 
        inputs: SMatrix<f64, X, N>, 
        outputs: SVector<f64, N>,
        step: f64, 
    ) {
        let mut new_weights: Vec<f64> = Vec::new(); 
        for i in 0..X {
            let mut new_weight: f64 = self.weights[i];
            for (j, d) in inputs.column_iter().enumerate() {
                new_weight -= step * (threshold(self.weights.dot(&d) + self.bias) - outputs[j]) * d[i];

            }
            new_weights.push(new_weight);
        }
        self.weights = SVector::<f64, X>::from_vec(new_weights);
    }
}

impl<const X: usize, const N: usize> Perceptron<X, N> {
    fn gradient_descent(
        &mut self, 
        num_steps: i32,
        inputs: SMatrix<f64, X, N>, 
        outputs: SVector<f64, N>,
        step: f64, 
    ) {
        for _ in 0..num_steps {
            self.one_step(
                inputs, 
                outputs,
                step, 
            )
        }
    }
}

fn main() {
    // Initialize perceptron
    let mut perceptron = Perceptron{
        weights: SVector::<f64, 2>::new(0.0, 1.0),
        bias: 0.,
    };
    
    // Train perceptron
    let inputs = SMatrix::<f64, 2, 4>::new(
        1.0, 2.0, 0.0, 1.0,
        -1.0, 0.0, 2.0, 1.0,
    );
    let outputs = SVector::<f64, 4>::new(0.0, 1.0, 1.0, 1.0);
    let step = 0.1;
    let num_steps = 10;
    
    for _ in 0..num_steps {
        perceptron.one_step(inputs, outputs, step);
        println!("{}", perceptron.weights);
    }
}




