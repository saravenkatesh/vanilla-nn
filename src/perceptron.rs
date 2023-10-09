use nalgebra::{SVector, SMatrix};

fn threshold(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0;
    }
    return 0.0;
}

// X = number of weights
// N = number of data points in the training set
// TODO: Decouple the training set
pub struct Perceptron<const X: usize, const N: usize> {
    pub weights: SVector<f64, X>,
    pub bias: f64,
}

impl<const X: usize, const N: usize> Perceptron<X, N> {
    fn one_step(
        &mut self, 
        inputs: SMatrix<f64, X, N>, 
        outputs: SVector<f64, N>,
        step: f64, 
    ) {
        // Update weights
        let mut new_weights: Vec<f64> = Vec::new();
        for i in 0..X {
            let mut new_weight: f64 = self.weights[i];
            for (j, d) in inputs.column_iter().enumerate() {
                new_weight -= step * (threshold(self.weights.dot(&d) + self.bias) - outputs[j]) * d[i];
            }
            new_weights.push(new_weight);
        }
        self.weights = SVector::<f64, X>::from_vec(new_weights);
        
        // Update bias
        for (j, d) in inputs.column_iter().enumerate() {
            self.bias -=step * (threshold(self.weights.dot(&d) + self.bias) - outputs[j])   
        }
    }

    pub fn gradient_descent(
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
            );
        }
    }
    
    pub fn classifier(&self, input: SVector<f64, X>) -> f64 {
        threshold(self.weights.dot(&input) + self.bias)
    }
}
