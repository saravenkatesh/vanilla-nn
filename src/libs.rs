use nalgebra::{DVector, DMatrix, dmatrix, dvector};
use libm::exp;

fn main() {
    // A (2, 2, 1) - network
    let feed_forward = FeedForward{
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
    
    let input: DVector<f64> = dvector![1.0, -1.0];
    let output: DVector<f64> = dvector![2.0];
    println!("{:?}", feed_forward.update(input, output));
}

// TODO: replace for loops with maps
fn sigmoid(mut x: f64) -> f64 {
    return 1.0 / ( 1.0 + exp(-x));
}

fn multi_dim_sigmoid(mut x: DVector<f64>) -> DVector<f64> {
    for i in &mut x {
       *i = sigmoid(*i);
    }
    return x;
}

fn sigmoid_derivative(mut x: DVector<f64>) -> DVector<f64> {
    for i in &mut x {
        *i = sigmoid(*i) * (1.0 - sigmoid(*i));
    }
    return x
}

// TODO: How do we enforce that weights and bias
// are initialized with compatible size?
pub struct FeedForward {
    pub weights: Vec<DMatrix<f64>>,
    pub bias: Vec<DVector<f64>>, 
    size: usize,
}

impl FeedForward {
    fn gradient(
        &self,
        inputs: Vec<DMatrix<f64>>,
        outputs: Vec<DVector<f64>>,
    ) {

    }

    fn update(
        self,
        input: DVector<f64>,
        training_output: DVector<f64>,
    ) -> (Vec<DMatrix<f64>>, Vec<DVector<f64>>) {
        /// Takes a single input vector and single desired output
        /// vector as training data.  Returns the gradient of
        /// the cost function w.r.t the network's weights
        /// and biases, for this one input / output pair
        /// i.e. C(input, training_output) = 
        /// ( N(input) - training_output) / 2, 
        /// where N(x) is the function described by the network. 
        
        // Vector to store the derivatives w.r.t. bias
        let mut change_in_bias: Vec<DVector<f64>> = Vec::new();
        // Vector to store the derivatives w.r.t weights
        let mut change_in_weights: Vec<DMatrix<f64>> = Vec::new();
        // Vector to store the outputs of each layer
        let mut layer_outputs: Vec<DVector<f64>> = vec![input];
        // Vector to store the linear component of the output
        // of each layer: w \dot x + b
        let mut linear_components: Vec<DVector<f64>> = Vec::new();
        let mut derivative_wrt_output: DVector<f64>;

        // Find and store the outputs of each layer
        (layer_outputs, linear_components) = 
            self.initialize_layer_outputs(layer_outputs, linear_components);
        // Find the final derivatives -- those from the last layer
        let mut last_linear_component = &linear_components[self.size - 1];
        let penulatimate_linear_component = &linear_components[self.size - 2];
        let layer_output = &layer_outputs[self.size];
        derivative_wrt_output = layer_output
            .component_mul(
                &sigmoid_derivative(last_linear_component.clone())
            );
        change_in_bias.push(derivative_wrt_output.clone());
        change_in_weights.push(
            derivative_wrt_output.clone() * (penulatimate_linear_component.transpose())
        );

        // Find the rest of the derivatives
        for i in 1..self.size {
            let output = &layer_outputs[self.size - i];
            let linear_output = &linear_components[self.size - 1 - i];
            let ref next_weights = &self.weights[self.size - i];
            derivative_wrt_output = 
                ((*next_weights).transpose() * derivative_wrt_output).component_mul(
                    &sigmoid_derivative(linear_output.clone())
                );

            // TODO: Find a better way to do this that isn't reversing
            // these vectors a bajillion times
            change_in_weights.reverse();
            change_in_bias.reverse();

            change_in_weights.push(derivative_wrt_output.clone() * (*output).transpose());
            change_in_bias.push(derivative_wrt_output.clone());

            change_in_weights.reverse();
            change_in_bias.reverse();
        }

        return (change_in_weights, change_in_bias)
    }

    fn initialize_layer_outputs(
        &self,
        mut layer_outputs: Vec<DVector<f64>>,
        mut linear_components: Vec<DVector<f64>>,
    ) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
        for i in 0..self.size {
            // TODO: Understand how refs work
            let ref w = &self.weights[i];
            let ref b = &self.bias[i];
            let ref output = &layer_outputs[i];
            let linear_component = (*w) * (*output) + (*b);
            linear_components.push(linear_component.clone());
            layer_outputs.push(multi_dim_sigmoid(linear_component));
        }
        return (layer_outputs, linear_components)
    }
}


