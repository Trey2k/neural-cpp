#include <iostream>

#include "../dep/opennn/opennn/opennn.h"
#include "../dep/opennn/opennn/training_strategy.h"


using namespace std;
using namespace OpenNN;

int main(int argc, char **arg){
    
    DataSet data_set("../datasets/iris_flowers.csv",',',true);
    const Vector<string> inputs_names = data_set.get_input_variables_names();
    const Vector<string> targets_names = data_set.get_target_variables_names();

    data_set.split_instances_random();

    const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();

    const size_t inputs_number = 4;
    const size_t hidden_neurons_number = 6;
    const size_t outputs_number = 3;

    const Vector<size_t> architecture = {inputs_number,hidden_neurons_number,outputs_number};

    NeuralNetwork neural_network(NeuralNetwork::Classification,architecture);
    neural_network.set_inputs_names(inputs_names);
    neural_network.set_outputs_names(targets_names);

    ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
    scaling_layer_pointer->set_descriptives(inputs_descriptives);
    scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(OpenNN::TrainingStrategy::NORMALIZED_SQUARED_ERROR);
    training_strategy.set_optimization_method(OpenNN::TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);

    training_strategy.perform_training();

    QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();
    quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);
    quasi_Newton_method_pointer->set_loss_goal(1.0e-3);
    quasi_Newton_method_pointer->set_minimum_parameters_increment_norm(0.0);
    quasi_Newton_method_pointer->perform_training();

    ModelSelection model_selection(&training_strategy);
    model_selection.perform_neurons_selection();

    data_set.unscale_inputs_minimum_maximum(inputs_descriptives);
    TestingAnalysis testing_analysis(&neural_network, &data_set);

    testing_analysis.calculate_confusion();


    return 0;
}