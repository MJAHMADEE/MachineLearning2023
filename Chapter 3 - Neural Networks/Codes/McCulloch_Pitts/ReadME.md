# McCulloch-Pitts Neuron and Deterministic Finite Automaton (DFA)

This code implements a simple deterministic finite automaton (DFA) using McCulloch-Pitts neurons.

## Description

The code consists of two main parts:

### McCulloch-Pitts Neuron

The `McCulloch_Pitts_neuron` class defines a single McCulloch-Pitts neuron. It takes in weights and a threshold during initialization and implements a simple model to compute the output based on the input and weights.

### Deterministic Finite Automaton (DFA)

The `DFA` function creates a DFA using three McCulloch-Pitts neurons and simulates its behavior. The DFA is designed to have three outputs representing its state transition and acceptance based on the input and current state.

## Usage

To use the code:

1. Import necessary libraries (`numpy` and `itertools`).
2. Define the `McCulloch_Pitts_neuron` class with specific weights and a threshold.
3. Use the `DFA` function to simulate the DFA behavior with different inputs and states.

The provided code demonstrates the DFA behavior for a specific state and input, printing the current state, input, next state, and acceptance.

## Example usage:

```python
# Run DFA for specific inputs and states
# Create DFA instance
neur1 = McCulloch_Pitts_neuron([2, 2, -1], 2)
neur2 = McCulloch_Pitts_neuron([2, 0, 2], 2)
neur3 = McCulloch_Pitts_neuron([2, 1, -1], 2)

# Define initial states and inputs
state_b = [1, 0]
state = list(itertools.product(state_b, state_b))
input = [1, 0]
X = list(itertools.product(state, input))

# Execute DFA transitions
for i in X:
    res = DFA(i[0], i[1])
    print("DFA with current state as", str(i[0][0]) + str(" ") + str(i[0][1]), "with input as",
          str(i[1]), "goes to next state ", str(res[0]) + str(" ") + str(res[1]), " with acceptance ", str(res[2]))


