# Testing in Workflows

Testing can be challenging when not done properly, which is why we recommend a practical approach 
for testing blocks that you create. Since a block is not a standalone element in the ecosystem, testing 
might seem complex, but with the right methodology, it becomes manageable.

We suggest the following approach when adding a new block:

* **Unit tests** should cover:

    * Parsing of the manifest, especially when aliases are in use. 

    * Utility functions within the block module. If written correctly, these functions should simply 
    transform input data into output data, making them easy to test.
  
    * The `run(...)` method should be tested unit-wise only if assembling the test is straightforward.
    Otherwise, we recommend focusing on integration tests for Workflow definitions that include the block.

    * Examples can be found [here](https://github.com/roboflow/inference/tree/main/tests/workflows/unit_tests/core_steps)
  
* **Integration tests** should contain:

    * practical use cases where the block is used in collaboration with others
    
    * assertions for results, particularly for **model predictions**. These assertions should be based on empirical 
    verification, such as by visualizing and inspecting predictions to ensure they are accurate.

    * When adopting models or inference techniques from external sources (e.g., open-source models), 
    assertions should confirm that the results are consistent with what you would get outside the Workflows ecosystem, 
    ensuring compatibility and correctness.
    
    * Examples can be found [here](https://github.com/roboflow/inference/tree/main/tests/workflows/integration_tests/execution)
